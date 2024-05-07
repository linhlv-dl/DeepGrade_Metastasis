import sys
import os
print(sys.path)

import math
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from model_pl_survival_3 import DeepNN_Model
from pytorch_lightning import loggers as pl_loggers

import numpy as np
import socket
import torch.distributed as dist
import time
import random
from datetime import datetime
from bergonie_dataloader_survival_wsi import get_data_cross_validation_KFold, get_data_cross_validation_SKFold, get_data_from_cross_validation

def cross_validation(hparams):
    # functions from bergonie_dataloader_survival_wsi
    if hparams.m_get_data == 'SKFold':
        print("Get data by SKFold CV.")
        train_patients_dict, valid_patients_dict, patients_wsi = get_data_cross_validation_SKFold(hparams.npz_labels, 
                                                                        hparams.npz_train,
                                                                        hparams.npz_region_indices,
                                                                        k_folds = hparams.n_folds,
                                                                        seed = hparams.seed)
    elif hparams.m_get_data == 'KFold':
        print("Get data by KFold CV.")
        train_patients_dict, valid_patients_dict, patients_wsi = get_data_cross_validation_KFold(hparams.npz_labels, 
                                                                        hparams.npz_train,
                                                                        hparams.npz_region_indices,
                                                                        k_folds = hparams.n_folds,
                                                                        seed = hparams.seed)

    region = 'Peripery'
    task = 'MFS'
    torch.save(valid_patients_dict, 'exports/MFS_5years/random_SEED_{}_10K_tiles_model2a_config1_5years_{}_folds_'.format(hparams.seed, hparams.n_folds) + datetime.now().strftime("%Y%m%d%H%M%S") + '.ckpt')
    #print(valid_patients_dict)
    assert(len(train_patients_dict) == len(valid_patients_dict))
    
    for idx in range(hparams.n_folds):
        train_dict = train_patients_dict['fold_' + str(idx)]
        valid_dict = valid_patients_dict['fold_' + str(idx)]
        
        train_loader_idx, valid_loader_idx = get_data_from_cross_validation(hparams.npz_train,
                                                                            hparams.npz_region_indices,
                                                                            train_dict,
                                                                            valid_dict, 
                                                                            patients_wsi,
                                                                            n_tiles = hparams.n_tiles, 
                                                                            batch_size = hparams.batch_size, 
                                                                            seed = hparams.seed)
        
        #train_loader_idx, valid_loader_idx = 1, 1
        model = DeepNN_Model(hparams, train_loader_idx, valid_loader_idx)
        estop_callback = EarlyStopping(monitor = 'val_loss', mode ='min', min_delta = 0.0000, patience = 20, verbose = True)
        chp_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min",)
        #tb_logger = pl_loggers.TensorBoardLogger(save_dir = 'lightning_logs_OS/cv_5_new_R1_TNormal/OS/Muscle/{}/'.format(hparams.area))
        tb_logger = pl_loggers.TensorBoardLogger(save_dir = 'lightning_logs_WSI_znormal_region/{}/{}/cv_model2a_2452_skfold_5years/'.format(region, task))
        trainer = pl.Trainer(max_epochs=hparams.epochs, \
                            weights_summary='top', \
                            gpus = hparams.n_gpus, \
                            #accelerator = 'gpu', \
                            #strategy= "ddp", \
                            #amp_level = "O1", \
                            precision = 16, \
                            callbacks = [chp_callback, estop_callback],
                            num_sanity_val_steps = 0,
                            #replace_sampler_ddp=False,
                            logger = tb_logger,)
        trainer.fit(model, train_loader_idx, valid_loader_idx)
        
def main(hparams):
    # use a random seed
    if hparams.seed == -1:
        hparams.seed = random.randint(0,5000)
        print('The SEED number was randomly set to {}'.format(hparams.seed))

    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    print(hparams)
    torch.cuda.empty_cache()
    cross_validation(hparams)
    
if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    main_arg_parser = argparse.ArgumentParser(description="parser for observation generator", add_help=False)
    main_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    main_arg_parser.add_argument("--checkpoint-interval", type=int, default=500,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    main_arg_parser.add_argument('--epochs', default = 200, type=int)
    main_arg_parser.add_argument('--n_gpus', default = 1, type=int)
    main_arg_parser.add_argument('--learning_rate', default=0.003, type=float)
    main_arg_parser.add_argument('--w_decay', default=1e-4, type=float)
    main_arg_parser.add_argument('--batch_size', default=1, type=int)
    main_arg_parser.add_argument('--n_folds', default= 5, type=int)

    main_arg_parser.add_argument('--npz_train', default = '/media/monc/Disk2/Data/Bergonie_features/WSI_features_znormal', type=str)
    main_arg_parser.add_argument('--npz_region_indices', default = '/media/monc/Disk2/Data/Bergonie_features/All_regions_features_znormal/Periphery/index', type=str) # Clusters_features Centroids_77 selection
    #
    # Overall survival: survival_data/overall/OS_new.csv (OS_muscle.csv, OS_fibro.csv) OS_new.csv
    # MFS survival: survival_data/mfs/MFS_220.csv (MFS_muscle.csv, MFS_fibro.csv) MFS_220_metas_event.csv
    # LRS survival: survival_data/lrs/LRS.csv (LRS_muscle.csv, LRS_fibro.csv) LRS_rc_event.csv
    #
    main_arg_parser.add_argument('--npz_labels', default = '/bergonie_data/Bergonie/csv/survival_data/mfs/MFS_220_metas_event_5years.csv', type=str)
    main_arg_parser.add_argument('--init_features', default = 2048, type=int)
    main_arg_parser.add_argument('--m_get_data', default = 'SKFold', type=str) # KFold or SKFold

    main_arg_parser.add_argument('--seed', default = 2452, type=int)
    main_arg_parser.add_argument('--n_tiles', default = 10000, type=int)

    main_arg_parser.add_argument('--fc_1', default = 256, type=int)
    main_arg_parser.add_argument('--fc_2', default = 128, type=int)
    main_arg_parser.add_argument('--num_classes', default = 1, type=int) 
    
    # add model specific args i
    parser = DeepNN_Model.add_model_specific_args(main_arg_parser, os.getcwd())
    hyperparams = parser.parse_args()

    main(hyperparams)

