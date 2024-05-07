import sys
import os
print(sys.path)
import math
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import socket
import torch.distributed as dist
import time
from datetime import datetime
import random

from model_pl_survival import DeepNN_Model

def main(hparams):
    print(hparams)
    torch.cuda.empty_cache()
    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    # Set region
    regions = ['Center', 'Periphery', 'R1', 'TNormal']
    if hparams.area in regions:
        hparams.npz_region_indices = hparams.npz_region_indices + '/' + hparams.area + '/index'

    # Create the model
    model = DeepNN_Model(hparams)
    ck_point_path = None
    estop_callback = EarlyStopping(monitor = 'val_loss', mode ='min', min_delta = 0.0000, patience = 20, verbose = True)
    chp_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min",)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir = 'lightning_logs_WSI_znormal_region_mfs_5years/')

    if ck_point_path == None:
        print("New training on {} epochs, {} labels".format(hparams.epochs, hparams.num_classes))
        trainer = pl.Trainer(max_epochs=hparams.epochs, \
                            weights_summary='top', \
                            gpus = hparams.n_gpus, \
                            #accelerator = 'gpu', \
                            #strategy= "ddp", \
                            #amp_level = "O2", \
                            precision = 16, \
                            num_sanity_val_steps = 0, \
                            callbacks = [chp_callback, estop_callback],
                            logger = tb_logger,
                            )
    else:
        print("Training from a checkpoint to {} epochs, {} labels".format(hparams.epochs, hparams.num_classes))
        trainer = pl.Trainer(resume_from_checkpoint = ck_point_path, max_epochs=hparams.epochs, \
                            weights_summary='top', \
                            gpus = hparams.n_gpus, \
                            accelerator = 'gpu', \
                            strategy= "ddp", \
                            #amp_level = "O2", \
                            precision = 16, \
                            callbacks = [chp_callback],)
    trainer.fit(model)

if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    main_arg_parser = argparse.ArgumentParser(description="parser for observation generator", add_help=False)
    main_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    main_arg_parser.add_argument("--checkpoint-interval", type=int, default=500,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    
    main_arg_parser.add_argument('--epochs', default = 500, type=int)
    main_arg_parser.add_argument('--n_gpus', default = 1, type=int)
    main_arg_parser.add_argument('--learning_rate', default=0.003, type=float)
    main_arg_parser.add_argument('--w_decay', default=1e-04, type=float)
    main_arg_parser.add_argument('--batch_size', default=1, type=int)

    main_arg_parser.add_argument('--npz_train', default = '/beegfs/vle/bergonie/data/WSI_features_znormal', type=str)
    main_arg_parser.add_argument('--npz_region_indices', default = '/beegfs/vle/bergonie/data/All_regions_features_znormal', type=str) # Clusters_features Centroids_77 selection
    main_arg_parser.add_argument('--area', default = 'PR1', type=str)

    #
    # Overall survival: survival_data/overall/OS.csv (OS_muscle.csv, OS_fibro.csv)
    # MFS survival: survival_data/mfs/MFS_220.csv (MFS_muscle.csv, MFS_fibro.csv)
    # LRS survival: survival_data/lrs/LRS.csv (LRS_muscle.csv, LRS_fibro.csv)
    #
    main_arg_parser.add_argument('--npz_labels', default = '/beegfs/vle/bergonie/data/csv/survival_data/mfs/MFS_220_metas_event_5years.csv', type=str)
    main_arg_parser.add_argument('--init_features', default = 2048, type=int)

    main_arg_parser.add_argument('--seed', default = 2452, type=int)
    main_arg_parser.add_argument('--n_tiles', default = 10000, type=int)

    main_arg_parser.add_argument('--fc_1', default = 256, type=int)
    main_arg_parser.add_argument('--fc_2', default = 128, type=int)
    main_arg_parser.add_argument('--num_classes', default = 1, type=int)
    main_arg_parser.add_argument('--val_per', default=0.3, type=float) # % data for validation
    
    # add model specific args i
    parser = DeepNN_Model.add_model_specific_args(main_arg_parser, os.getcwd())
    hyperparams = parser.parse_args()

    main(hyperparams)

