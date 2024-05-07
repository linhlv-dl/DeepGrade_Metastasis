import os
import torch
import torch.utils.data as data_utils
from torch.nn import functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
import sklearn.metrics as sm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_attention_survival import NN_Model2a
import bergonie_dataloader_survival_wsi as pro
from loss_2 import NegativeLogLikelihood, c_index
from datetime import datetime

class DeepNN_Model(pl.LightningModule):
    def __init__(self, hparams, train_loader = None, valid_loader = None):
        super().__init__()
        
        self.save_hyperparameters(hparams)
        self.learning_rate = hparams.learning_rate
        print("global seed: ", os.environ.get("PL_GLOBAL_SEED"))
        pl.seed_everything(hparams.seed)
        
        ## DataLoader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        if self.train_loader == None and self.valid_loader == None:
            print("Get data because no loader provided....")
            self.train_loader, self.valid_loader, self.valid_patients_dict = pro.get_data_npz_folder(hparams.npz_train, 
                                                            hparams.npz_labels, 
                                                            hparams.npz_region_indices,
                                                            n_tiles = hparams.n_tiles,
                                                            batch_size = hparams.batch_size, 
                                                            val_per = hparams.val_per,
                                                            seed = hparams.seed,
                                                            gregion = hparams.area)
            # To save the list of validation patients
            torch.save(self.valid_patients_dict, 'exports/validation_patients_MFS_5years_{}_'.format(hparams.area) + datetime.now().strftime("%Y%m%d%H%M%S") + '.ckpt')

        self.model = NN_Model2a(in_features = hparams.init_features,
                    fc_1= hparams.fc_1, 
                    fc_2 = hparams.fc_2, 
                    fc_output = hparams.num_classes)

        self.loss_f = NegativeLogLikelihood()
        #self.iter = 0
        self.lbl_preds = []
        self.survtime_all = []
        self.status_all = []
        self.automatic_optimization = False

    # Delegate forward to underlying model
    def forward(self, x):
        y_prob, x_att = self.model(x)
        return y_prob, x_att

    # Train on one batch
    def training_step(self, batch, batch_idx):
        x, y_status, y_survtime = batch
        y_prob, x_att = self.forward(x)

        # (risk_pred, y, e, model)
        loss = self.loss_f(y_prob, y_survtime, y_status, self.model)

        tensorboard_logs = {'train_loss':loss}
        self.log('train_loss', loss)

        return {'loss': loss, 
                'log': tensorboard_logs,
                'time': y_survtime,
                'status': y_status,
                'pred_score': y_prob, 
                'batch_idx': batch_idx}
    
    def training_step_end(self, train_step_output):
        y_survtime = train_step_output['time']
        y_status = train_step_output['status']
        y_prob = train_step_output['pred_score']
        batch_idx = train_step_output['batch_idx']
        self.survtime_all.append(y_survtime)
        self.status_all.append(y_status)

        if batch_idx == 0:
            self.lbl_preds = y_prob
        else:
            self.lbl_preds = torch.cat([self.lbl_preds, y_prob])

        if batch_idx == len(self.train_loader) - 1:
            self.survtime_all = torch.stack(self.survtime_all)
            self.status_all = torch.stack(self.status_all)
            self.survtime_all = self.survtime_all.view(self.survtime_all.size(0), -1)
            self.status_all = self.status_all.view(self.status_all.size(0), -1)
            print('Compute the loss....', self.survtime_all.shape)
            
            loss = self.loss_f(self.lbl_preds, self.survtime_all, self.status_all, self.model)
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss, retain_graph=True)
            opt.step()

            torch.cuda.empty_cache()
            self.lbl_preds = []
            self.survtime_all = []
            self.status_all = []
            self.log('train_loss', loss)
            return {'loss': loss,
                    'time': y_survtime.detach(),
                    'status': y_status.detach(),
                    'pred_score': y_prob.detach(),}

    def training_epoch_end(self, training_step_outputs):
        y_trues = [x['status'].detach().cpu().numpy() for x in training_step_outputs]
        y_survtimes = [x['time'].detach().cpu().numpy() for x in training_step_outputs]
        y_preds = [x['pred_score'].detach().cpu().numpy() for x in training_step_outputs]

        # To train the survival estimator
        y_trues = np.asarray(y_trues).reshape(-1)
        y_survtimes = np.asarray(y_survtimes).reshape(-1)
        y_preds = np.asarray(y_preds).reshape(-1)
    
    # Validate on one batch
    
    def validation_step(self, batch, batch_idx):
        x, y_status, y_survtime = batch
        y_prob, x_att = self.forward(x)

        return {'time': y_survtime,
                'status': y_status,
                'pred_score': y_prob}   

    def validation_epoch_end(self, outputs):
        y_trues = [x['status'] for x in outputs]
        y_preds = [x['pred_score'] for x in outputs]
        y_survtimes = [x['time'] for x in outputs]

        y_trues = torch.stack(y_trues)
        y_preds = torch.stack(y_preds)
        y_survtimes = torch.stack(y_survtimes)

        y_preds = y_preds.view(y_preds.size(0), -1)
        y_survtimes = y_survtimes.view(y_survtimes.size(0), -1)
        y_trues = y_trues.view(y_trues.size(0), -1)

        avg_loss = self.loss_f(y_preds, y_survtimes, y_trues, self.model)

        cindex = c_index(-y_preds, y_survtimes, y_trues)

        tensorboard_logs = {'loss': avg_loss, 
                            'c-index': cindex}
                            #'auc':auc_score}

        self.log('val_loss',avg_loss)
        self.log('val_cindex',cindex)
        sch = self.lr_schedulers()
        sch.step(self.trainer.callback_metrics["val_loss"])
        
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # Setup optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate, weight_decay = self.hparams.w_decay)
        return optimizer
        #lr_schedulers = {"scheduler": ReduceLROnPlateau(optimizer, patience = 3, factor = 0.5, verbose = True), "monitor": "val_loss"}
        #scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 20, verbose = True)
        #return [optimizer], [lr_schedulers]
    
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--name', default='Deep NN model for survival', type=str)
        return parser


