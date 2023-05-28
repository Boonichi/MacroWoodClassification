import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch import optim

import pytorch_lightning as pl

from timm import create_model
from timm.scheduler import CosineLRScheduler

from optim_factory import create_optimizer

import matplotlib.pyplot as plt


class WoodModel(pl.LightningModule):
    def __init__(self, args, model_name, criterion):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        #self.save_hyperparameters()
        self.args = args

        # Create Model
        self.model_name = model_name
        self.model = create_model(self.model_name, pretrained=False, num_classes = args.nb_classes)
        
        #self.optimizer = create_optimizer(self.args, self.model)
        self.criterion = criterion
        self.train_acc = Accuracy(task = 'multiclass', num_classes= args.nb_classes).to(self.device)
        self.val_acc = Accuracy(task = 'multiclass', num_classes= args.nb_classes).to(self.device)
    
    def configure_optimizers(self):
        opt= create_optimizer(self.args, self.model)
        return opt
    
    def training_step(self,dl,idx):
        x,y=dl
        z= self.model(x)
        loss= self.criterion(z,y)
        self.log('train_loss',loss)
        return loss 
    
    def validation_step(self,dl,idx):
        x,y=dl
        z= self.model(x)
        loss= self.criterion(z,y)
        self.log('val_loss',loss)
        return loss
    

    def logging_metric(self,monitor, loss, acc):
        if monitor == "train":
            on_step = True
        else:
            on_step = False
        self.log(
            monitor + "_loss",
            loss,
            on_step = False,
            on_epoch = True,
            prog_bar = True
        )
        self.log(
            monitor + "_acc",
            acc,
            on_step = on_step,
            on_epoch = True,
            prog_bar = True
        )
