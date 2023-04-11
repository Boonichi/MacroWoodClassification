import torch
import torch.nn as nn

import pytorch_lightning as pl

from timm.models import create_model

import torch.nn.functional as F
from torchmetrics import Accuracy


class WoodModel(pl.LightningModule):
    def __init__(self, args, model_name, criterion, optimizer, dropout):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        #self.save_hyperparameters()
        self.args = args
        # Create Model
        self.model_name = model_name
        self.model = create_model(self.model_name, pretrained=False, num_classes = args.nb_classes)
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric = Accuracy(task = 'multiclass', num_classes= args.nb_classes).to(args.device)

        self.test_preds = []
    
    def forward(self, sample):
        x = self.model(sample)
        
        return x
    
    def training_step(self, batch, batch_index):
        
        sample, target = batch
        
        logit = self(sample)

        loss = self.criterion(logit, target)

        preds = torch.argmax(logit, dim=1)

        acc = self.metric(preds, target)    

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_epoch = True, on_step=True, prog_bar = True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        sample, target = batch
        
        logit = self(sample)

        loss = self.criterion(logit, target)

        preds = torch.argmax(logit, dim=1)

        acc = self.metric(preds, target)        
        
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("val_acc", acc, on_epoch = True, prog_bar = True)

        return loss
    
    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer
    
    def predict(tensor, model):
        yhat = model(tensor.unsqueeze(0))
        yhat = yhat.clone().detach()
        return yhat