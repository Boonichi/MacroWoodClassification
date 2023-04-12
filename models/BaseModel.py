from optim_factory import create_optimizer

import torch
import torch.nn.functional as F
from torchmetrics import Accuracy

import pytorch_lightning as pl

from timm import create_model

import matplotlib.pyplot as plt


class WoodModel(pl.LightningModule):
    def __init__(self, args, model_name, criterion, optimizer, device):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        #self.save_hyperparameters()
        self.args = args
        self.devicce = device
        # Create Model
        self.model_name = model_name
        self.model = create_model(self.model_name, pretrained=False, num_classes = args.nb_classes)
        
        self.criterion = criterion
        self.optimizer = create_optimizer(self.args, self.model)
        self.train_acc = Accuracy(task = 'multiclass', num_classes= args.nb_classes).to(self.device)
        self.val_acc = Accuracy(task = 'multiclass', num_classes= args.nb_classes).to(self.device)
        
    
    def forward(self, sample):
        x = self.model(sample)
        
        return x
    
    def step(self, batch):
        x, y = batch
        
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        return loss, logits, y
    
    def training_step(self, batch, batch_index) -> None:
        loss, logits, targets = self.step(batch)
        preds = torch.argmax(logits, dim=1)

        acc = self.train_acc(preds, targets)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx) -> None:
        val_loss, val_logits, val_targets = self.step(batch)
        val_preds = torch.argmax(val_logits, dim = 1)
        val_acc = self.val_acc(val_preds, val_targets)

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_acc",
            val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"val_loss": val_loss}
    
    def configure_optimizers(self):
        optimizer = self.optimizer
        return {"optimizer" :optimizer }
    
    def predict(tensor, model):
        yhat = model(tensor.unsqueeze(0))
        yhat = yhat.clone().detach()
        return yhat