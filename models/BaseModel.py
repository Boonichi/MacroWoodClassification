import torch
import torch.nn.functional as F
from torchmetrics import Accuracy

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
        
        self.optimizer = create_optimizer(self.args, self.model)
        self.criterion = criterion
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

        # Loss Backward
        loss, logits, targets = self.step(batch)

        # Predicts, Accuracy
        preds = torch.argmax(logits, dim=1)

        train_acc = self.train_acc(preds, targets)

        self.logging_metric(monitor = "train", loss = loss, acc = train_acc)

        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx) -> None:
        val_loss, val_logits, val_targets = self.step(batch)
        val_preds = torch.argmax(val_logits, dim = 1)
        val_acc = self.val_acc(val_preds, val_targets)
        
        self.logging_metric(monitor = "val", loss = val_loss, acc = val_acc)

        return {"loss": val_loss}
    
    def configure_optimizers(self):
        optimizer = self.optimizer
        lr_scheduler = CosineLRScheduler(
            optimizer = optimizer,
            t_initial = self.args.epochs,
            lr_min = self.args.min_lr,
            warmup_t = self.args.warmup_epochs
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def logging_metric(self,monitor, loss, acc):
        
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
            on_step = False,
            on_epoch = True,
            prog_bar = True
        )
