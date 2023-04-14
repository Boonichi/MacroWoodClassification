import os
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.distributed as dist

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

import wandb

def TBLogger(args):
    logger = TensorBoardLogger(args.output_dir + "{}/{}".format(args.log_dir,args.model_name))
    return logger

def WbLogger(args):
    # Intialize wandb environment
    if args.wandb_key != None:
        os.environ["WANDB_API_KEY"] = args.wandb_key

    wandb_logger = WandbLogger(
        project = "MacroWoodClassification",
        save_dir = args.log_dir,
        name = args.model_name,
    )
    wandb_logger.experiment.config.update(args)
    
    return wandb_logger

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def cosine_scheduler():
    pass
def save_model():
    pass

def Data_Visualize(dataloader):
    for index, (imgs, labels) in enumerate(dataloader):
        for img_idx, img in enumerate(imgs):
            print(img)
            plt.imshow(img.permute(1,2,0))
            plt.show()
            print(f"Label: {labels[img_idx]}")
        break

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()