# System Packages
import argparse
import logging
from pathlib import Path
import time
import datetime
import os
import pickle

from numba.core.errors import NumbaWarning
import warnings

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Own packages
from configs import get_args_parser
from datasets import build_dataset
from optim_factory import create_optimizer
from utils import Data_Visualize 
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import TensorBoardLogger, WandbLogger
from engine import train_one_epoch


# Timm packages
from timm.models import create_model
from timm.scheduler import CosineLRScheduler

class Model():
    def __init__(self):
        pass

def main(args):
    print(args)
    # Intialize device
    device = torch.device(args.device)
    accelerator = 'cpu' if args.device == "cpu" else 'gpu'
    devices = 0 if args.device == "cpu" else 1

    #Fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Build Dataset
    trainset, nb_classes = build_dataset(args, is_train = True)
    if args.disable_eval:
        valset = None
    else:
        valset, _ = build_dataset(args, is_train = False)
    

    # Build DataLoader
    data_loader_train = DataLoader(
        dataset= trainset,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        pin_memory= args.pin_mem,
        drop_last = False
    )
    
    # Visualize the dataset if set args.data_verbose
    if args.data_verbose:
        Data_Visualize(args,trainset)
        
    model = create_model(
        args.model,
        pretrained = False,
        num_classes = args.nb_classes,
        drop_path_rate = args.drop_path,
        layer_scale_init_value = args.layer_scale_init_value,
        head_init_scale = args.head_init_scale
    )
    
    if args.finetune:
        pass

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    print('number of classes:', args.nb_classes)

    total_batch_size = args.batch_size * args.update_freq
    num_training_steps_per_epoch = len(trainset) // total_batch_size

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(trainset))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    # Logging Writer
    if args.log_write == "tensorboard":
        log_writer = TensorBoardLogger()
    elif args.wandb_logger:
        wandb_logger = WandbLogger()

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list= None
    )
    
    loss_scaler = NativeScaler()
    
    print("Use cosine LR Scheduler")
    lr_scheduler_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_epochs = args.warmup_epochs, warmup_steps = args.warmup_steps
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    print("criterion = %s" % str(criterion))

    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    print("Start training for {} epochs".format(args.epochs))
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, args.clip_grad
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(   
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

         
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Wood Classification', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)