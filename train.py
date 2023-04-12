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
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Own packages
from models.BaseModel import WoodModel

from configs import get_args_parser
from datasets import build_dataset
from optim_factory import create_optimizer

import utils

from engine import train_one_epoch

# Lightning Package
import pytorch_lightning as pl
# Timm packages
from timm import create_model
        

def main(args):
    print(args)
    # Intialize device
    device = torch.device(args.device)
    accelerator = 'cpu' if args.device == "cpu" else 'gpu'
    devices = 0 if args.device == "cpu" else 1

    #Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Build Dataset
    trainset, nb_classes = build_dataset(args, is_train = True)
    if args.disable_eval:
        valset = None
    else:
        valset, _ = build_dataset(args, is_train = False)
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        trainset, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))

    # Build DataLoader
    data_loader_train = DataLoader(
        dataset= trainset,
        sampler = sampler_train,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        pin_memory= args.pin_mem,
        drop_last = True
    )
    
    if valset is not None:
        sampler_val = torch.utils.data.SequentialSampler(valset)
        print("Sampler_val = %s" % str(sampler_val))
        
        data_loader_val = DataLoader(
            dataset = valset, 
            sampler = sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None
    
    # Visualize the dataset if set args.data_verbose
    if args.data_verbose:
        utils.Data_Visualize(data_loader_train)
        return

    criterion = torch.nn.CrossEntropyLoss()
    
    print("criterion = %s" % str(criterion))
        
    if args.finetune:
        pass
    
    model = create_model(
        model_name=args.model_name,
        pretrained=False,
        num_classes = args.nb_classes,
    )
    #model.to(device)

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
    if args.log_dir is not None:
        log_writer = utils.TBLogger(args)
    else:
        log_writer = None

    if args.enable_wandb:
        wandb_logger = utils.WdbLogger(args)
    else:
        wandb_logger = None

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list= None
    )
    
    loss_scaler = utils.NativeScalerWithGradNormCount()
    
    print("Use cosine LR Scheduler")
    #lr_scheduler_values = utils.cosine_scheduler(
    #    args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch, warmup_epochs = args.warmup_epochs, warmup_steps = args.warmup_steps
    #)
    

    max_accuracy = 0.0
    #if args.model_ema and args.model_ema_eval:
    #    max_accuracy_ema = 0.0
    
    callback, logger = utils.create_callbacks_loggers(args = args, log_writer = log_writer, wandb_logger = wandb_logger)

    model = WoodModel(args, model_name = args.model_name, criterion = criterion, optimizer = optimizer, device = device)
    model.to(device)
    trainer = pl.Trainer(
        max_epochs = args.epochs,
        accelerator= accelerator,
        devices = 1,
        #enable_model_summary= True,
        gradient_clip_val= args.clip_grad,
        #callbacks=callback,
        #logger = logger,
        #log_every_n_steps=args.log_interval,
        inference_mode=False
    )

    print("Start training for {} epochs".format(args.epochs))
    start_time = time.time()
    trainer.fit(
        model = model,
        train_dataloaders= data_loader_train,
        val_dataloaders = data_loader_val,
        
    )
    '''for epoch in range(args.start_epoch, args.epochs):
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

            if data_loader_val is not None:
                pass'''
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

         
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Wood Classification', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)