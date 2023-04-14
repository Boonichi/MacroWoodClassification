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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
# Timm packages
from timm import create_model
from timm.loss import LabelSmoothingCrossEntropy
        

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
    
    model_without_ddp = create_model(
        model_name=args.model_name,
        pretrained=False,
        num_classes = args.nb_classes,
    )
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)

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

    # Logging Writer (TensorBoard/ Wandb)
    logger = []
    if args.log_dir is not None:
        log_writer = utils.TBLogger(args)
        logger.append(log_writer)
    else:
        log_writer = None

    if args.enable_wandb:
        if not args.wandb_key:
            raise AssertionError("Please add your Wandb API key")
        else:
            wandb_logger = utils.WbLogger(args)
            logger.append(wandb_logger)
    else:
        wandb_logger = None

    # Criterion
    if args.smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing = args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    print("criterion = %s" % str(criterion))
        
    # Loss Scaler
    loss_scaler = utils.NativeScalerWithGradNormCount()
    
    # Learning Rate Scheduler
    print("Use cosine LR Scheduler")
    
    # Callbacks, Logger
    early_stop_callback = EarlyStopping(monitor = "val_loss", min_delta = 1e-7, patience = args.patience, verbose = True, mode = "min")
    model_checkpoint = ModelCheckpoint(monitor = "val_loss", save_last = True, save_top_k = 1, mode = "min", verbose = True)
    lr_monitor = LearningRateMonitor(logging_interval = "epoch")
    
    # Create Model
    model = WoodModel(args, model_name = args.model_name, 
                      criterion = criterion, 
                      )
    

    # Setup Trainer
    trainer = pl.Trainer(
        max_epochs = args.epochs,
        accelerator= accelerator,
        devices = 1,
        enable_model_summary= False,
        #gradient_clip_val= args.clip_grad,
        callbacks=[early_stop_callback, model_checkpoint, lr_monitor],
        logger = logger,
    )

    # Finetune Process
    if args.finetune:
        wandb_logger.init(resume = True)
        path = args.output_dir + "model_logs/{}".format(args.model_name) + "/lightning_logs/"
        newest_version = max([os.path.join(path,d) for d in os.listdir(path) if d.startswith("version")], key=os.path.getmtime) + "/checkpoints"
        if args.finetune == "last":
            checkpoint = os.listdir(newest_version)[1]
        else: 
            checkpoint = os.listdir(newest_version)[0]
        
        model = model.load_from_checkpoint(checkpoint)

    model.to(device)

    # Train Process
    print("Start training for {} epochs".format(args.epochs))
    start_time = time.time()
    trainer.fit(
        model = model,
        train_dataloaders= data_loader_train,
        val_dataloaders = data_loader_val,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

         
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Wood Classification', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)