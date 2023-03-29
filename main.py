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
import numpy as np
import matplotlib.pyplot as plt

from configs import get_args_parser
from prepare_data import prepare_dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def main(args):
    print(args)
    # Intialize device
    device = torch.device(args.device)

    #Fix the seed for reproducibility
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
            
    # Create DataLoader
    training, val, train_dataloader, val_dataloader = create_dataloader(args)

    # Callbacks
    early_stop_callback = EarlyStopping(monitor = "val_loss", min_delta = 1e-7, patience=5, verbose = True, mode = "min")
    lr_logger = LearningRateMonitor()
    if args.target_mode == "multiple":
        logger = TensorBoardLogger(args.output_dir + "model_logs/{}_{}".format(args.station, args.model))
    else:
        logger = TensorBoardLogger(args.output_dir + "model_logs/{}_{}_{}".format(args.station, args.model, args.target))
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        enable_model_summary= True,
        gradient_clip_val= args.clip_grad,
        callbacks=[early_stop_callback, lr_logger],
        logger = logger
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser('MacroWoodClassification configs', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)