import torch
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adamp import AdamP

import json

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False

    
def create_optimizer(args, model, get_num_layer = None, get_layer_scale = None, skip_list = None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    skip = model.no_weight_decay()
    parameters = model.parameters()

    opt_args = dict(lr = args.lr, weight_decay = weight_decay)
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, momentum = args.momentum, nesterov= True, **opt_args)
        
    
    return optimizer