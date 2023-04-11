import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Wood Classification', add_help=False)
    
    # Train parameters
    parser.add_argument('--input_size', default = 224, type = int,
                        help = "Input size of image")
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--num_workers', default = 8, type=int,
                        help="Number of worker in DataLoader")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Finetune paramaters:
    parser.add_argument('--finetune', action = "store_true",
                        help = "Finetuning model with exist checkpoint")
    parser.add_argument('--cpkt_dir', default = "model_logs", type = str)
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')

    # Predict parameters
    parser.add_argument('--verbose', action = "store_true",
                        help = "Display prediction from model")
    # Model parameters
    parser.add_argument('--model_name', default="resnet18", type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_ver', default = 0, type = int,
                        help ="Number of version of model")
    parser.add_argument('--config', default = None,
                        help = "Add config file that include model params")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout (Default: 0.1)')
    parser.add_argument('--clip_grad', type = float, default = 0.1, metavar="NORM",
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--hidden_size', type = int, default = 64,
                        help = "Size of hidden layer of model")
    parser.add_argument('--hidden_continuous_size', type = int, default=8,
                        help = "Size of hidden continuous layer of model")
    parser.add_argument('--attention_head', type = int, default = 4,
                        help = "Number of attention head in Transformer Architecture")
    parser.add_argument('--loss', type = str, default = "QuantileLoss",
                        help = "Loss Function (Quantile Loss, RMSE, MAE)")
    parser.add_argument('--log_interval', type = int, default = 10,
                        help = "Log Interval")
    
    # Hyperparams Optimaztion
    parser.add_argument("--param_optimize", action = "store_true",
                        help = "Find best params for model")
    # Optimization parameters
    parser.add_argument("--opt", default = "adam", type = str, metavar = 'OPTIMIZER',
                        help = "Optimizer function (ranger, adam)")
    parser.add_argument("--lr", default = 1.e-2, type = int,
                        help = "learning rate of optimizer")
    parser.add_argument("--patience", default = 3, type = int,
                        help = "Patience number for Early Stopping")
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    

    # Dataset parameters
    parser.add_argument("--nb_classes", default = 46, type = int,
                        help = "Number of classes in classification")
    parser.add_argument('--data_dir', default='./dataset/fold_0/train', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default = "image_folder", type = str)
    parser.add_argument('--eval_data_dir', default="./dataset/fold_0/val", type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--disable_eval', type=bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--device', default='mps',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--pin_mem', type=bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    
    
    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--hflip', type=float, default=0.5, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--vflip', type=float, default=0.5, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--data_mean_std', default = None, type = str,
                        help = "Normalize dataset with data mean and std (ImageNet)")
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=bool, default=False,
                        help='Do not random erase first (clean) augmentation split')
    
    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # Verbose parameters
    parser.add_argument("--data_verbose", default = False, type = bool,
                        help = "Display/Visualize data in data loader")
    
    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=bool, default=False,
                        help="enable logging to Weights and Biases")
    return parser