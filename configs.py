import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('MacroWoodClassification Configs', add_help=False)
    
    # Train parameters
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

    # Predict parameters
    parser.add_argument('--test', action = "store_true",
                        help = "Test Process")
    parser.add_argument('--verbose', action = "store_true",
                        help = "Display prediction from model")
    # Model parameters
    parser.add_argument('--model', default='TFT', type=str, metavar='MODEL',
                        help='Name of model to train')
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
    parser.add_argument("--opt", default = "ranger", type = str, metavar = 'OPTIMIZER',
                        help = "Optimizer function (ranger, adam)")
    parser.add_argument("--lr", default = 1.e-2, type = int,
                        help = "learning rate of optimizer")
    parser.add_argument("--patience", default = 3, type = int,
                        help = "Patience number for Early Stopping")
    

    # Dataset parameters
    parser.add_argument('--data_dir', default='./dataset/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='mps',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--max_encoder_day', default = 7, type = int)
    parser.add_argument('--max_pred_day', default = 30, type = int)

    return parser