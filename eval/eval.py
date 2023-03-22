r'''
    modified test script of GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''

import argparse
import os
import pickle
import random
import time
from os import path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from termcolor import colored
from torch.utils.data import DataLoader

import utils_training.optimize as optimize
from utils_training.evaluation import Evaluator
from utils_training.utils import parse_list, log_args, load_checkpoint, save_checkpoint, boolean_string
from eval import download

from optimize_token import load_ldm


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='CATs Test Script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--snapshots', type=str, default='./eval')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=8,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Pseudo-RNG seed')
                        
    parser.add_argument('--datapath', type=str, default='../Datasets_CATs')
    parser.add_argument('--benchmark', type=str, choices=['pfpascal', 'spair', 'pfwillow'], default='spair')
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--noise_level', type=int, default=1, help='noise level for the test set between 1000 and 1 where 1000 is the highest noise level and 1 is the lowest noise level')
    parser.add_argument('--upsample_res', type=int, default=32, help='Resolution to upsample the attention maps to')
    parser.add_argument('--layers', type=int, nargs='+', default= [6, 7, 8, 9, 10])
    parser.add_argument('--num_words', type=int, default= 2)

    # Seed
    args = parser.parse_args()
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)
    
    # Dataloader
    download.download_dataset(args.datapath, args.benchmark)
    test_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'test', False, 16)
    test_dataloader = DataLoader(test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True)
    
    
    # initialize model
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    ldm, _ = load_ldm(device)

    train_started = time.time()

    val_loss_grid, val_mean_pck = optimize.validate_epoch(ldm,
                                                    test_dataloader,
                                                    device,
                                                    epoch=0,
                                                    num_steps = args.num_steps,
                                                    noise_level = args.noise_level,
                                                    upsample_res=args.upsample_res,
                                                    layers = args.layers,
                                                    num_words=args.num_words,)
    print(colored('==> ', 'blue') + 'Test average grid loss :',
            val_loss_grid)
    print('mean PCK is {}'.format(val_mean_pck))

    print(args.seed, 'Test took:', time.time()-train_started, 'seconds')
