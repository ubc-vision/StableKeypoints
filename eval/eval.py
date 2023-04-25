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

from diffusers import StableDiffusionPipeline, DDIMScheduler

from optimize_token import load_ldm

import wandb


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='CATs Test Script')

    # Dataset
    parser.add_argument('--datapath', type=str, default='../Datasets_CATs')
    parser.add_argument('--benchmark', type=str,
                        choices=['pfpascal', 'spair', 'pfwillow'], default='spair')
    parser.add_argument('--thres', type=str, default='auto',
                        choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='alpha for the pck threshold')
    parser.add_argument('--sub_class', type=str, default="all", choices=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                                                                         'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor', 'all'])
    parser.add_argument('--split', type=str, default="test",
                        choices=['test', 'trn', 'val'])
    parser.add_argument('--item_index', type=int, default=-1)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='training batch size')

    # Hyperparameters
    parser.add_argument('--num_steps', type=int, default=129)
    parser.add_argument('--noise_level', type=int, default=-8,
                        help='noise level for the test set between 0 and 49 where 0 is the highest noise level and 49 is the lowest noise level')
    parser.add_argument('--flip_prob', type=float, default=0.36270331047928606,
                        help='probability of flipping the image during optimization')
    parser.add_argument('--sigma', type=float, default=27.97853316316864,
                        help='sigma for the gaussian kernel')
    parser.add_argument('--layers', type=int, nargs='+', default=[5, 6, 7, 8])
    parser.add_argument('--learning_rate', type=float,
                        default=0.0023755632081200314, help='learning rate for the optimizer')
    parser.add_argument('--crop_percent', type=float, default=93.16549294381423,
                        help='the percent of the image to crop to')
    parser.add_argument('--num_iterations', type=int, default=5,
                        help='number of iterations to run the optimization for')

    # Network details
    parser.add_argument('--model_type', type=str,
                        default='CompVis/stable-diffusion-v1-4', help='ldm model type')
    parser.add_argument('--upsample_res', type=int, default=512,
                        help='Resolution to upsample the attention maps to')
    parser.add_argument('--num_words', type=int, default=2)

    # Run details
    parser.add_argument('--wandb_log', action='store_true',
                        help='whether to use wandb for logging')
    parser.add_argument('--device', type=str,
                        default='cuda:0', help='device to use')
    parser.add_argument('--wandb_name', type=str,
                        default='test', help='name of the wandb run')
    parser.add_argument('--mode', type=str, default="optimize", choices=[
                        "train", "evaluate", "optimize", "retest"], help='whether to train, validate, or optimize the model')
    parser.add_argument('--ablate_results', action='store_true', help='whether to ablate the results')
    parser.add_argument('--visualize', action='store_true',
                        help='whether to visualize the attention maps')
    parser.add_argument('--epoch', type=int, default=0,
                        help='what epoch of the model to load')
    parser.add_argument('--save_loc', type=str, default='outputs',
                        help='save location for the trained model')
    parser.add_argument('--seed', type=int, default=2023,
                        help='Pseudo-RNG seed')
    parser.add_argument('--ablate', action='store_true',
                        help='evaluate over a smaller number of points')

    args = parser.parse_args()
    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb_log:
        # initialize wandb
        wandb.init(project="estimated_correspondences",
                   name=f"{args.wandb_name}")
        wandb.config.update(args)

    # Initialize Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)

    # Dataloader
    download.download_dataset(args.datapath, args.benchmark)
    if args.mode == "optimize":
        
        test_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device,
                                            args.split, False, 16, sub_class=args.sub_class, item_index=args.item_index)
    else:
        test_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device,
                                            args.split, False, 16, sub_class=args.sub_class, item_index=-1)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=0,
                                 shuffle=True)
    
    # optimize.rewrite_idxs()
    # exit()

    # results = test_dataset.collect_results()
    # results is a dict with values being lists
    # import ipdb ; ipdb.set_trace()
    # this_avg = []
    # for key in results.keys():
    #     print(key, sum(results[key])/len(results[key]))
    #     this_avg.append(sum(results[key])/len(results[key]))

    # overal_avg = sum(this_avg)/len(this_avg)

    # print("overall average", overal_avg)
    # exit()

    # initialize model
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    ldm = load_ldm(args.device, args.model_type)
    
    # if args.save_loc doesnt exist, create it
    if not os.path.exists(args.save_loc):
        os.makedirs(args.save_loc)
        
    train_started = time.time()

    if args.mode == "evaluate" or args.mode == "optimize":
        print("validating")
        pck_array = optimize.validate_epoch(ldm,
                                            test_dataloader,
                                            num_steps=args.num_steps,
                                            noise_level=args.noise_level,
                                            upsample_res=args.upsample_res,
                                            layers=args.layers,
                                            num_words=args.num_words,
                                            device=args.device,
                                            visualize=args.visualize,
                                            epoch=args.epoch,
                                            optimize=args.mode == "optimize",
                                            lr=args.learning_rate,
                                            wandb_log=args.wandb_log,
                                            sigma=args.sigma,
                                            flip_prob=args.flip_prob,
                                            num_iterations=args.num_iterations,
                                            crop_percent=args.crop_percent,
                                            save_folder = args.save_loc,
                                            item_index = args.item_index,
                                            ablate = args.ablate,)
        if args.item_index != -1:
            # save the pck array to a text file
            np.savetxt(
                f"{args.save_loc}/pck_array_{args.item_index:06d}.txt", pck_array)

        print('Test took:', time.time()-train_started, 'seconds')
    elif args.mode == "train":
        print("training")
        val_loss_grid, val_mean_pck = optimize.train(ldm,
                                                     test_dataloader,
                                                     num_steps=args.num_steps,
                                                     noise_level=args.noise_level,
                                                     upsample_res=args.upsample_res,
                                                     layers=args.layers,
                                                     num_words=args.num_words,
                                                     wandb_log=args.wandb_log,
                                                     device=args.device,
                                                     save_loc=args.save_loc,
                                                     learning_rate=args.learning_rate,)
    elif args.mode == "retest":
        print("retesting")
        pck_array = optimize.retest(ldm,
                                    test_dataset,
                                    num_steps=args.num_steps,
                                    noise_level=args.noise_level,
                                    upsample_res=args.upsample_res,
                                    layers=args.layers,
                                    num_words=args.num_words,
                                    device=args.device,
                                    visualize=args.visualize,
                                    epoch=args.epoch,
                                    optimize=args.mode == "optimize",
                                    # lr=args.learning_rate,
                                    wandb_log=args.wandb_log,
                                    crop_percent=args.crop_percent,
                                    # sigma=args.sigma,
                                    # flip_prob=args.flip_prob,
                                    # num_iterations=args.num_iterations,
                                    item_index=args.item_index,
                                    save_folder = args.save_loc,
                                    ablate_results = args.ablate_results,)
    else:
        raise ValueError("mode must be one of train, evaluate, or optimize")
