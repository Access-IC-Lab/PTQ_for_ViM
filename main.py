# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import argparse
import numpy as np
import torch

from pathlib import Path
from timm.models import create_model

from datasets import build_dataset
from engine import evaluate

from contextlib import suppress

import models_mamba

import utils

import copy

import random

# log about
import mlflow


def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # parser.add_argument('--repeated-aug', action='store_true')
    # parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    # parser.set_defaults(repeated_aug=True)
    
    # parser.add_argument('--train-mode', action='store_true')
    # parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    # parser.set_defaults(train_mode=True)
    
    # parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    # parser.add_argument('--resplit', action='store_true', default=False,
    #                     help='Do not random erase first (clean) augmentation split')
    
    # # * Cosub params
    # parser.add_argument('--cosub', action='store_true') 

    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    # parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # # if continue with inf
    # parser.add_argument('--if_continue_inf', action='store_true')
    # parser.add_argument('--no_continue_inf', action='store_false', dest='if_continue_inf')
    # parser.set_defaults(if_continue_inf=False)

    # # if use nan to num
    # parser.add_argument('--if_nan2num', action='store_true')
    # parser.add_argument('--no_nan2num', action='store_false', dest='if_nan2num')
    # parser.set_defaults(if_nan2num=False)

    # # if use random token position
    # parser.add_argument('--if_random_cls_token_position', action='store_true')
    # parser.add_argument('--no_random_cls_token_position', action='store_false', dest='if_random_cls_token_position')
    # parser.set_defaults(if_random_cls_token_position=False)    

    # # if use random token rank
    # parser.add_argument('--if_random_token_rank', action='store_true')
    # parser.add_argument('--no_random_token_rank', action='store_false', dest='if_random_token_rank')
    # parser.set_defaults(if_random_token_rank=False)

    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--quantization', action='store_true')
    parser.set_defaults(quantization=False)
    parser.add_argument('--quantization-config', default='MinMaxPTQ', type=str)
    parser.add_argument('--calibration-size', default=256, type=int)

    return parser


def main(args):
    # utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    np.random.seed(seed)
    random.seed(seed)

    # log about
    run_name = args.output_dir.split("/")[-1]
    if args.local_rank == 0 and args.gpu == 0:
        mlflow.start_run(run_name=run_name)
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

    # dataset_train, args.nb_classes = build_dataset(is_train=True, is_calib=False, args=args)
    dataset_calib, args.nb_classes = build_dataset(is_train=False, is_calib=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, is_calib=False, args=args)
    
    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        # batch_size=int(1.5 * args.batch_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # for name, m in model.named_modules():
    #     print("name:", name, "m:", m)
    # exit()

    # amp about
    amp_autocast = suppress

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')

        model.load_state_dict(checkpoint['model'])


    # Calibration & Quantization
    inds=np.random.permutation(len(dataset_calib))[:args.calibration_size]
    dataset_calib = torch.utils.data.Subset(copy.deepcopy(dataset_calib), inds)
    data_loader_calib = torch.utils.data.DataLoader(
        dataset_calib, 
        batch_size=args.calibration_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=True,
    )

    if args.quantization:
        from quantization import quantization
        model = quantization(model, data_loader_calib, config=args.quantization_config)


    # Evaluation
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, amp_autocast)
        # test_stats = evaluate(data_loader_calib, model, device, amp_autocast)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)