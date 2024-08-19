import os 
import random 
import argparse
import errno
import time
import copy 
from tqdm import tqdm 

from lib.models.models import DARNN 
from lib.data.data import *
from lib.utils.learning import *
from lib.utils.utils import *

import numpy as np 
import torch

from torch.utils.tensorboard import SummaryWriter


if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/basic.yaml", help="Path to the config file.")
    parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts


def get_model(args): 
    model = DARNN(
        T = args.timesteps, 
        m = args.m_dim, 
        p = args.p_dim, 
        n = args.num_exogenous
    )
    model.to(device)
    print(f"[INFO] : Running model on device : {device}")
    return model


def train_with_config(args, opts): 

    try: 
        os.mkdir(opts.checkpoint)
    except OSError as e: 
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
        
    train_writer = SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    train_loader, val_loader, test_loader = get_dataloader(args)
    model = get_model(args)
    criterion = get_criterion(args)
    optimizer, scheduler = get_optimizer(args, model)

    min_loss = np.inf
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    for epoch in tqdm(range(args.epochs)): 
        print('Training epoch %d.' % epoch)

        losses = {}
        losses["train_MSE_loss"] = AverageMeter()
        losses["val_MSE_loss"] = AverageMeter()
        losses["val_MAE_loss"] = AverageMeter()
        losses["val_RMSE_loss"] = AverageMeter()
        losses["val_MAPE_loss"] = AverageMeter()
        losses["test_MSE_loss"] = AverageMeter()
        losses["test_MAE_loss"] = AverageMeter()
        losses["test_RMSE_loss"] = AverageMeter()
        losses["test_MAPE_loss"] = AverageMeter()
        train_epoch(args, opts, model, train_loader, criterion, optimizer, scheduler, losses, epoch) 
        model, gt, pred = validate_epoch(args, opts, model, val_loader, criterion, losses, epoch, mode="val")
        
        # logs
        lr = optimizer.param_groups[0]['lr']
        train_writer.add_scalar("train_MSE_loss", losses["train_MSE_loss"].avg, epoch + 1)
        train_writer.add_scalar("val_MSE_loss", losses["val_MSE_loss"].avg, epoch + 1)
        train_writer.add_scalar("lr", lr, epoch + 1)
        for i in range(len(gt)):

            train_writer.add_scalars(
                "validation_prediction", 
                {
                    "gt": gt[i], 
                    "pred": pred[i]
                }, 
                i
            )
        chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
        chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))

        save_checkpoint(chk_path_latest, epoch, lr, optimizer, scheduler, model, min_loss)
        if losses["val_RMSE_loss"].avg < min_loss: 
            min_loss = losses["val_RMSE_loss"].avg
            save_checkpoint(chk_path_best, epoch, lr, optimizer, scheduler, model, min_loss)

    print("[INFO] : Training done. ")
    print("[INFO] : Performing inference on test data") 
    model, gt, pred = validate_epoch(args, opts, model, test_loader, criterion, losses, epoch, mode="test")
    for i in range(len(gt)):
        train_writer.add_scalars(
            "test_prediction", 
            {
                "gt": gt[i], 
                "pred": pred[i]
            }, 
            i
        )


    train_writer.add_scalar("test_MSE_loss", losses["test_MSE_loss"].avg, 0)
    train_writer.add_scalar("test_MAE_loss", losses["test_MAE_loss"].avg, 0)
    train_writer.add_scalar("test_RMSE_loss", losses["test_RMSE_loss"].avg, 0)
    train_writer.add_scalar("test_MAPE_loss", losses["test_MAPE_loss"].avg, 0)
 

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__": 
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)