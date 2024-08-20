import argparse
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


def get_model(args, device): 
    model = DARNN(
        T = args.timesteps, 
        m = args.m_dim, 
        p = args.p_dim, 
        n = args.num_exogenous
    )
    model.to(device)
    print(f"[INFO] : Running model on device : {device}")
    return model


def train_with_config(args, opts, device):
    """
    Train the model using the provided configuration and options.
    """

    # Create checkpoint directory if it doesn't exist
    create_checkpoint_directory(opts.checkpoint)

    # Initialize TensorBoard writer
    train_writer = SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    # Load data, model, and training utilities
    train_loader, val_loader, test_loader = get_dataloader(args)
    model = get_model(args, device)
    criterion = get_criterion(device)
    optimizer, scheduler = get_optimizer(args, model)

    # Display model parameter count
    model_params = sum(p.numel() for p in model.parameters())
    print(f'[INFO]: Trainable parameter count: {model_params}')

    # Initialize minimum loss
    min_loss = np.inf

    # Main training loop
    for epoch in tqdm(range(args.epochs)):
        print(f'Training epoch {epoch}.')

        # Initialize loss trackers
        losses = initialize_loss_trackers()

        # Train for one epoch
        train_epoch(model, device, train_loader, criterion, optimizer, scheduler, losses)

        # Validate the model after each epoch
        model, gt, pred = validate_epoch(model, device, val_loader, criterion, losses, mode="val")

        # Log metrics to TensorBoard
        log_metrics(train_writer, losses, optimizer, gt, pred, epoch)

        # Save checkpoints
        save_checkpoints(opts.checkpoint, model, optimizer, scheduler, epoch, losses, min_loss)
        min_loss = min(min_loss, losses["val_RMSE_loss"].avg)

    print("[INFO] : Training done.")
    print("[INFO] : Performing inference on test data")

    # Perform inference on the test set
    model, gt, pred = validate_epoch(model, device, test_loader, criterion, losses, mode="test")

    # Log test results to TensorBoard
    log_test_results(train_writer, losses, gt, pred)


if __name__ == "__main__": 
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts, device)

