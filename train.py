import argparse
from tqdm import tqdm 

from lib.models.models import DARNN 
from lib.data.data import *
from lib.utils.learning import *
from lib.utils.utils import *

from torch.utils.tensorboard import SummaryWriter


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/basic.yaml", help="Path to the config file.")
    parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
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
    return model


def train_with_config(args, opts, device, mode='val'):
    """
    Train the model using the provided configuration and options.
    """
    print('--------------------------------------------')
    print('######## INITIATING DARNN TRAINING ########')
    print('--------------------------------------------')
    
    # Create checkpoint directory if it doesn't exist
    create_checkpoint_directory(opts.checkpoint)

    # Initialize TensorBoard writer
    tb_writer = SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    # Load data, model, and training utilities
    train_loader, val_loader = get_dataloader(args, mode)
    model = get_model(args, device)
    criterion = get_criterion(device)
    optimizer, scheduler = get_optimizer(args, model)
    min_loss = np.inf

    # Display settings
    model_params = sum(p.numel() for p in model.parameters())
    print(f'[INFO MODEL] Trainable parameter count: {model_params} | Device: {device}')

    for epoch in tqdm(range(args.epochs)):
        print(f'[INFO TRAIN] Epoch {epoch}.')

        # Initialize loss trackers
        losses = initialize_loss_trackers(mode)

        # Train for one epoch
        train_epoch(model, device, train_loader, criterion, optimizer, scheduler, losses)

        # Validate the model after each epoch
        model, gt, pred = evaluate_model(model, device, val_loader, criterion, losses, mode)

        # Log metrics to TensorBoard
        log_metrics(tb_writer, losses, optimizer, gt, pred, epoch)

        # Checkpoints
        manage_checkpoints(opts.checkpoint, model, optimizer, scheduler, epoch, losses, min_loss)

        # Update min loss
        min_loss = min(min_loss, losses["val_RMSE_loss"].avg)

    print("[INFO] Training completed.")


def test_model(args, opts, device, mode='test'):
    """
    Perform inference on the test set using the trained model.
    """
    print('--------------------------------------------')
    print('######## STARTING TEST INFERENCE ########')
    print('--------------------------------------------')

    # Load test data
    test_loader = get_dataloader(args, mode)
    
    # Load the model
    model = get_model(args, device)
    checkpoint = torch.load(os.path.join(opts.checkpoint, 'best_epoch.bin'), weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO MODEL] Best Model Loaded | Epoch: {checkpoint['epoch']} | Validation RMSE: {checkpoint['min_loss']:.4f}")

    # Perform inference on the test set
    criterion = get_criterion(device)
    losses = initialize_loss_trackers(mode)
    model, gt, pred = evaluate_model(model, device, test_loader, criterion, losses, mode)

    # Log test results to TensorBoard
    tb_writer = SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    log_test_results(tb_writer, losses, gt, pred)
    print("[INFO] Test inference available on TensorBoard.")


if __name__ == "__main__": 
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts, device)
    test_model(args, opts, device)

