import numpy as np
import torch
import random
import os


def set_random_seed(seed):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_loss_trackers():
    """
    Initialize loss trackers for training, validation, and test.
    """
    return {
        "train_MSE_loss": AverageMeter(),
        "val_MSE_loss": AverageMeter(),
        "val_MAE_loss": AverageMeter(),
        "val_RMSE_loss": AverageMeter(),
        "val_MAPE_loss": AverageMeter(),
        "test_MSE_loss": AverageMeter(),
        "test_MAE_loss": AverageMeter(),
        "test_RMSE_loss": AverageMeter(),
        "test_MAPE_loss": AverageMeter(),
    }


def mape(y_true, y_pred): 
    """
    MAPE score.
    """
    epsilon = 1e-10  # To prevent division by zero
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100



def lr_lambda(iteration):
    """
    Return a learning rate decay factor that reduces by 10% every 10_000 iterations.
    """
    return 0.9 ** (iteration // 10000)


def get_criterion(device):
    """
    Create and return the MSE loss criterion moved to the specified device.
    """ 
    criterion = torch.nn.MSELoss() 
    criterion.to(device)
    return criterion


def get_optimizer(args, model): 
    """
    Initialize and return an Adam optimizer with a learning rate scheduler.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)    
    return optimizer, scheduler


def train_epoch(model, device, train_loader, criterion, optimizer, scheduler, losses):
    """
    Execute one training epoch: forward pass, loss computation, backward pass, and parameter update.
    """
    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y_known, target = data
        x, y_known = x.to(device), y_known.to(device) # Move to device
        target = target.to(device)[:, None] # Future value
        preds = model(x, y_known)  # Forward pass
        loss = criterion(target, preds)  # Compute loss
        losses["train_MSE_loss"].update(loss.item(), preds.shape[0])  # Update loss tracker
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters
        scheduler.step()  # Update learning rate with scheduler (if applicable)


def validate_epoch(model, device, val_loader, criterion, losses, mode="val"):
    """Evaluate the model on the validation set, updating loss metrics and returning predictions."""
    model.eval()
    pred, gt = [], []

    with torch.no_grad(): 
        for x, y_known, target in val_loader:  
            x, y_known, target = x.to(device), y_known.to(device), target.to(device)[:, None] # Move to device

            # Model prediction and loss calculation
            preds = model(x, y_known)
            loss = criterion(target, preds)
            mae_loss = torch.nn.functional.l1_loss(target, preds)
            mape_loss = mape(target, preds)

            # Update
            batch_size = preds.shape[0]
            losses[f"{mode}_MSE_loss"].update(loss.item(), batch_size)
            losses[f"{mode}_MAE_loss"].update(mae_loss.item(), batch_size)
            losses[f"{mode}_RMSE_loss"].update(torch.sqrt(loss).item(), batch_size)
            losses[f"{mode}_MAPE_loss"].update(mape_loss.item(), batch_size)
            
            # Store predictions and targets
            gt.append(target.cpu().numpy())
            pred.append(preds.cpu().numpy())

    gt = np.concatenate(gt)
    pred = np.concatenate(pred)[:, 0]

    return model, gt, pred



def save_checkpoint(chk_path, epoch, lr, optimizer, scheduler, model, min_loss):
    """
    Save the current training state, including model, optimizer, scheduler, and metrics, to a checkpoint file.
    """
    model.eval()
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_loss': min_loss,
        'scheduler': scheduler.state_dict()
    }, chk_path)


def save_checkpoints(checkpoint_dir, model, optimizer, scheduler, epoch, losses, min_loss):
    """
    Save the model checkpoints during training.
    """
    chk_path_latest = os.path.join(checkpoint_dir, 'latest_epoch.bin')
    chk_path_best = os.path.join(checkpoint_dir, 'best_epoch.bin')

    save_checkpoint(chk_path_latest, epoch, optimizer.param_groups[0]['lr'], optimizer, scheduler, model, min_loss)
    print("[INFO LOG]: Saving model checkpoint")
    if losses["val_RMSE_loss"].avg < min_loss:
        save_checkpoint(chk_path_best, epoch, optimizer.param_groups[0]['lr'], optimizer, scheduler, model, losses["val_RMSE_loss"].avg)
        print("[INFO LOG]: Best model checkpoint saved")



def log_metrics(writer, losses, optimizer, gt, pred, epoch):
    """
    Log training and validation metrics to TensorBoard.
    """
    lr = optimizer.param_groups[0]['lr']
    writer.add_scalar("train_MSE_loss", losses["train_MSE_loss"].avg, epoch + 1)
    writer.add_scalar("val_MSE_loss", losses["val_MSE_loss"].avg, epoch + 1)
    writer.add_scalar("lr", lr, epoch + 1)
    for i in range(len(gt)):
        writer.add_scalars("validation_prediction", {"gt": gt[i], "pred": pred[i]}, i)


def log_test_results(writer, losses, gt, pred):
    """
    Log test results to TensorBoard.
    """
    for i in range(len(gt)):
        writer.add_scalars("test_prediction", {"gt": gt[i], "pred": pred[i]}, i)
    writer.add_scalar("test_MSE_loss", losses["test_MSE_loss"].avg, 0)
    writer.add_scalar("test_MAE_loss", losses["test_MAE_loss"].avg, 0)
    writer.add_scalar("test_RMSE_loss", losses["test_RMSE_loss"].avg, 0)
    writer.add_scalar("test_MAPE_loss", losses["test_MAPE_loss"].avg, 0)

