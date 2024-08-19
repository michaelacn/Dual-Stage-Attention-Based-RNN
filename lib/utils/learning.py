import numpy as np
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def mape(y_true, y_pred): 
    """
    MAPE score
    """
    epsilon = 1e-10  # To prevent division by zero
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def save_checkpoint(chk_path, epoch, lr, optimizer, scheduler, model, min_loss):
    """
    Save the current training state, including model, optimizer, scheduler, and metrics, to a checkpoint file.
    """
    model.eval()

    print("[INFO]: Saving model checkpoint")
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_loss': min_loss,
        'scheduler': scheduler.state_dict()
    }, chk_path)


def get_criterion(device):
    """
    Create and return the MSE loss criterion moved to the specified device.
    """ 
    criterion = torch.nn.MSELoss() 
    criterion.to(device)
    return criterion


def lr_lambda(iteration):
    """
    Return a learning rate decay factor that reduces by 10% every 10,000 iterations.
    """
    return 0.9 ** (iteration // 10000)


def get_optimizer(args, model): 
    """
    Initialize and return an Adam optimizer with a learning rate scheduler.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)    
    return optimizer, scheduler


def train_epoch(model, device, train_loader, criterion, optimizer, scheduler, losses, epoch):
    """
    Execute one training epoch: forward pass, loss computation, backward pass, and parameter update.
    """
    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y_known, target = data
        x, y_known, target = x.to(device), y_known.to(device) # Move to device
        target.to(device)[:, None] # Future value
        preds = model(x, y_known)  # Forward pass
        loss = criterion(target, preds)  # Compute loss
        losses["train_MSE_loss"].update(loss.item(), preds.shape[0])  # Update loss tracker
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters
        scheduler.step()  # Update learning rate with scheduler (if applicable)


def validate_epoch(model, device, val_loader, criterion, losses, epoch, mode="val"):
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

