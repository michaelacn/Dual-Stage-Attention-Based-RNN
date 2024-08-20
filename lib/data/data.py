import numpy as np 
import pandas as pd 

import torch
from torch.utils.data import DataLoader, TensorDataset


def get_data_and_preprocess(csv_path, target, timesteps, train_split, val_split, prediction_horizon):
    """
    Preprocess time series data by creating sequences, applying min-max scaling, and splitting into training, validation, and test sets with a specified prediction horizon.
    """
    data = pd.read_csv(csv_path)
    n_exogenous = data.shape[1] - 1

    # Placeholders
    X = np.zeros((len(data), timesteps, n_exogenous))
    y = np.zeros((len(data), timesteps, 1))

    # Fill X and y
    for i, name in enumerate(data.columns[:-1]):
        for j in range(timesteps):
            X[:, j, i] = data[name].shift(timesteps - j - 1).bfill()
            y[:, j, 0] = data[target].shift(timesteps - j - 1).bfill()

    target_shifted = data[target].shift(-prediction_horizon).ffill().values

    # Split the data
    train_len = int(len(data) * train_split)
    val_len = int(len(data) * val_split)
    test_len = len(data) - train_len - val_len

    print("TRAIN LENGTH: ", train_len)
    print("VALIDATION LENGTH: ", val_len)
    print("TEST LENGTH: ", test_len)

    X_train, X_val, X_test = X[:train_len], X[train_len:train_len+val_len], X[train_len+val_len:]
    y_train, y_val, y_test = y[:train_len], y[train_len:train_len+val_len], y[train_len+val_len:]
    target_train, target_val, target_test = target_shifted[:train_len], target_shifted[train_len:train_len+val_len], target_shifted[train_len+val_len:]

    # Min-max scaling
    def scale(data, data_min, data_max):
        return (data - data_min) / (data_max - data_min)

    X_min, X_max = X_train.min(axis=0), X_train.max(axis=0)
    y_min, y_max = y_train.min(axis=0), y_train.max(axis=0)
    target_min, target_max = target_train.min(), target_train.max()

    X_train = scale(X_train, X_min, X_max)
    X_val = scale(X_val, X_min, X_max)
    X_test = scale(X_test, X_min, X_max)
    y_train = scale(y_train, y_min, y_max)
    y_val = scale(y_val, y_min, y_max)
    y_test = scale(y_test, y_min, y_max)
    target_train = scale(target_train, target_min, target_max)
    target_val = scale(target_val, target_min, target_max)
    target_test = scale(target_test, target_min, target_max)

    # Convert to tensors
    def to_tensor(*arrays):
        return [torch.Tensor(arr) for arr in arrays]

    return to_tensor(X_train, X_val, X_test, y_train, y_val, y_test, target_train, target_val, target_test)


def get_dataloader(args):
    """
    Create and return DataLoader objects for training, validation, and testing.
    """
    data = get_data_and_preprocess(args.csv_file, args.target, args.timesteps, args.train_split, args.val_split, args.prediction_horizon)
    X_train_t, X_val_t, X_test_t, y_his_train_t, y_his_val_t, y_his_test_t, target_train_t, target_val_t, target_test_t = data

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_his_train_t, target_train_t), 
        shuffle=True, 
        batch_size=args.batch_size
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t, y_his_val_t, target_val_t), 
        shuffle=False, 
        batch_size=args.batch_size
    )

    test_loader = DataLoader(
        TensorDataset(X_test_t, y_his_test_t, target_test_t), 
        shuffle=False, 
        batch_size=args.batch_size
    )

    return train_loader, val_loader, test_loader



