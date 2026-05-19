import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from dataset import random_split

def data_prep(dataset, fraction = 0.8):
    """
    Prepares train/test tensors from the dataset.
    Adds a combin
    Splits data, clamps outliers, and normalizes using training set statistics.
    """
    train, test = random_split(dataset, frac = fraction)

    # Stack list of samples into tensors
    X_train = torch.stack([item["x"] for item in train]).float()
    y_train = torch.tensor([item["y"] for item in train], dtype=torch.long)
    X_test  = torch.stack([item["x"] for item in test]).float()
    y_test  = torch.tensor([item["y"] for item in test], dtype=torch.long)

    X_train = add_polynomial_feat(X_train)
    X_test = add_polynomial_feat(X_test)

    # Clamp outliers to [-5, 5] range (as per claude)
    X_train = X_train.clamp(-5, 5)
    X_test  = X_test.clamp(-5, 5)

    # Re-normalize after clipping using training set mean/std (as per claude)
    mean = X_train.mean(dim=(0, 2), keepdim=True)
    std  = X_train.std(dim=(0, 2), keepdim=True)

    X_train = (X_train - mean) / (std + 1e-8)
    X_test  = (X_test  - mean) / (std + 1e-8)

    return X_train, X_test, y_train, y_test

def add_polynomial_feat(X):
    # adds a feature channel multiplying channel c3 by channel c4
    c3_c4 = X[:, 1, :] * X[:, 2, :]
    c3_c4 = torch.unsqueeze(c3_c4, dim = 1)
    X = torch.cat((X, c3_c4), dim = 1)
    return X