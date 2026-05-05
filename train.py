# Log-reg model training
# Run from project root with `python train.py`

import random
import torch

from epocher import Data_Epoch
from dataset import subject_split
from features import build_feature_matrix
from logistic_regression_model import BinaryLogReg, GradientDescentOptimizer, binary_cross_entropy, accuracy

import numpy as np


DATA_DIR     = "data/edffile"
SEED         = 42
N_SUBJECTS   = 49
N_TRAIN      = 40
LR           = .1
EPOCHS       = 1000
LOG_EVERY    = 10


def main():
    # Build dataset
    print("Loading and epoching EDF files...")
    epocher = Data_Epoch()
    dataset = epocher.build_dataset(DATA_DIR)
    print(f"Total trials: {len(dataset)}")

    # Test/train split by subject
    subject_ids = list(range(N_SUBJECTS))
    random.seed(SEED)
    random.shuffle(subject_ids)

    train_ids = subject_ids[:N_TRAIN]
    test_ids = subject_ids[N_TRAIN:]

    train_data, test_data = subject_split(dataset, train_ids, test_ids)
    print(f"Train subjects: {train_ids}, Test subjects: {test_ids}")

    # Feature extraction
    print("\nExtracting features...")
    X_train, y_train = build_feature_matrix(train_data)
    X_test, y_test = build_feature_matrix(test_data)

    # Cast to float; reshape y to (N, 1) for loss
    X_train = X_train.float()
    y_train = y_train.float().unsqueeze(1)
    X_test = X_test.float()
    y_test = y_test.float().unsqueeze(1)

    print(f"Feature matrix shape: {X_train.shape}, Labels shape: {y_train.shape}")

    # Model and optimizer
    n_features = X_train.shape[1]
    model = BinaryLogReg(n_features=n_features)
    opt   = GradientDescentOptimizer(model, lr=LR)
 
    # DEBUG
    print(f"debug: {np.linalg.norm(model.w)}")

    # Training loop
    print(f"\nTraining for {EPOCHS} epochs (lr={LR})...")
    for ep in range(EPOCHS):
        q    = model.forward(X_train)
        loss = binary_cross_entropy(q, y_train)
        opt.step(X_train, y_train)
 
        if (ep + 1) % LOG_EVERY == 0:
            train_acc = accuracy(model, X_train, y_train)
            print(
                f"  Epoch {ep+1:3d} | "
                f"loss: {loss.item():.4f} | "
                f"train acc: {train_acc:.3f}"
            )
 
    # DEBUG
    print(f"debug: {np.linalg.norm(model.w)}")

    # Evaluation
    test_acc = accuracy(model, X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.3f}")
 
 
if __name__ == "__main__":
    main()