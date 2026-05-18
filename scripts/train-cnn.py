import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import model_1d_cnn as cnn
from epocher import Data_Epoch
from dataset import random_split, subject_split
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from evaluation import plot_confusion_matrix_cnn, plot_loss, accuracy

# --- Hyperparameters ---
DATA_DIR     = "data"
LR           = 0.00005  # learning rate
EPOCHS       = 100
BATCH_SIZE   = 68
PATIENCE     = 10       # number of epochs without improvement before early stopping
TRAIN        = True    # set to False to skip training and load saved model


def data_prep(dataset):
    """
    Prepares train/test tensors from the dataset.
    Adds a combin
    Splits data, clamps outliers, and normalizes using training set statistics.
    """
    train, test = random_split(dataset)

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
    print("XSHAPE: ", X.shape, " COMBO SHAPE: ", c3_c4.shape)
    X = torch.cat((X, c3_c4), dim = 1)
    return X

def main():
    model = cnn.ConvNet()
    epoch = Data_Epoch()

    # Build dataset using seconds 2 to 6 from each trial
    dataset = epoch.build_dataset(DATA_DIR, end = 4)

    # Prepare train/test tensors
    X_train, X_test, y_train, y_test = data_prep(dataset)

    # Wrap training data in a DataLoader for batching
    train_loader  = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    
    # Adam optimizer with learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Reduce LR when validation loss plateaus (suggested by claude)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)
    loss_fn = nn.CrossEntropyLoss()
    # train the model
    total_loss = 0
    if TRAIN:
        train_losses = []
        val_losses = []
        
        # Early stopping trackers (from claude)
        best_val = float('inf')
        strikes = 0

        for epoch in range(EPOCHS): 
            train_batch_losses = []

            for X_batch, y_batch in train_loader:
                model.train() # enable dropout during training (per claude)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss   = loss_fn(y_pred, y_batch)
                train_batch_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            
            # Average loss across all batches in this epoch
            epoch_train_loss = sum(train_batch_losses)/len(train_batch_losses)
            train_losses.append(epoch_train_loss)

            # Evaluate on test set
            model.eval() # disable dropout for evaluation (per claude)
            with torch.no_grad():
                val_y_pred = model(X_test)
                val_loss = loss_fn(val_y_pred, y_test)
            val_losses.append(val_loss.item())
            
            print("epoch loss: ", loss)
            print("epoch val loss: ", val_loss, "\n")
            print('Strikes: ', strikes)
            # total_loss += loss.item() #claude

            # Save best model and apply early stopping (from claude)
            if val_loss < best_val:
                best_val = val_loss
                strikes = 0
                torch.save(model.state_dict(), "best_model.pth")
            else:
                strikes += 1
                if strikes >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break

            scheduler.step(val_loss)
        
        plot_loss(train_losses, val_losses)
      #  plot_confusion_matrix_cnn(model, X_test, y_test)

        torch.save(model.state_dict(), "eeg_model_rec10.pth")
        print("Model saved!")

    else:
        # Load a previously saved model for evaluation
        model = cnn.ConvNet()
        model.load_state_dict(torch.load("eeg_model_rec10.pth"))
        model.eval()
        
    plot_confusion_matrix_cnn(model, X_test, y_test)
    print("train accuracy: ", accuracy(model, X_train, y_train))
    print("test accuracy: ", accuracy(model, X_test, y_test))

if __name__ == "__main__":
    main()

