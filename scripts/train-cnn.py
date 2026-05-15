import model_1d_cnn as cnn
from epocher import Data_Epoch
from dataset import random_split, subject_split
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from evaluation import plot_confusion_matrix_cnn, plot_loss, accuracy

DATA_DIR     = "data"
LR           = 0.00005
EPOCHS       = 100
BATCH_SIZE   = 32
PATIENCE     = 10

def data_prep(dataset):
    train, test = random_split(dataset)

    X_train = torch.stack([item["x"] for item in train]).float()
    y_train = torch.tensor([item["y"] for item in train], dtype=torch.long)
    X_test  = torch.stack([item["x"] for item in test]).float()
    y_test  = torch.tensor([item["y"] for item in test], dtype=torch.long)

    #getting rid of outliers as per claude
    X_train = X_train.clamp(-5, 5)
    X_test  = X_test.clamp(-5, 5)

    # Re-normalize after clipping as per claude
    mean = X_train.mean(dim=(0, 2), keepdim=True)
    std  = X_train.std(dim=(0, 2), keepdim=True)

    X_train = (X_train - mean) / (std + 1e-8)
    X_test  = (X_test  - mean) / (std + 1e-8)

    return X_train, X_test, y_train, y_test

def main():
    model = cnn.ConvNet()
    epoch = Data_Epoch()
    dataset = epoch.build_dataset(DATA_DIR, end = 6)

    # create train and test datasets
    X_train, X_test, y_train, y_test = data_prep(dataset)

    train_loader  = DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=68, shuffle=True)
    #create optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    #scheduler suggested by claude
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)
    loss_fn = nn.CrossEntropyLoss()
    # train the model
    TRAIN = True
    total_loss = 0
    if TRAIN:
        train_losses = []
        val_losses = []
        #following 3 lines from claude on patience metric
        best_val = float('inf')
        strikes = 0

        for epoch in range(EPOCHS): # note performs better with around 5
            train_batch_losses = []

            for X_batch, y_batch in train_loader:
                model.train() # per claude makes sure training mode is on (ie with dropout)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss   = loss_fn(y_pred, y_batch)
                train_batch_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            
            epoch_train_loss = sum(train_batch_losses)/len(train_batch_losses)
            train_losses.append(epoch_train_loss)

            model.eval() #per claude removes dropout
            with torch.no_grad():
                val_y_pred = model(X_test)
                val_loss = loss_fn(val_y_pred, y_test)
            val_losses.append(val_loss.item())
            
            print("epoch loss: ", loss)
            print("epoch val loss: ", val_loss, "\n")
            print('Strikes: ', strikes)
            # total_loss += loss.item() #claude

            # if__else from claude, trains until 10 subsequent worse losses in a row
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
        model = cnn.ConvNet()
        model.load_state_dict(torch.load("eeg_model_rec10.pth"))
        model.eval()
        
    plot_confusion_matrix_cnn(model, X_test, y_test)
    print("train accuracy: ", accuracy(model, X_train, y_train))
    print("test accuracy: ", accuracy(model, X_test, y_test))

if __name__ == "__main__":
    main()

