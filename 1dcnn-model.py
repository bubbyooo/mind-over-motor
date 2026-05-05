import torch.nn as nn
import torch
from torch.nn import Conv1d, MaxPool1d, Parameter
from torch.nn.functional import relu
from torch.nn import ReLU
from epocher import Data_Epoch
from features import build_feature_matrix
import random
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import random_split

#modified from lecture 13 notes
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pipeline = torch.nn.Sequential(
            nn.Conv1d(3, 64, 12),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            ReLU(),
            nn.Conv1d(64, 32, 3),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            ReLU(),
            nn.Flatten(),
            nn.LazyLinear(2)
        )

    def forward(self, x):
        return self.pipeline(x)
    
# grabbed from logreg class
def accuracy(model, X, y):
    with torch.no_grad():
        preds = model.forward(X).argmax(dim=1)
        return (preds == y).float().mean().item()

model = ConvNet()
epoch = Data_Epoch()
dataset = epoch.build_dataset("edffile")

#use random_split from dataset.py to get a random split (not by subject)
train, test = random_split(dataset)

# what follows is a train test split by subject, which we will temporarily avoid #

# ids = list(range(0,49))
# random.seed(42)
# random.shuffle(ids)
#train_ids = ids[:40]
#test_ids = ids[40:]
#train = [x for x in dataset if x['subject'] in train_ids] # from chatgpt
#test = [x for x in dataset if x['subject'] in test_ids] #from chatgpt

# following four lines from claude
X_train = torch.stack([item["x"] for item in train])
y_train = torch.tensor([item["y"] for item in train], dtype=torch.long)
X_test  = torch.stack([item["x"] for item in test])
y_test  = torch.tensor([item["y"] for item in test], dtype=torch.long)

# Convert to float tensors and make sure y is (N, 1)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.long)

#sanity check as per claude:
print("X_train stats:")
print("  mean:", X_train.mean().item())
print("  std: ", X_train.std().item())
print("  min: ", X_train.min().item())
print("  max: ", X_train.max().item())

#normalizing as per claude
mean = X_train.mean(dim=(0, 2), keepdim=True)
std  = X_train.std(dim=(0, 2), keepdim=True)

X_train = (X_train - mean) / (std + 1e-8)
X_test  = (X_test  - mean) / (std + 1e-8)

#getting rid of outliers as per claude
X_train = X_train.clamp(-10, 10)
X_test  = X_test.clamp(-10, 10)

# Re-normalize after clipping as per claude
mean2 = X_train.mean(dim=(0, 2), keepdim=True)
std2  = X_train.std(dim=(0, 2), keepdim=True)

X_train = (X_train - mean2) / (std2 + 1e-8)
X_test  = (X_test  - mean2) / (std2 + 1e-8)

#updated sanity check post normalizing and outlier removing
print("X_train stats:")
print("  mean:", X_train.mean().item())
print("  std: ", X_train.std().item())
print("  min: ", X_train.min().item())
print("  max: ", X_train.max().item())


loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
TRAIN = True
total_loss = 0
if TRAIN:
    train_losses = []
    val_losses = []
    for epoch in range(10):
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss   = loss_fn(y_pred, y_train)
            train_losses.append(loss.item())
            val_y_pred = model(X_test)
            val_loss = loss_fn(val_y_pred, y_test)
            val_losses.append(val_loss.item())
            loss.backward()
            optimizer.step()
            print("loss: ", loss)
            print("val loss: ", val_loss)
            total_loss += loss.item() #claude
  #  if total_loss < best_loss:
     #   best_loss = total_loss
     #   torch.save(model.state_dict(), "eeg_model_best.pth")

    plt.plot(train_losses, label = "Loss")
    plt.plot(val_losses, label="Test Loss")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "eeg_model_rec5.pth")
    print("Model saved!")

    # Save whenever we hit a new best
    #if avg_loss < best_loss:
    #    best_loss = avg_loss
    #    torch.save(model.state_dict(), "eeg_model_best.pth")

else:
    model = ConvNet()
    model.load_state_dict(torch.load("eeg_model_recent2.pth"))
    model.eval()



#second sanity check as per claude
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    print("Predicted class distribution:", preds.bincount())
    print("Actual class distribution:   ", y_test.bincount())

print("train accuracy: ", accuracy(model, X_train, y_train))
print("test accuracy: ", accuracy(model, X_test, y_test))

