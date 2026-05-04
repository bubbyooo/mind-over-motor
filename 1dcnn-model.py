import torch.nn as nn
import torch
from torch.nn import Conv1d, MaxPool1d, Parameter
from torch.nn.functional import relu
from torch.nn import ReLU
from data_epoching_pipeline import Data_Epoch
from features import build_feature_set
import random
import torch.optim as optim

#modified from lecture 13 notes
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pipeline = torch.nn.Sequential(
            nn.Conv1d(3, 100, 12),
            nn.MaxPool1d(2),
            ReLU(),
            nn.Conv1d(100, 50, 3),
            nn.MaxPool1d(2),
            ReLU(),
            nn.Conv1d(50, 50, 3),
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
print("length of dataset: ", len(dataset))
ids = list(range(0,49))
random.seed(42)
random.shuffle(ids)
train_ids = ids[:40]
test_ids = ids[40:]
train = [x for x in dataset if x['subject'] in train_ids] # from chatgpt
test = [x for x in dataset if x['subject'] in test_ids] #from chatgpt

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

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss   = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        print("loss: ", loss)
print(accuracy(model, X_test, y_test))


