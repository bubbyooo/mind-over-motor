from data_epoching_pipeline import Data_Epoch
from features import build_feature_set
import random

# basis of logistic regression model

# train-test split: 

epoch = Data_Epoch()
dataset = epoch.build_dataset("edffile")
print("length of dataset: ", len(dataset))
ids = list(range(0,49))
random.seed(42)
random.shuffle(ids)
train_ids = ids[:40]
test_ids = ids[41:]
train = [x for x in dataset if x['subject'] in train_ids] # from chatgpt
test = [x for x in dataset if x['subject'] in test_ids] #from chatgpt
X_train, y_train = build_feature_set(train)
X_test, y_test = build_feature_set(test)


import torch

# copied from lecture mar2
def binary_cross_entropy(q, y):
    return -(y * torch.log(q) + (1-y)*torch.log(1-q)).mean()


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

# from lecture mar2
class BinaryLogReg:
    def __init__(self, n_features):
        self.w = torch.zeros(n_features, 1)

    def forward(self, X):
        return sigmoid(X @ self.w)
    
# from lecture mar2
class GradientDescentOptimizer:
    def __init__(self, model, lr=.1):
        self.model = model
        self.lr = lr

    def grad_func(self, X, y):
        q = self.model.forward(X)
        return 1 / X.shape[0] * ((q - y).T @ X).T
    
    def step(self, X, y):
        grad  = self.grad_func(X, y)
        with torch.no_grad():
            self.model.w -= self.lr * grad


def accuracy(model, X, y):
    with torch.no_grad():
        preds = (model.forward(X) >= 0.5).float()
        return (preds == y).float().mean().item()

# Convert to float tensors and make sure y is (N, 1)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

n_features = X_train.shape[1]

model = BinaryLogReg(n_features=n_features)
opt   = GradientDescentOptimizer(model, lr=.1)

losses = []
for ep in range(100):
    q    = model.forward(X_train)
    loss = binary_cross_entropy(q, y_train)
    losses.append(loss.item())
    opt.step(X_train, y_train)
 
    # Claude for epoch reporting
    if (ep + 1) % 10 == 0:
        train_acc = accuracy(model, X_train, y_train)
        print(f"Epoch {ep+1:3d} | loss: {loss.item():.4f} | train acc: {train_acc:.3f}")

test_acc = accuracy(model, X_test, y_test)
print(f"\nTest accuracy: {test_acc:.3f}")