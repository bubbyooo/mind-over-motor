# Binary logistic regression model

import torch

from epocher import Data_Epoch


#---Loss and activation functions---

# copied from lecture mar2
def binary_cross_entropy(q, y):
    # Claude to prevent log(0) numerical issues
    eps = 1e-8
    q = torch.clamp(q, eps, 1 - eps)
    return -(y * torch.log(q) + (1-y)*torch.log(1-q)).mean()


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


#---Model---

# from lecture mar2
class BinaryLogReg:
    def __init__(self, n_features):
        self.w = torch.randn(n_features, 1) * 0.01  # small random init
        self.b = torch.zeros(1)  # bias term

    def forward(self, X):
        return sigmoid(X @ self.w + self.b)

#---Optimizer---

# from lecture mar2
class GradientDescentOptimizer:
    def __init__(self, model, lr=.1):
        self.model = model
        self.lr = lr

    def _grad(self, X, y):
        q = self.model.forward(X)
        residuals = q - y
        grad_w = (1 / X.shape[0]) * X.T @ residuals
        grad_b = residuals.mean()
        return grad_w, grad_b
    
    def step(self, X, y):
        grad_w, grad_b = self._grad(X, y)
        with torch.no_grad():
            self.model.w -= self.lr * grad_w
            self.model.b -= self.lr * grad_b


#---Evaluation---

def accuracy(model, X, y):
    with torch.no_grad():
        preds = (model.forward(X) >= 0.5).float()
        return (preds == y).float().mean().item()
