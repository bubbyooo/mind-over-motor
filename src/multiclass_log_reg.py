# Multiclass logistic regression (softmax classifier)

import torch


#---Loss and activation functions---

# from lecture mar2
def softmax_rows(S):
    S = S - torch.max(S, dim=1, keepdim=True).values  # for numerical stability
    exp_S = torch.exp(S)
    return exp_S / torch.sum(exp_S, dim=1, keepdim=True)

# from lecture mar2
# used Claude to select true class log-probs
def cross_entropy(Q, Y, weights=None):
    # Q : (n, K) predicted probabilities from softmax
    # Y : (n,)  integer class labels in {0, ..., K-1}
    log_Q = torch.log(Q + 1e-8)  # add small value for numerical stability
    log_Q = log_Q[range(len(Y)), Y]  # select log-prob of true class for each sample
    if weights is not None:
        sample_weights = weights[Y]
        return -(log_Q * sample_weights).sum() / sample_weights.sum()
    return -torch.mean(log_Q)


#---Model---

# from lecture mar2
class MulticlassLogReg:

    def __init__(self, d_features, k_classes):
        # Matrix W
        self.W = torch.randn(d_features, k_classes) * .01

    def __getstate__(self):
        return {'W': self.W}

    def forward(self, X):
        return softmax_rows(X @ self.W)


#---Optimizer---

# from lecture mar2
class GradientDescentOptimizer:
    def __init__(self, model, lr=0.1, class_weights=None):
        self.model = model
        self.lr = lr
        self.class_weights = class_weights  # used Claude to implement balancing by class

    def step(self, X, y):
        self.model.W -= self.lr * self.grad_func(X, y)

    def grad_func(self, X, y):
        q = self.model.forward(X)
        y_one_hot = torch.zeros_like(q)
        y_one_hot[range(len(y)), y] = 1.0
        residual = q - y_one_hot

        if self.class_weights is not None:
            sample_weights = self.class_weights[y]
            sample_weights = sample_weights / sample_weights.sum()
            residual = residual * sample_weights.unsqueeze(1)
            return X.T @ residual
        
        return X.T @ residual / X.shape[0]


#---Evaluation---

def accuracy(model, X, y):
    with torch.no_grad():
        probs = model.forward(X)
        preds = probs.argmax(dim=1)
        return (preds == y).float().mean().item()