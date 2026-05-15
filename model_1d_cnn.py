import torch.nn as nn
import torch
from torch.nn import ReLU

# plans to improve
# 1. work in batching to reduce overfitting
# 2. add regularizer to reduce overfitting
# 3. consider/research dropout and weight decay
# 4. consider return to subject switch
# 5. comment code etc

#modified from lecture 13 notes
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pipeline = torch.nn.Sequential(
            nn.Conv1d(3, 32, kernel_size = 25, padding = 12),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(4),
            ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(32, 16, kernel_size = 13, padding = 6),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(4),
            ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(16, 32, kernel_size = 9, padding = 4),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            ReLU(),
            nn.Flatten(),
            nn.LazyLinear(2)
        )

    def forward(self, x):
        return self.pipeline(x)

