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
    """
    1D CNN for 3-class EEG motor imagery classification (left / right / rest).

    Input:  (batch, 4, time) — 4 EEG feature channels
    Output: (batch, 3)       — raw class logits
    """

    def __init__(self):
        super().__init__()

        self.pipeline = torch.nn.Sequential(
            # Block 1: 4 → 32 filters
            nn.Conv1d(4, 32, kernel_size = 13, padding = 12),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(4),
            ReLU(),
            nn.Dropout(0.2),

            # Block 2: 32 → 16 filters
            nn.Conv1d(32, 16, kernel_size = 11, padding = 6),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(4),
            ReLU(),
            nn.Dropout(0.2),

            # Block 3: 16 → 32 filters
            nn.Conv1d(16, 32, kernel_size = 9, padding = 4),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            ReLU(),
            nn.Flatten(),
            nn.LazyLinear(3) # Infers input size automatically; outputs 3 logits

        )

    def forward(self, x):
        return self.pipeline(x)

