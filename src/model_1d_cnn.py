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

    Input:  (batch, 4, time) — 4 EEG feature channels (cz, c3, c4, and a derived channel: c3*c4)
    Output: (batch, 3)       — raw class logits
    """

    # claude helped extensively with the inception blocks and implementation
    def __init__(self):
        super().__init__()

        self.incept1 = InceptionBlock(in_channels=4, out_channels=8) # out 24 channels

        self.block1 = nn.Sequential(
            nn.BatchNorm1d(24),
            nn.MaxPool1d(4),
            ReLU(),
            nn.Dropout(0.2),
        )

        self.resid1 = ResidualBlock(24)

        self.pipeline = torch.nn.Sequential(
            # Block 1: 4 → 32 filters
            # changed to 24 and 16 instead of 4 and 32
            nn.Conv1d(24, 16, kernel_size = 13, padding = 12),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(4),
            ReLU(),
            nn.Dropout(0.2),

             #temporarily commented out bc of new block
            # Block 2: 32 → 16 filters
           # nn.Conv1d(32, 16, kernel_size = 11, padding = 6),
           # nn.BatchNorm1d(16),
           # nn.MaxPool1d(4),
           # ReLU(),
           # nn.Dropout(0.2),

            # Block 3: 16 → 32 filters
            nn.Conv1d(16, 32, kernel_size = 9, padding = 4),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            ReLU(),
            nn.Flatten(),
            nn.LazyLinear(3) # Infers input size automatically; outputs 3 logits

        )

    def forward(self, x):
        x = self.incept1(x) 
        x = self.block1(x) 
        x = self.resid1(x)
        return self.pipeline(x)
    

# inception block was inspired by Kaviri and Vinjamuri (referenced in report)
# code for the Inception class heavily from Claude
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Three branches, each with a different kernel size
        self.branch_narrow = nn.Conv1d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.branch_mid = nn.Conv1d(in_channels, out_channels, kernel_size = 7, padding = 3)
        self.branch_wide = nn.Conv1d(in_channels, out_channels, kernel_size = 13, padding = 6)

    def forward(self, x):
        n = self.branch_narrow(x)
        m = self.branch_mid(x)
        w = self.branch_wide(x)
        return torch.cat([n, m, w], dim = 1) # stack along channel dimension

# also inspired by Kaviri and Vinjamuri
# code taken from Claude
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=9, padding=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)
    