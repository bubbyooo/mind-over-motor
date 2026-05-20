import torch.nn as nn
import torch
from torch.nn import ReLU

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

        # Inception block: captures features at 3 temporal scales simultaneously
        self.incept1 = InceptionBlock(in_channels=5, out_channels=8) # out 24 channels

        # Post-inception normalization, downsampling, and regularization
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(24), # normalize across 24 inception output channels
            nn.MaxPool1d(4),# downsample by 4x
            ReLU(),
            nn.Dropout(0.2), # randomly zero 20% of units to reduce overfitting
        )

        # Main conv pipeline: further feature extraction and classification
        self.pipeline = torch.nn.Sequential(
            # Conv block: 24 → 16 filters, large kernel to capture long-range patterns
            nn.Conv1d(24, 16, kernel_size = 13, padding = 12),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(4),    # downsample by 4x
            ReLU(),
            nn.Dropout(0.2),

            # Conv block: 16 → 32 filters, smaller kernel for local features
            nn.Conv1d(16, 32, kernel_size = 9, padding = 4),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),    # downsample by 2x
            ReLU(),
            nn.Flatten(),
            nn.LazyLinear(3) # Infers input size automatically; outputs 3 logits

        )

    def forward(self, x):
        x = self.incept1(x)     # multi-scale feature extraction
        x = self.block1(x)      # normalize, pool, regularize
        return self.pipeline(x)
    

# inception block was inspired by Kaviri and Vinjamuri (referenced in report)
# code for the Inception class heavily from Claude
class InceptionBlock(nn.Module):
    """
    Parallel convolutions at 3 kernel sizes to capture features at different timescales.
    Outputs are concatenated along the channel dimension.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Three branches, each with a different kernel size
        self.branch_narrow = nn.Conv1d(in_channels, out_channels, kernel_size = 3, padding = 1) # fine-grained
        self.branch_mid = nn.Conv1d(in_channels, out_channels, kernel_size = 7, padding = 3)    # mid scale
        self.branch_wide = nn.Conv1d(in_channels, out_channels, kernel_size = 13, padding = 6)  # course

    def forward(self, x):
        n = self.branch_narrow(x)
        m = self.branch_mid(x)
        w = self.branch_wide(x)
        return torch.cat([n, m, w], dim = 1) # concatenate along channel dim → 3 * out_channels
    