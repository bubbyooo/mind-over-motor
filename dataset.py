# PyTorch Dataset wrapper
# Keeps PyTorch interface layer separate from EEG data

import torch
from torch.utils.data import Dataset, DataLoader
import random

class EEGDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]['x']
        y = self.data[idx]['y']
        if self.transform:
            x = self.transform(x)
        return x, y
    
# Test/train split by subject
def subject_split(dataset, train_ids, test_ids):
    # deals with train_ids/test_ids here instead of in train.py
    subject_ids = list(range(49))
    random.seed(42)
    random.shuffle(subject_ids)
    train_ids = subject_ids[:40]
    test_ids = subject_ids[40:]

    train = [x for x in dataset if x['subject'] in set(train_ids)]
    test = [x for x in dataset if x['subject'] in set(test_ids)]
    return train, test

def random_split(dataset, frac = .8, seed = 42):
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(len(dataset) * frac)
    train_idx = set(indices[:split])
    test_idx = set(indices[split:])
    train = [dataset[i] for i in train_idx]
    test = [dataset[i] for i in test_idx]
    return train, test