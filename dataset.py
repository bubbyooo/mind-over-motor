# PyTorch Dataset wrapper
# Keeps PyTorch interface layer separate from EEG data

import torch
from torch.utils.data import Dataset, DataLoader

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
    

def subject_split(dataset, train_ids, test_ids):
    train = [x for x in dataset if x['subject'] in set(train_ids)]
    test = [x for x in dataset if x['subject'] in set(test_ids)]
    return train, test