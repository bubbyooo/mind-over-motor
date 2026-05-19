# PyTorch Dataset wrapper
# Keeps PyTorch interface layer separate from EEG data

import torch
from torch.utils.data import Dataset
import random


N_SUBJECTS      = 50        #Total number of subjects in the dataset.
N_TRAIN         = 40        # Number of subjects allocated to the training partition in subject_split().

class EEGDataset(Dataset):
    """
    PyTorch Dataset wrapper for EEG data.

    Args:
        data (list[dict]): Samples with keys 'x' (EEG signal), 'y' (label),
            and 'subject' (subject ID).
        transform (callable, optional): Transform applied to 'x' on retrieval.
    """
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform # Optional transform applied to each sample

    def __len__(self):
        """Return total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.

        Args:
            idx (int): Sample index.

        Returns:
            tuple: (x, y) where x is the (optionally transformed) EEG signal
                   and y is the corresponding label.
        """
        x = self.data[idx]['x']  # EEG signal for this sample
        y = self.data[idx]['y']  # Ground-truth label
         # Apply optional transform (e.g. normalisation, data augmentation)
        if self.transform:
            x = self.transform(x)
        return x, y

    
# Test/train split by subject
def subject_split(dataset):
    """
    Split by subject ID so no subject appears in both partitions.  
    Prevents data leakage and gives a more realistic cross-subject evaluation
    than a random sample split. 

    Returns:
        tuple: (train, test) sample lists.
    """
    subject_ids = list(range(N_SUBJECTS - 1))
    random.seed(42)
    random.shuffle(subject_ids)

    train_ids = subject_ids[:N_TRAIN]
    test_ids = subject_ids[N_TRAIN:]

    train = [x for x in dataset if x['subject'] in set(train_ids)]
    test = [x for x in dataset if x['subject'] in set(test_ids)]
    return train, test

def random_split(dataset, frac=0.8, seed=42):
    """
    Randomly split at the sample level (same subject may appear in both sets).

    Args:
        dataset: Any indexable sequence.
        frac (float): Fraction for training (default 0.8).
        seed (int):   Random seed (default 42).

    Returns:
        tuple: (train, test) sample lists.
    """
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    n_train = int(frac * len(dataset))
    train = [dataset[i] for i in indices[:n_train]]
    test = [dataset[i] for i in indices[n_train:]]
    return train, test