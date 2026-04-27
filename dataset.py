
# maybe getting ahead -> this will be useful for neural network
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, data, path, transform=None):
        self.data = data
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        return self.data[idx]['x'], self.data[idx]['y']