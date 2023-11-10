import torch
from torch.utils.data import Dataset, DataLoader
class TextEmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # should be a 2D numpy array
        self.y = y  # should be a 1D numpy array or list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)