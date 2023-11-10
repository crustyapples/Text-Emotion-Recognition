import torch
from torch.utils.data import Dataset, DataLoader
class TextEmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # should be a 2D numpy array
        self.y = y  # should be a 1D numpy array or list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # If your X is already a numpy array, you don't need from_numpy for X
        # For y, since it is a single value, you can use torch.tensor directly
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)



# Example usage:
# dataset = TextEmotionDataset(X_train_pad, y_train)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
