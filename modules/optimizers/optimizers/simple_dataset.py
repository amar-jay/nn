import torch
from torch.utils.data import Dataset

# Data preparation
class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = -3 * self.x + 1 + 0.1 * torch.randn(self.x.size())
        self.len = self.x.shape

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len[0]
