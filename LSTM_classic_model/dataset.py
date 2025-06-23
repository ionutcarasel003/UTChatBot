import torch
from torch.utils.data import Dataset
from utils import encode
from preprocessing import tokenize

class QADataset(Dataset):
    def __init__(self, pairs, word2idx):
        self.data = [(encode(tokenize(q), word2idx), encode(tokenize(a), word2idx)) for q, a in pairs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, a = self.data[idx]
        return torch.tensor(q), torch.tensor(a)
