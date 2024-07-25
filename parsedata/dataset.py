import sys
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

class MLDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)
        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.labels = [torch.tensor(example.labels).float() for example in features]


    def __len__(self):
        return self.nums


    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index], 'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index], 'labels': self.labels[index]}
        return data