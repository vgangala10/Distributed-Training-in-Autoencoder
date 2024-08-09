import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
import json
from torchvision.datasets import CIFAR10
from torchvision import transforms

class LoadCifar(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(data)
    def __getitem__(self, index):
        image = self.data[index][0]
        return image
    def __len__(self):
        return self.length
    
class AEdata(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        # Download the train and validation datasets
        self.train_set = CIFAR10('./', download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        ]), train=True)

        self.train_set = LoadCifar(self.train_set)
        
        
        self.val_set = CIFAR10('./', download=True,
        transform=transforms.Compose([
        transforms.ToTensor(),
        ]), train=False)
        
        self.val_set = LoadCifar(self.val_set)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )