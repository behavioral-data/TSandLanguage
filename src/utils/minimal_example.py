"""
A minimal example of using PyTorch Lightning to train a model. Usefull for debugging.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np


class DummyDataset(Dataset):
    def __init__(self, size=1000, input_dim=10, output_dim=1, noise_std=0.1):
        self.inputs = torch.randn(size, input_dim)
        noise = torch.randn(size, output_dim) * noise_std
        self.targets = torch.sin(self.inputs) + noise
        self.targets = self.targets.view(-1, 1)  # reshape targets to (size, 1)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class DummyModel(pl.LightningModule):
    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


def main(args):
    train_data = DummyDataset()
    train_loader = DataLoader(train_data, batch_size=32)

    # initialize the model
    model = DummyModel()

    # initialize the trainer
    if args.use_cuda:
        trainer = pl.Trainer(gpus=1, max_epochs=20)
    else:
        trainer = pl.Trainer(max_epochs=20)

    # train the model
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', help='Train on a CUDA-enabled GPU')
    args = parser.parse_args()

    main(args)
