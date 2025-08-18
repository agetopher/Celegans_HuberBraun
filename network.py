
## Optimization Network

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data 
import numpy as np
from typing import Optional

class NeuralNetwork(nn.Module):
    def __init__(
            self,
            input_dim=1,
            output_dim=5,
            n_units=100,
            epochs=1000,
            loss=nn.MSELoss(),
            lr=1e-3,
            ) -> None:
        super(NeuralNetwork, self).__init__()

        self.epochs = epochs
        self.loss = loss
        self.lr = lr
        self.n_unit = n_units

        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_units),
            nn.SiLU(),
            nn.Linear(n_units, n_units),
            nn.SiLU(),
            nn.Linear(n_units, n_units),
            nn.SiLU(),
            nn.Linear(n_units, n_units),
            nn.SiLU(),
        )
        self.out = nn.Linear(n_units, output_dim)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)
        return out

    def fit(self, X, y):
        dataset = data.TensorDataset(X, y)
