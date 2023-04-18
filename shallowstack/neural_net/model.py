from typing import Tuple
from shallowstack.neural_net.util import create_output_vector
import torch
from torch import nn, optim
import lightning as pl
import numpy as np


class ValueNetwork(pl.LightningModule):
    def __init__(self, range_size: int, public_info_size: int):
        super().__init__()
        self.range_size = range_size
        self.public_info_size = public_info_size

        self.fc1 = nn.Linear(range_size * 2 + public_info_size + 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)

        self.value_output = nn.Linear(32, range_size * 2)

    def predict_values(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper function that returns the interesting values
        for other parts of the application
        """
        out = self(x).squeeze(0)
        v1, v2, _ = out.split([self.range_size, self.range_size, 1])
        return v1.detach().numpy(), v2.detach().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r1, r2, _ = x.clone().split(
            [
                self.range_size,
                self.range_size,
                1 + self.public_info_size,
            ],
            dim=1,
        )

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))

        values = self.value_output(x)
        v1, v2 = values.split([self.range_size, self.range_size], dim=1)

        dot_sum = (
            torch.bmm(r1.unsqueeze(1), v1.unsqueeze(2))
            + torch.bmm(r2.unsqueeze(1), v2.unsqueeze(2))
        ).squeeze(1)

        return create_output_vector(v1, v2, dot_sum)

    def training_step(self, batch: torch.Tensor, batch_idx):
        """
        Implements one interation in the training loop
        """
        # Split the batch into input and target values
        x, y = batch.split(
            [
                self.range_size * 2 + 1 + self.public_info_size,
                self.range_size * 2 + 1,
            ],
            dim=1,
        )

        y_hat = self(x)

        loss = nn.functional.huber_loss(y_hat, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx):
        """
        Implements a single validation iteration
        """
        x, y = batch.split(
            [
                self.range_size * 2 + 1 + self.public_info_size,
                self.range_size * 2 + 1,
            ],
            dim=1,
        )

        y_hat = self(x)

        loss = nn.functional.huber_loss(y_hat, y)

        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
