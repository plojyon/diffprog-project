import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class PINN(nn.Module):
    """A simple feedforward neural network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
    ):
        """Initialize the neural network.

        Args:
            input_dim: Number of input dimensions
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output dimensions
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.Tanh()]
        for i in range(len(hidden_dims) - 1):
            layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.Tanh()])
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.net = nn.Sequential(*layers)
        self.net.to(self.device)

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            S: Tensor of shape (batch_size, n_assets) containing stock prices
            t: Tensor of shape (batch_size, 1) containing time points
        """
        return self.net(torch.cat([S, t], dim=1))

    def generate_collocation_points(
        self, count: int, min_S: List[float], max_S: List[float], max_T: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate uniformly randomly scattered collocation points."""
        n_assets = len(min_S)

        # Generate random points for each asset
        S_points = []
        for i in range(n_assets):
            S_i = torch.tensor(
                np.random.uniform(min_S[i], max_S[i], (count, 1)),
                dtype=torch.float32,
                requires_grad=True,
                device=self.device,
            )
            S_points.append(S_i)

        # Stack all asset prices
        S = torch.cat(S_points, dim=1)

        # Generate time points
        t = torch.tensor(
            np.random.uniform(0, max_T, (count, 1)),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )

        return S, t
