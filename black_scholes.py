import torch
import torch.autograd as autograd
import numpy as np
from torch import nn
from typing import TypedDict


from tqdm import tqdm


def pde_residual(model, S, t, r, sigma):
    """Calculate the PDE residual for the Black-Scholes equation."""
    C = model(S, t)
    dC_dt = autograd.grad(
        C, t, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True
    )[0]
    dC_dS = autograd.grad(
        C, S, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True
    )[0]
    d2C_dS2 = autograd.grad(
        dC_dS, S, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True
    )[0]
    return dC_dt + 0.5 * sigma**2 * S**2 * d2C_dS2 + r * S * dC_dS - r * C


class PINN(nn.Module):
    """A simple feedforward neural network."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, S, t):
        return self.net(torch.cat([S, t], dim=1))


class BSConfig(TypedDict):
    """Configuration for the Black-Scholes PINN model."""

    epochs: int
    lr: float
    alpha: float
    path: str
    colloc_min_S: float
    colloc_max_S: float
    colloc_max_T: float


class BlackScholesPINN:
    """A Physics-Informed Neural Network (PINN) for the Black-Scholes equation."""
    def __init__(self, config: BSConfig):
        self.config = config
        self.model = PINN()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

    def _data_loss(self, C_pred, C_data):
        """Calculate the loss on the supplied data points."""
        return torch.mean((C_pred - C_data) ** 2)

    def _pde_loss(self, S_colloc, t_colloc, r, sigma):
        """Calculate the PDE residual loss on collocation points."""
        pde = pde_residual(self.model, S_colloc, t_colloc, r, sigma)
        return torch.mean(pde**2)

    def train(self, X):
        """Train the PINN model."""
        S_data, t_data, C_data, r, sigma = X

        pbar = tqdm(range(self.config["epochs"]), desc="Training", unit="epoch")
        for epoch in pbar:
            self.optimizer.zero_grad()
            S_colloc, t_colloc = self._generate_collocation_points()

            C_pred = self.model(S_data, t_data)
            loss_data = self._data_loss(C_pred, C_data)
            loss_pde = self._pde_loss(S_colloc, t_colloc, r, sigma)
            loss = loss_data + loss_pde

            loss.backward()
            self.optimizer.step()

            pbar.set_postfix(
                {
                    "Total loss": f"{loss.item():.5f}",
                    "Collocation point loss": f"{loss_data.item():.5f}",
                    "PDE residual": f"{loss_pde.item():.5f}",
                }
            )

    def _generate_collocation_points(self):
        """Generate uniformly randomly scattered collocation points."""
        S = torch.tensor(
            np.random.uniform(
                self.config["colloc_min_S"], self.config["colloc_max_S"], (self.config["colloc_count"], 1)
            ),
            dtype=torch.float32,
            requires_grad=True,
        )
        t = torch.tensor(
            np.random.uniform(0, self.config["colloc_max_T"], (self.config["colloc_count"], 1)),
            dtype=torch.float32,
            requires_grad=True,
        )
        return S, t

    def export(self, path="model.pth"):
        """Export the trained model to a file."""
        torch.save(self.model.state_dict(), path)

    def predict(self, S_eval, t_eval):
        """Predict option prices for given S and t."""
        with torch.no_grad():
            return self.model(S_eval, t_eval)
