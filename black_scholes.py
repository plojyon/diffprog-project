from typing import TypedDict, List, Tuple

import torch
from tqdm import tqdm
from pinn import PINN
from pde import pde_residual


class BSConfig(TypedDict):
    """Configuration for the multi-asset Black-Scholes PINN model."""

    epochs: int
    lr: float
    alpha: float
    path: str
    colloc_min_S: List[float]  # Minimum price for each asset
    colloc_max_S: List[float]  # Maximum price for each asset
    colloc_max_T: float
    colloc_count: int
    hidden_dims: List[int]


class BlackScholesPINN:
    """A Physics-Informed Neural Network (PINN) for the multi-asset Black-Scholes equation."""

    def __init__(self, config: BSConfig):
        self.config = config
        self.n_assets = len(config["colloc_min_S"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Input dimension is n_assets + 1 (for time)
        self.model = PINN(
            input_dim=self.n_assets + 1, hidden_dims=config["hidden_dims"]
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])

    def _data_loss(self, C_pred: torch.Tensor, C_data: torch.Tensor) -> torch.Tensor:
        """Calculate the loss on the supplied data points."""
        return torch.mean((C_pred - C_data) ** 2)

    def _pde_loss(
        self,
        S_colloc: torch.Tensor,
        t_colloc: torch.Tensor,
        r: float,
        sigmas: List[float],
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the PDE residual loss on collocation points."""
        pde = pde_residual(self.model, S_colloc, t_colloc, r, sigmas, rho)
        return torch.mean(pde**2)

    def train(
        self,
        X: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, float, List[float], torch.Tensor
        ],
    ):
        """Train the PINN model.

        Args:
            X: Tuple containing (S_data, t_data, C_data, r, sigmas, rho)
               where S_data has shape (batch_size, n_assets)
        """
        S_data, t_data, C_data, r, sigmas, rho = X

        # Move data to device
        S_data = S_data.to(self.device)
        t_data = t_data.to(self.device)
        C_data = C_data.to(self.device)
        rho = rho.to(self.device)

        pbar = tqdm(range(self.config["epochs"]), desc="Training", unit="epoch")
        for _ in pbar:
            self.optimizer.zero_grad()
            S_colloc, t_colloc = self.model.generate_collocation_points(
                count=self.config["colloc_count"],
                min_S=self.config["colloc_min_S"],
                max_S=self.config["colloc_max_S"],
                max_T=self.config["colloc_max_T"],
            )

            # Move collocation points to device
            S_colloc = S_colloc.to(self.device)
            t_colloc = t_colloc.to(self.device)

            C_pred = self.model(S_data, t_data)
            loss_data = self._data_loss(C_pred, C_data)
            loss_pde = self._pde_loss(S_colloc, t_colloc, r, sigmas, rho)
            loss = loss_data + self.config["alpha"] * loss_pde

            loss.backward()
            self.optimizer.step()

            pbar.set_postfix(
                {
                    "Total loss": f"{loss.item():.5f}",
                    "Data loss": f"{loss_data.item():.5f}",
                    "Residual loss": f"{loss_pde.item():.5f}",
                }
            )

    def export(self):
        """Export the trained model to a file."""
        torch.save(self.model.state_dict(), self.config["path"])

    def predict(self, S_eval: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
        """Predict option prices for given S and t."""
        with torch.no_grad():
            S_eval = S_eval.to(self.device)
            t_eval = t_eval.to(self.device)
            return self.model(S_eval, t_eval)
