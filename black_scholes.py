from typing import TypedDict, List, Tuple

import torch
from tqdm import tqdm
from pinn import PINN
from pde import pde_residual
import os
import time
import json


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

        os.makedirs(config["path"], exist_ok=True)

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

        # Initialize lists to track losses
        total_losses = []
        data_losses = []
        pde_losses = []

        # Start timing
        start_time = time.time()

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

            # Track losses
            total_losses.append(loss.item())
            data_losses.append(loss_data.item())
            pde_losses.append(loss_pde.item())

            pbar.set_postfix(
                {
                    "Total loss": f"{loss.item():.5f}",
                    "Data loss": f"{loss_data.item():.5f}",
                    "Residual loss": f"{loss_pde.item():.5f}",
                }
            )

        # Calculate training time
        training_time = time.time() - start_time

        # Save the loss plot
        self._save_loss_plot(total_losses, data_losses, pde_losses)
        self._save_training_log(total_losses, data_losses, pde_losses, training_time)

    def _save_loss_plot(self, total_losses: List[float], data_losses: List[float], pde_losses: List[float]):
        """Save a plot of the training losses to disk.
        
        Args:
            total_losses: List of total loss values
            data_losses: List of data loss values
            pde_losses: List of PDE loss values
        """
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(total_losses) + 1)

        # Plot total loss and data loss on primary y-axis
        line1 = ax1.plot(epochs, total_losses, 'b-', label='Total Loss', linewidth=2)
        line2 = ax1.plot(epochs, data_losses, 'g-', label='Data Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total/Data Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Create secondary y-axis for PDE loss
        ax2 = ax1.twinx()
        line3 = ax2.plot(epochs, pde_losses, 'r-', label='PDE Loss', linewidth=2)
        ax2.set_ylabel('PDE Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')

        plt.title('Training Losses')
        plt.grid(True)

        plot_path = os.path.join(self.config["path"], "loss.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Loss plot saved to {plot_path}")

    def _save_training_log(
        self, 
        total_losses: List[float], 
        data_losses: List[float], 
        pde_losses: List[float],
        training_time: float
    ):
        """Save training metrics to a JSON file.
        
        Args:
            total_losses: List of total loss values
            data_losses: List of data loss values
            pde_losses: List of PDE loss values
            training_time: Total training time in seconds
        """
        log_data = {
            "training_time_seconds": training_time,
            "loss_history": {
                "total": total_losses,
                "data": data_losses,
                "pde": pde_losses
            },
        }

        log_path = os.path.join(self.config["path"], "training_log.json")
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"Training log saved to {log_path}")

    def export(self):
        """Export the trained model to a file."""
        weights_path = os.path.join(self.config["path"], "weights.pth")
        torch.save(self.model.state_dict(), weights_path)
        print(f"Model weights saved to {weights_path}")

    def predict(self, S_eval: torch.Tensor, t_eval: torch.Tensor) -> torch.Tensor:
        """Predict option prices for given S and t."""
        with torch.no_grad():
            S_eval = S_eval.to(self.device)
            t_eval = t_eval.to(self.device)
            return self.model(S_eval, t_eval)
