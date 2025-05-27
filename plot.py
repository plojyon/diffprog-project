import os
from typing import List
import matplotlib.pyplot as plt


def save_loss_plot(
    path: str,
    total_losses: List[float],
    data_losses: List[float],
    pde_losses: List[float],
):
    """Save a plot of the training losses to disk.

    Args:
        total_losses: List of total loss values
        data_losses: List of data loss values
        pde_losses: List of PDE loss values
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(total_losses) + 1)

    # Plot total loss and data loss on primary y-axis
    line1 = ax1.plot(epochs, total_losses, "b-", label="Total Loss", linewidth=2)
    line2 = ax1.plot(epochs, data_losses, "g-", label="Data Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total/Data Loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # Create secondary y-axis for PDE loss
    ax2 = ax1.twinx()
    line3 = ax2.plot(epochs, pde_losses, "r-", label="PDE Loss", linewidth=2)
    ax2.set_ylabel("PDE Loss", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title("Training Losses")
    plt.grid(True)

    plot_path = os.path.join(path, "loss.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Loss plot saved to {plot_path}")
