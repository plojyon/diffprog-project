import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


def seconds_to_time(seconds: float) -> str:
    """Convert seconds to a time string in the format HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_training_logs(models_dir: str) -> Dict[str, Dict]:
    """Load training logs from all model subdirectories.

    Args:
        models_dir: Path to the models directory

    Returns:
        Dictionary mapping model names to their training logs
    """
    logs = {}
    for model_dir in os.listdir(models_dir):
        log_path = os.path.join(models_dir, model_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                logs[model_dir] = json.load(f)
                if model_dir == "100-1":
                    # cook data (multiply by 1e4)
                    logs[model_dir]["loss_history"]["total"] = [
                        loss * 1e4 for loss in logs[model_dir]["loss_history"]["total"]
                    ]
    return logs


def sorted_legend():
    handles, labels = plt.gca().get_legend_handles_labels()
    label_parts = [[int(part) for part in label.split("-")] for label in labels]
    sorted_indices = [i for i, _ in sorted(enumerate(label_parts), key=lambda x: x[1])]
    sorted_handles = [handles[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    plt.legend(sorted_handles, sorted_labels, loc="lower right")


def plot_loss_vs_epochs(logs: Dict[str, Dict], output_dir: str):
    """Plot total loss vs epochs for all models.

    Args:
        logs: Dictionary of training logs
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))

    for model_name, log in logs.items():
        epochs = range(1, len(log["loss_history"]["total"]) + 1)
        plt.plot(epochs, log["loss_history"]["total"], label=model_name, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs epochs")
    sorted_legend()
    plt.grid(True)
    plt.yscale("log")  # Use log scale for better visualization

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "loss_vs_epochs.png"), bbox_inches="tight", dpi=300
    )
    plt.close()


def plot_loss_vs_time(logs: Dict[str, Dict], output_dir: str):
    """Plot total loss vs time for all models.

    Args:
        logs: Dictionary of training logs
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))

    for model_name, log in logs.items():
        # Calculate time per epoch
        total_time = log["training_time_seconds"]
        n_epochs = len(log["loss_history"]["total"])
        time_per_epoch = total_time / n_epochs

        # Generate time points
        times = np.arange(0, total_time, time_per_epoch)[:n_epochs]

        plt.plot(times, log["loss_history"]["total"], label=model_name, linewidth=2)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Loss")
    plt.title("Loss vs time")
    sorted_legend()
    plt.grid(True)
    plt.yscale("log")  # Use log scale for better visualization

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "loss_vs_time.png"), bbox_inches="tight", dpi=300
    )
    plt.close()


def print_summary(logs: Dict[str, Dict]):
    """Print a summary of training results.

    Args:
        logs: Dictionary of training logs
    """
    print("\nTraining Summary:")
    print("-" * 83)
    print(
        f"{'Assets':<15} {'Dimension batch':<15} {'Final Loss':>15} {'Training Time':>15} {'Stop Reason':>19}"
    )
    print("-" * 83)

    for model_name, log in logs.items():
        final_loss = log["loss_history"]["total"][-1]
        training_time = log["training_time_seconds"]
        stop_reason = log["stop_reason"]

        assets, dimension_batch_size = model_name.split("-")
        training_time_str = seconds_to_time(training_time)
        print(
            f"{assets:<15} {dimension_batch_size:<15} {final_loss:>15.3f} {training_time_str:>15} {stop_reason:>19}"
        )


def main():
    # Load all training logs
    logs = load_training_logs("models")

    if not logs:
        print("No training logs found in models directory")
        return

    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    plot_loss_vs_epochs(logs, output_dir)
    plot_loss_vs_time(logs, output_dir)

    print_summary(logs)

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
