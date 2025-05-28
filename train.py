import argparse
import json
import torch
from black_scholes import BlackScholesPINN

import warnings

warnings.filterwarnings(
    "ignore", message="Attempting to run cuBLAS, but there was no current CUDA context"
)


def load_data(data_path: str):
    """Load training data from a JSON file.

    Args:
        data_path: Path to the data file

    Returns:
        Tuple containing (S_data, t_data, C_data, r, sigmas, rho)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    S = torch.tensor(data["S"], dtype=torch.float32, requires_grad=True, device=device)
    t = torch.tensor(data["t"], dtype=torch.float32, requires_grad=True, device=device)
    C = torch.tensor(data["C"], dtype=torch.float32, device=device)
    r = data["r"]
    sigmas = data["sigmas"]
    rho = torch.tensor(data["rho"], dtype=torch.float32, device=device)

    return S, t, C, r, sigmas, rho


def main(model_parameters: dict, data_path: str, time_limit_minutes: int):
    """Train the multi-asset Black-Scholes PINN model.

    Args:
        model_parameters: Model parameters
        data_path: Path to the training data file
        time_limit_minutes: Time limit in minutes for training
    """
    X = load_data(data_path)

    model = BlackScholesPINN(model_parameters)
    model.train(X, time_limit_minutes * 60)
    model.export()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PINN on the multi-asset Black-Scholes equation"
    )
    parser.add_argument(
        "--model_parameters",
        type=str,
        default="model_parameters.json",
        help="Path to the model parameters file (default: model_parameters.json)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.json",
        help="Path to the training data file (default: data.json)",
    )
    parser.add_argument(
        "--time_limit_minutes",
        type=int,
        default=5,
        help="Time limit in minutes for training (default: 5)",
    )

    args = parser.parse_args()
    with open(args.model_parameters, "r", encoding="utf-8") as f:
        model_parameters = json.load(f)
    main(model_parameters, args.data, args.time_limit_minutes)
