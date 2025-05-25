import argparse
import json
import torch
from black_scholes import BlackScholesPINN


def load_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return (
        torch.tensor(data["S"], dtype=torch.float32, requires_grad=True),
        torch.tensor(data["t"], dtype=torch.float32, requires_grad=True),
        torch.tensor(data["C"], dtype=torch.float32),
        data["r"],
        data["sigma"],
    )


def main(model_parameters, data_path):
    X = load_data(data_path)
    model = BlackScholesPINN(model_parameters)
    model.train(X)
    model.export()
    print(f"Model saved to {model_parameters['path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PINN on the Black-Scholes equation"
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

    args = parser.parse_args()
    with open(args.model_parameters, "r") as f:
        model_parameters = json.load(f)
    main(model_parameters, args.data)
