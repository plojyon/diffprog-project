import argparse
import json
import numpy as np
from typing import List, Tuple


def generate_correlation_matrix(n_assets: int) -> List[List[float]]:
    """Generate a valid correlation matrix for n assets.

    Args:
        n_assets: Number of assets

    Returns:
        Correlation matrix as a list of lists
    """
    # correlations are between -0.3 and 0.7; this is typical for financial assets
    correlations = np.random.uniform(-0.3, 0.7, (n_assets, n_assets))
    correlations = (correlations + correlations.T) / 2
    np.fill_diagonal(correlations, 1.0)

    # ensure positive definiteness
    min_eig = np.min(np.real(np.linalg.eigvals(correlations)))
    if min_eig < 0:
        correlations += (-min_eig + 1e-6) * np.eye(n_assets)

    return correlations.tolist()


def generate_data_parameters(n_assets: int) -> dict:
    """Generate data parameters for n-dimensional Black-Scholes model.

    Args:
        n_assets: Number of assets

    Returns:
        Dictionary of parameters
    """
    params = {
        "K": 20.0,  # Strike price
        "T": 1.0,  # Time to maturity
        "r": 0.035,  # Risk-free rate
        "sigmas": np.random.uniform(
            0.15, 0.35, n_assets
        ).tolist(),  # Volatilities between 15% and 35%
        "rho": generate_correlation_matrix(n_assets),
        "N_data": 100,
        "S0": np.random.uniform(15.0, 35.0, n_assets).tolist(),  # Initial prices
        "noise_variance": 0.1,
    }

    return params


def main():
    parser = argparse.ArgumentParser(
        description="Generate data parameters for N-dimensional Black-Scholes model"
    )
    parser.add_argument(
        "--n_assets", type=int, required=True, help="Number of assets/dimensions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_parameters.json",
        help="Output file path (default: data_parameters.json)",
    )

    args = parser.parse_args()

    # Generate parameters
    params = generate_data_parameters(args.n_assets)

    # Save to file
    with open(args.output, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
