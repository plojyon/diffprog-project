import argparse
import json
import numpy as np
from typing import Tuple
from scipy.stats import multivariate_normal


def generate_correlated_brownian_motion(
    n_steps: int, n_assets: int, rho: np.ndarray, dt: float
) -> np.ndarray:
    """Generate correlated Brownian motion paths.

    Args:
        n_steps: Number of time steps
        n_assets: Number of assets
        rho: Correlation matrix of shape (n_assets, n_assets)
        dt: Time step size

    Returns:
        Array of shape (n_steps, n_assets) containing the Brownian motion paths
    """
    Z = multivariate_normal.rvs(mean=np.zeros(n_assets), cov=rho, size=n_steps)
    dW = Z * np.sqrt(dt)
    W = np.cumsum(dW, axis=0)

    return W


def generate_multi_asset_paths(
    params: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic price paths for multiple assets.

    Args:
        params: Dictionary containing simulation parameters

    Returns:
        Tuple of (S, t, C) where:
        - S: Array of shape (n_paths, n_assets) containing final prices
        - t: Array of shape (n_paths, 1) containing time points
        - C: Array of shape (n_paths, 1) containing option prices
    """
    n_assets = len(params["sigmas"])
    n_paths = params["N_data"]
    T = params["T"]
    r = params["r"]
    sigmas = params["sigmas"]
    rho = np.array(params["rho"])
    S0 = params["S0"]
    K = params["K"]

    t = np.random.uniform(0, T, (n_paths, 1))

    W = generate_correlated_brownian_motion(n_paths, n_assets, rho, T)

    # final prices
    drift = (r - 0.5 * np.array(sigmas) ** 2) * T
    diffusion = np.array(sigmas) * W
    S = S0 * np.exp(drift + diffusion)

    # option prices
    basket_value = np.mean(S, axis=1, keepdims=True)
    C = np.maximum(basket_value - K, 0) * np.exp(-r * (T - t))
    C += np.random.normal(0, params["noise_variance"], size=C.shape)

    return S, t, C


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for multi-asset options"
    )
    parser.add_argument(
        "--data_parameters",
        type=str,
        default="data_parameters.json",
        help="Path to the data parameters file (default: data_parameters.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data.json",
        help="Where to save the data file (default: data.json)",
    )

    args = parser.parse_args()
    with open(args.data_parameters, "r", encoding="utf-8") as f:
        params = json.load(f)

    S, t, C = generate_multi_asset_paths(params)

    data = {
        "S": S.tolist(),
        "t": t.tolist(),
        "C": C.tolist(),
        "r": params["r"],
        "sigmas": params["sigmas"],
        "rho": params["rho"],
    }

    json.dump(data, open(args.output, "w", encoding="utf-8"))
    print(f"Created {args.output}")
