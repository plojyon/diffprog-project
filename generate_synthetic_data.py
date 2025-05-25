import argparse
import json
import numpy as np
from common import black_scholes


def generate_synthetic_data(params):
    S = np.random.uniform(params["min_S"], params["max_S"], (params["N_data"], 1))
    t = np.random.uniform(0, params["T"], (params["N_data"], 1))
    C = black_scholes(S, params["K"], params["T"] - t, params["r"], params["sigma"])
    C += np.random.normal(params["bias"], params["noise_variance"], size=C.shape)
    return S, t, C


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data")
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
    with open(args.data_parameters, "r") as f:
        params = json.load(f)

    S, t, C = generate_synthetic_data(params)
    data = {
        "S": S.tolist(),
        "t": t.tolist(),
        "C": C.tolist(),
        "r": params["r"],
        "sigma": params["sigma"],
    }
    json.dump(data, open(args.output, "w"), indent="\t")

    print(f"Saved to {args.output}")
