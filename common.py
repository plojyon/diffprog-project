import numpy as np
import torch
from scipy.stats import norm


def black_scholes(S: np.array, K: float, T: float, r: float, sigma: float) -> np.array:
    """Calculate the price of a European call.

    Args:
        S (np.array): Current stock price.
        K (float): Strike price.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying asset (annualized).
    Returns:
        np.array: The price of the European call option.
    """
    S = np.array(S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
