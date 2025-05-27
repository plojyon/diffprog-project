import torch
import torch.autograd as autograd
from typing import List


def pde_residual(
    model,
    S: torch.Tensor,
    t: torch.Tensor,
    r: float,
    sigmas: List[float],
    rho: torch.Tensor,
    selected_dims: List[int],
):
    """Calculate the PDE residual for the multi-asset Black-Scholes equation.

    Args:
        model: The neural network model
        S: Tensor of shape (batch_size, n_assets) containing stock prices
        t: Tensor of shape (batch_size, 1) containing time points
        r: Risk-free interest rate
        sigmas: List of volatilities for each asset
        rho: Correlation matrix of shape (n_assets, n_assets)
        selected_dims: List of indices of the dimensions to use for the PDE residual
    """
    C = model(S, t)
    # batch_size = S.shape[0]
    n_assets = S.shape[1]

    # Time derivative
    dC_dt = autograd.grad(
        C, t, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True
    )[0]

    # First derivatives with respect
    dC_dS = []
    for i in selected_dims:
        dC_dSi = autograd.grad(
            C, S, grad_outputs=torch.ones_like(C), create_graph=True, retain_graph=True
        )[0][:, i : i + 1]
        dC_dS.append(dC_dSi)

    # Second derivatives and cross-derivatives
    d2C_dS2 = []
    for i_idx, i in enumerate(selected_dims):
        row = []
        for j_idx, j in enumerate(selected_dims):
            if i == j:
                # Second derivative
                d2C_dSi2 = autograd.grad(
                    dC_dS[i_idx],
                    S,
                    grad_outputs=torch.ones_like(C),
                    create_graph=True,
                    retain_graph=True,
                )[0][:, i : i + 1]
                row.append(d2C_dSi2)
            else:
                # Cross derivative
                d2C_dSidSj = autograd.grad(
                    dC_dS[i_idx],
                    S,
                    grad_outputs=torch.ones_like(C),
                    create_graph=True,
                    retain_graph=True,
                )[0][:, j : j + 1]
                row.append(d2C_dSidSj)
        d2C_dS2.append(row)

    # Compute the PDE residual
    residual = dC_dt

    # Add diffusion terms
    for i_idx, i in enumerate(selected_dims):
        for j_idx, j in enumerate(selected_dims):
            residual += (
                0.5
                * sigmas[i]
                * sigmas[j]
                * rho[i, j]
                * S[:, i : i + 1]
                * S[:, j : j + 1]
                * d2C_dS2[i_idx][j_idx]
            )

    # Add drift terms
    for i_idx, i in enumerate(selected_dims):
        residual += r * S[:, i : i + 1] * dC_dS[i_idx]

    # Add discounting term
    residual -= r * C

    return residual
