"""Ground-truth LGC surfaces for the Monte Carlo study.

For each copula in the study, we need a "true" LGC surface against which
finite-sample estimators can be compared. With non-Gaussian copulas there
is no closed form, so we compute the surface by running LGC on a very
large sample drawn from the joint distribution with standard-normal
margins. With enough observations the variance of the estimator is small
enough to treat the result as the population object.

For the Gaussian copula the population LGC is identically the underlying
correlation rho, which we also use as a numerical sanity check on the
large-sample estimate.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .copulas import (
    sample_clayton_copula,
    sample_gaussian_copula,
    sample_gumbel_copula,
    sample_t_copula,
)
from .lgc import lgc_on_grid


# --- Evaluation grid -----------------------------------------------------

GRID_AXIS = np.linspace(-3.0, 3.0, 11)
"""1D axis for the standard 11x11 Z-space evaluation grid."""


def make_eval_grid() -> np.ndarray:
    """Return the (121, 2) array of evaluation points used throughout the MC."""
    gx, gy = np.meshgrid(GRID_AXIS, GRID_AXIS)
    return np.column_stack([gx.ravel(), gy.ravel()])


QUANTILE_LEVELS = np.array([0.01, 0.05, 0.5, 0.95, 0.99])
"""Quantile levels at which pointwise bias and variance are reported."""


def quantile_points() -> np.ndarray:
    """Return the (5, 2) array of diagonal evaluation points in Z-space:
    z = (Phi^{-1}(q), Phi^{-1}(q)) for q in QUANTILE_LEVELS."""
    zq = norm.ppf(QUANTILE_LEVELS)
    return np.column_stack([zq, zq])


# --- Sampling in Z-space -------------------------------------------------

def sample_z_space(copula: str, params: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample (n, 2) bivariate observations with standard-normal margins
    and the specified copula. This is the joint that the canonical /
    oracle / fitted estimators all target.
    """
    if copula == "gaussian":
        U = sample_gaussian_copula(n, params["rho"], rng)
    elif copula == "clayton":
        U = sample_clayton_copula(n, params["theta"], rng)
    elif copula == "t":
        U = sample_t_copula(n, params["rho"], params["df"], rng)
    elif copula == "gumbel":
        U = sample_gumbel_copula(n, params["theta"], rng)
    else:
        raise ValueError(f"unknown copula '{copula}'")
    return norm.ppf(np.clip(U, 1e-12, 1.0 - 1e-12))


def compute_ground_truth_surface(
    copula: str,
    params: dict,
    n_gt: int = 20000,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute large-sample LGC on the standard grid and quantile points.

    Returns
    -------
    rho_grid : (121,) array of LGC at the 11x11 grid points.
    rho_quant : (5,) array of LGC at the diagonal quantile points.
    """
    rng = np.random.default_rng(seed)
    Z = sample_z_space(copula, params, n_gt, rng)
    grid = make_eval_grid()
    qpts = quantile_points()
    all_pts = np.vstack([grid, qpts])
    rhos = lgc_on_grid(Z, all_pts)
    return rhos[: grid.shape[0]], rhos[grid.shape[0]:]
