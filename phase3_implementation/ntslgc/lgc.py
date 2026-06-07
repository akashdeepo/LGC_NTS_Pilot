"""Bivariate Local Gaussian Correlation (LGC).

Implements the estimator of Tjostheim & Hufthammer (2013, J. Econometrics):
at each point x = (x1, x2) in R^2, approximate the joint density locally by
a bivariate Gaussian with parameters theta(x) = (mu1, mu2, sigma1, sigma2, rho),
and read off rho as the Local Gaussian Correlation at x.

Estimation is via kernel-weighted local likelihood. Given observations
(X1_i, X2_i) for i = 1..n, we maximize

    L_n(theta; x) = (1/n) sum_i K_b(X_i - x) log phi(X_i; theta)

where K_b is a product Gaussian kernel with bandwidth b and phi is the
bivariate Gaussian density with parameters theta.

The bandwidth follows the plug-in rule b = 1.75 * n^(-1/6) from the
foundational paper, applied per dimension to standardized data.

Reference: Tjostheim, D. and Hufthammer, K.O. (2013). Local Gaussian
correlation: A new measure of dependence. Journal of Econometrics 172(1),
33-48.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def plugin_bandwidth(n: int, scale: float = 1.75) -> float:
    """Plug-in bandwidth b = scale * n^(-1/6).

    Parameters
    ----------
    n : int
        Sample size.
    scale : float
        Scaling constant, default 1.75 per Tjostheim-Hufthammer (2013).

    Returns
    -------
    float
        Bandwidth value.
    """
    return scale * n ** (-1.0 / 6.0)


def _bivariate_gaussian_log_density(
    X: np.ndarray,
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    rho: float,
) -> np.ndarray:
    """Log density of a bivariate Gaussian evaluated at rows of X.

    Parameters
    ----------
    X : (n, 2) array
    mu1, mu2 : floats
        Means.
    sigma1, sigma2 : floats
        Standard deviations (must be positive).
    rho : float
        Correlation, in (-1, 1).

    Returns
    -------
    (n,) array of log-densities.
    """
    z1 = (X[:, 0] - mu1) / sigma1
    z2 = (X[:, 1] - mu2) / sigma2
    one_minus_rho2 = 1.0 - rho * rho
    quad = (z1 * z1 - 2.0 * rho * z1 * z2 + z2 * z2) / one_minus_rho2
    log_norm = -np.log(2.0 * np.pi) - np.log(sigma1) - np.log(sigma2) - 0.5 * np.log(one_minus_rho2)
    return log_norm - 0.5 * quad


def _kernel_weights(X: np.ndarray, x: np.ndarray, b: float) -> np.ndarray:
    """Product Gaussian kernel K_b(X_i - x).

    Returns (n,) array of kernel values (not normalized weights).
    K_b is a bivariate Gaussian density with mean 0 and covariance b^2 * I.
    """
    diff = (X - x) / b
    return np.exp(-0.5 * np.sum(diff * diff, axis=1)) / (2.0 * np.pi * b * b)


def _integral_penalty(
    x: np.ndarray,
    mu1: float,
    mu2: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    b: float,
) -> float:
    """Compute integral K_b(y - x) f(y; theta) dy in closed form.

    For bivariate Gaussian f with mean (mu1, mu2) and covariance Sigma, and
    product Gaussian kernel K_b with bandwidth b, this integral equals the
    bivariate Gaussian density N(x; mu, Sigma + b^2 * I). This follows from
    the Gaussian convolution identity.

    Returns a scalar.
    """
    # Elements of Sigma + b^2 * I:
    s11 = sigma1 * sigma1 + b * b
    s22 = sigma2 * sigma2 + b * b
    s12 = rho * sigma1 * sigma2
    det = s11 * s22 - s12 * s12
    if det <= 0.0:
        return np.inf  # Guard against numerical failure.

    d1 = x[0] - mu1
    d2 = x[1] - mu2
    # Quadratic form (x - mu)^T (Sigma + b^2 I)^(-1) (x - mu) for 2x2 case.
    quad = (s22 * d1 * d1 - 2.0 * s12 * d1 * d2 + s11 * d2 * d2) / det
    return np.exp(-0.5 * quad) / (2.0 * np.pi * np.sqrt(det))


def _neg_local_loglik(
    params: np.ndarray,
    X: np.ndarray,
    x: np.ndarray,
    b: float,
) -> float:
    """Negative local log-likelihood, Tjostheim-Hufthammer (2013) eq. 2.3.

    L_n(theta; x) = (1/n) sum_i K_b(X_i - x) log f(X_i; theta)
                    - integral K_b(y - x) f(y; theta) dy

    Parameters packed as [mu1, mu2, log_sigma1, log_sigma2, atanh_rho] so
    the optimizer can work in unconstrained space.
    """
    mu1, mu2, log_s1, log_s2, atanh_rho = params
    sigma1 = np.exp(log_s1)
    sigma2 = np.exp(log_s2)
    rho = np.tanh(atanh_rho)

    n = X.shape[0]
    weights = _kernel_weights(X, x, b)
    log_dens = _bivariate_gaussian_log_density(X, mu1, mu2, sigma1, sigma2, rho)

    first = float(np.sum(weights * log_dens)) / n
    second = _integral_penalty(x, mu1, mu2, sigma1, sigma2, rho, b)

    loglik = first - second
    return -loglik


def _local_sample_moments(
    X: np.ndarray, x: np.ndarray, b: float
) -> tuple[float, float, float, float, float]:
    """Kernel-weighted local sample mean, sd, and correlation.

    These form a sensible initialization for the local MLE.
    """
    weights = _kernel_weights(X, x, b)
    w_sum = float(np.sum(weights))
    if w_sum <= 0:
        # Fallback to global moments if no weight in neighborhood.
        m = X.mean(axis=0)
        s = X.std(axis=0, ddof=1)
        r = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
        return m[0], m[1], s[0], s[1], np.clip(r, -0.99, 0.99)

    # Weighted means.
    mu1 = float(np.sum(weights * X[:, 0])) / w_sum
    mu2 = float(np.sum(weights * X[:, 1])) / w_sum
    # Weighted second moments.
    d1 = X[:, 0] - mu1
    d2 = X[:, 1] - mu2
    var1 = float(np.sum(weights * d1 * d1)) / w_sum
    var2 = float(np.sum(weights * d2 * d2)) / w_sum
    cov = float(np.sum(weights * d1 * d2)) / w_sum
    sd1 = max(np.sqrt(var1), 1e-3)
    sd2 = max(np.sqrt(var2), 1e-3)
    r = cov / (sd1 * sd2)
    r = float(np.clip(r, -0.99, 0.99))
    return mu1, mu2, sd1, sd2, r


def _estimate_lgc_at_point(
    X: np.ndarray,
    x: np.ndarray,
    b: float,
    init: np.ndarray | None = None,
) -> tuple[float, float, float, float, float]:
    """Estimate local Gaussian parameters (mu1, mu2, sigma1, sigma2, rho) at x.

    Initialization: kernel-weighted local sample moments at x. This is much
    more stable than a global initialization, especially in low-density regions
    and near-independence cases where the local likelihood is flat in rho.

    Uses L-BFGS-B with bounds to prevent sigma or |rho| blowups during line
    search. Without bounds, the optimizer can wander into regions where
    (1 - rho^2) = 0 or sigma overflows, producing numerical warnings that
    don't affect the final estimate but pollute the log.
    """
    if init is None:
        mu1, mu2, sd1, sd2, r = _local_sample_moments(X, x, b)
        init = np.array([mu1, mu2, np.log(sd1), np.log(sd2), np.arctanh(r)])

    # Bounds in the unconstrained parameterization.
    # mu_j: allow to wander by a few standard deviations from initialization.
    # log_sigma_j: sigma in [1e-3, 1e2].
    # atanh_rho: rho in [-0.995, 0.995].
    mu_half_width = 10.0
    bounds = [
        (init[0] - mu_half_width, init[0] + mu_half_width),
        (init[1] - mu_half_width, init[1] + mu_half_width),
        (np.log(1e-3), np.log(1e2)),
        (np.log(1e-3), np.log(1e2)),
        (np.arctanh(-0.995), np.arctanh(0.995)),
    ]

    # Suppress numerical warnings from transient exploration near bounds.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        res = minimize(
            _neg_local_loglik,
            init,
            args=(X, x, b),
            method="L-BFGS-B",
            bounds=bounds,
        )

    mu1, mu2, log_s1, log_s2, atanh_rho = res.x
    return mu1, mu2, np.exp(log_s1), np.exp(log_s2), np.tanh(atanh_rho)


def lgc_on_grid(
    X: np.ndarray,
    grid: np.ndarray,
    b: float | None = None,
    standardize: bool = True,
    return_effective_n: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Estimate LGC on a grid of evaluation points.

    Parameters
    ----------
    X : (n, 2) array of observations.
    grid : (m, 2) array of points at which to evaluate LGC.
    b : float, optional
        Bandwidth. If None, uses plugin_bandwidth(n).
    standardize : bool
        If True, standardize X and grid using the sample mean/std of X
        before estimation, then return results in the standardized scale.
        This matches the typical LGC estimation convention and makes the
        bandwidth scale-invariant.
    return_effective_n : bool
        If True, also return the effective sample size at each grid point,
        defined as (sum of kernel weights) / (max kernel value). This
        measures how many observations contribute meaningfully to each
        local estimate and is useful for masking unreliable regions of
        the LGC surface.

    Returns
    -------
    rhos : (m,) array of estimated rho values.
    eff_n : (m,) array of effective sample sizes (only if return_effective_n).
    """
    X = np.asarray(X, dtype=float)
    grid = np.asarray(grid, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must be an (n, 2) array")
    if grid.ndim != 2 or grid.shape[1] != 2:
        raise ValueError("grid must be an (m, 2) array")

    if standardize:
        m = X.mean(axis=0)
        s = X.std(axis=0, ddof=1)
        Xs = (X - m) / s
        grid_s = (grid - m) / s
    else:
        Xs = X
        grid_s = grid

    n = Xs.shape[0]
    if b is None:
        b = plugin_bandwidth(n)

    # Max kernel value (at the center) used to normalize effective_n.
    k_max = 1.0 / (2.0 * np.pi * b * b)

    rhos = np.empty(grid_s.shape[0])
    eff_n = np.empty(grid_s.shape[0])
    for i, pt in enumerate(grid_s):
        _, _, _, _, rho = _estimate_lgc_at_point(Xs, pt, b, init=None)
        rhos[i] = rho
        if return_effective_n:
            weights = _kernel_weights(Xs, pt, b)
            eff_n[i] = float(np.sum(weights)) / k_max

    if return_effective_n:
        return rhos, eff_n
    return rhos
