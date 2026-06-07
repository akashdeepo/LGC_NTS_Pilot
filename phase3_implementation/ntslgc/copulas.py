"""Copula samplers for the Monte Carlo study.

Each sampler returns an (n, 2) array of uniform pseudo-observations
(U_1, U_2). To obtain bivariate samples with NTS marginals --- the
data-generating process of interest in this pilot --- pass the uniforms
through :func:`apply_nts_marginals`.

Parameter conventions (so Kendall's tau is comparable across families):
- Gaussian: rho is the correlation of the underlying normals.
            tau = (2/pi) * arcsin(rho).
- Clayton:  theta > 0; tau = theta / (theta + 2).
- t:        rho is the correlation of the underlying t, df = degrees of freedom.
            tau = (2/pi) * arcsin(rho).
- Gumbel:   theta >= 1; tau = (theta - 1) / theta.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student_t


# --- Gaussian copula -----------------------------------------------------

def sample_gaussian_copula(
    n: int, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample (U_1, U_2) from the bivariate Gaussian copula with correlation rho."""
    cov = np.array([[1.0, rho], [rho, 1.0]])
    Z = rng.multivariate_normal(np.zeros(2), cov, size=n)
    return norm.cdf(Z)


# --- Clayton copula ------------------------------------------------------

def sample_clayton_copula(
    n: int, theta: float, rng: np.random.Generator
) -> np.ndarray:
    """Conditional-inverse sampling for the bivariate Clayton copula.

    For theta > 0: lower-tail dependence with coefficient 2^{-1/theta}.
    For theta -> 0 the copula tends to independence (use Gaussian rho=0 instead).
    """
    if theta <= 0:
        raise ValueError("Clayton theta must be > 0")
    u1 = rng.uniform(0.0, 1.0, size=n)
    v = rng.uniform(0.0, 1.0, size=n)
    # Conditional inverse: U2 | U1 has cdf given by C_{2|1}, invert by V.
    u2 = (u1 ** (-theta) * (v ** (-theta / (1.0 + theta)) - 1.0) + 1.0) ** (-1.0 / theta)
    return np.column_stack([u1, u2])


# --- t-copula ------------------------------------------------------------

def sample_t_copula(
    n: int, rho: float, df: float, rng: np.random.Generator
) -> np.ndarray:
    """Bivariate t-copula sampler.

    Sample multivariate t via the Gaussian / chi-squared mixture, then
    transform each margin to a uniform through the univariate t CDF.
    Both margins have the same degrees of freedom.
    """
    R = np.array([[1.0, rho], [rho, 1.0]])
    Z = rng.multivariate_normal(np.zeros(2), R, size=n)
    chi2 = rng.chisquare(df, size=n)
    T = Z * np.sqrt(df / chi2)[:, None]
    return student_t.cdf(T, df)


# --- Gumbel copula -------------------------------------------------------

def _sample_positive_stable(
    alpha: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Positive alpha-stable variable with Laplace transform E[exp(-tS)] = exp(-t^alpha).

    Implements the Chambers--Mallows--Stuck construction
    (Devroye 1986, Ch.~IV.6) for alpha in (0, 1).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    U = rng.uniform(0.0, np.pi, size=size)
    E = rng.exponential(1.0, size=size)
    a = np.sin(alpha * U) / np.power(np.sin(U), 1.0 / alpha)
    b = np.power(np.sin((1.0 - alpha) * U) / E, (1.0 - alpha) / alpha)
    return a * b


def sample_gumbel_copula(
    n: int, theta: float, rng: np.random.Generator
) -> np.ndarray:
    """Marshall--Olkin sampling for the bivariate Gumbel copula.

    Sample S ~ positive-stable(alpha = 1/theta), then independent
    E_1, E_2 ~ Exp(1), and form U_j = exp(-(E_j / S)^{1/theta}).
    The Laplace transform of S is the inverse of the Gumbel generator.

    For theta = 1 the Gumbel reduces to the independence copula.
    """
    if theta < 1.0:
        raise ValueError("Gumbel theta must be >= 1")
    if theta == 1.0:
        return rng.uniform(0.0, 1.0, size=(n, 2))
    alpha = 1.0 / theta
    S = _sample_positive_stable(alpha, n, rng)
    E1 = rng.exponential(1.0, size=n)
    E2 = rng.exponential(1.0, size=n)
    U1 = np.exp(-np.power(E1 / S, 1.0 / theta))
    U2 = np.exp(-np.power(E2 / S, 1.0 / theta))
    return np.column_stack([U1, U2])


# --- NTS marginal applicator --------------------------------------------

def apply_nts_marginals(
    U: np.ndarray, params1, params2
) -> np.ndarray:
    """Map uniform pseudo-observations to bivariate samples with NTS marginals.

    X_j = F_NTS^{-1}(U_j; params_j) for j = 1, 2. The copula of (X_1, X_2)
    is the copula of (U_1, U_2), since the marginal transform is monotone.
    """
    from .nts import ppf  # local import avoids any circular concerns
    X1 = ppf(U[:, 0], params1)
    X2 = ppf(U[:, 1], params2)
    return np.column_stack([X1, X2])


# --- Parameter helpers (Kendall's tau -> copula parameter) ---------------

def gaussian_rho_for_tau(tau: float) -> float:
    """Pearson rho such that the Gaussian copula has Kendall's tau = tau."""
    return float(np.sin(np.pi * tau / 2.0))


def clayton_theta_for_tau(tau: float) -> float:
    """Clayton theta such that the copula has Kendall's tau = tau (tau > 0)."""
    if not (0.0 < tau < 1.0):
        raise ValueError("Clayton requires tau in (0, 1)")
    return 2.0 * tau / (1.0 - tau)


def gumbel_theta_for_tau(tau: float) -> float:
    """Gumbel theta such that the copula has Kendall's tau = tau (tau in [0, 1))."""
    if not (0.0 <= tau < 1.0):
        raise ValueError("Gumbel requires tau in [0, 1)")
    return 1.0 / (1.0 - tau)
