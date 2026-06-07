"""LGC estimation pipelines for the Monte Carlo comparison.

The four estimators under comparison are:

- :func:`raw_lgc`              -- LGC applied directly to (X_1, X_2).
- :func:`canonical_lgc`        -- empirical-quantile transform of each
                                  margin, then LGC on the Gaussian
                                  pseudo-observations (Berentsen et al. 2014).
- :func:`nts_lgc_oracle`       -- transform each margin through the TRUE
                                  data-generating NTS CDF, then LGC.
                                  Isolates the benefit of the parametric
                                  tail shape from marginal-fit error.
- :func:`nts_lgc_fitted`       -- fit NTS marginals from the sample, then
                                  apply the fitted transform and LGC.
                                  The realistic case.

All four return an (m,) array of local correlation estimates at the
``grid`` points. For the transformed pipelines (canonical and the two
NTS variants), the grid is interpreted in the standardised Z-space
(Gaussian pseudo-observation space) rather than in the original X-space.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .lgc import lgc_on_grid
from .nts import cdf as nts_cdf, fit_mle


# Clipping bound so that Phi^{-1}(U) stays finite when U is at the
# numerical boundary. Phi^{-1}(1e-6) ~= -4.75 -- well outside any region
# the local likelihood would have effective sample size in, and inside
# any region it would.
_EPS = 1e-6


def _empirical_cdf_transform(x: np.ndarray) -> np.ndarray:
    """Rank-based empirical CDF transform with the (rank - 0.5)/n convention.

    Avoids the values 0 and 1 exactly, which would map to +/- infinity
    under Phi^{-1}.
    """
    n = len(x)
    ranks = np.argsort(np.argsort(x)) + 1
    return (ranks - 0.5) / n


def _to_normal_scores(U: np.ndarray) -> np.ndarray:
    """Map uniforms to standard normal pseudo-observations, clipped for safety."""
    return norm.ppf(np.clip(U, _EPS, 1.0 - _EPS))


# --- The four pipelines --------------------------------------------------

def raw_lgc(
    X: np.ndarray, grid: np.ndarray, **kwargs
) -> np.ndarray:
    """LGC on the raw bivariate data with no marginal pre-transform."""
    return lgc_on_grid(X, grid, **kwargs)


def canonical_lgc(
    X: np.ndarray, grid_z: np.ndarray, **kwargs
) -> np.ndarray:
    """Canonical LGC of Berentsen et al. (2014).

    Each margin is transformed to standard normality through the
    rank-based empirical CDF and Phi^{-1}, after which LGC is computed
    on the standardised sample.
    """
    U1 = _empirical_cdf_transform(X[:, 0])
    U2 = _empirical_cdf_transform(X[:, 1])
    Z = np.column_stack([_to_normal_scores(U1), _to_normal_scores(U2)])
    return lgc_on_grid(Z, grid_z, **kwargs)


def nts_lgc_oracle(
    X: np.ndarray,
    grid_z: np.ndarray,
    params1,
    params2,
    **kwargs,
) -> np.ndarray:
    """NTS-transformed LGC using the TRUE NTS parameters of the
    data-generating process for the marginal transform.

    The gap between this estimator and :func:`nts_lgc_fitted` quantifies
    how much of the parametric tail-shape benefit is lost to marginal
    estimation error.
    """
    U1 = nts_cdf(X[:, 0], params1)
    U2 = nts_cdf(X[:, 1], params2)
    Z = np.column_stack([_to_normal_scores(U1), _to_normal_scores(U2)])
    return lgc_on_grid(Z, grid_z, **kwargs)


def nts_lgc_fitted(
    X: np.ndarray, grid_z: np.ndarray, **kwargs
) -> np.ndarray:
    """NTS-transformed LGC with NTS marginal parameters fit from the sample.

    This is the realistic estimator: no oracle access to the truth. The
    NTS marginal fit and the LGC estimation are performed on the same data.
    """
    p1 = fit_mle(X[:, 0])
    p2 = fit_mle(X[:, 1])
    U1 = nts_cdf(X[:, 0], p1)
    U2 = nts_cdf(X[:, 1], p2)
    Z = np.column_stack([_to_normal_scores(U1), _to_normal_scores(U2)])
    return lgc_on_grid(Z, grid_z, **kwargs)
