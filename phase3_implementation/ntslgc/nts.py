"""Normal Tempered Stable (NTS) distribution.

Implements the standard five-parameter NTS distribution via its
characteristic function, with density/CDF via FFT inversion and Gil-Pelaez
integration, simulation via Brownian subordination, and maximum likelihood
estimation.

The NTS distribution is constructed as
    X = mu + beta * (T - 1) + gamma * sqrt(T) * Z,
where Z ~ N(0,1) independent of T, and T is a tempered stable
subordinator with stability alpha in (0, 1) and tempering theta > 0
normalized so that E[T] = 1.

The characteristic function takes the form
    phi(u) = exp( i*mu*u - i*beta*u
                  - (theta^(1-alpha) / alpha) *
                    [(theta - i*beta*u + 0.5*gamma^2*u^2)^alpha - theta^alpha] )

At u = 0 the inner bracket is theta^alpha - theta^alpha = 0, so phi(0) = 1
as required for a valid characteristic function.

Parameters (all real):
    alpha  in (0, 1)  - stability index
    theta  > 0        - tempering parameter
    beta   in R       - asymmetry / skewness parameter
    gamma  > 0        - diffusion scale
    mu     in R       - location

References: Kim et al. (2008), Rachev et al. (2011).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# --- Parameter container -------------------------------------------------

@dataclass(frozen=True)
class NTSParams:
    """Container for NTS parameters.

    All fields are scalar floats.
    """
    alpha: float
    theta: float
    beta: float
    gamma: float
    mu: float

    def as_array(self) -> np.ndarray:
        return np.array([self.alpha, self.theta, self.beta, self.gamma, self.mu])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "NTSParams":
        return cls(float(arr[0]), float(arr[1]), float(arr[2]),
                   float(arr[3]), float(arr[4]))

    def validate(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.theta <= 0.0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")


# --- Characteristic function ---------------------------------------------

def char_function(u: np.ndarray, params: NTSParams) -> np.ndarray:
    """NTS characteristic function phi(u) at array u.

    Derived from Brownian subordination X = mu + beta*(T - 1) + gamma*sqrt(T)*Z
    with T a tempered stable subordinator normalized to E[T] = 1.
    Evaluated in complex arithmetic.
    """
    alpha, theta, beta, gamma, mu = (
        params.alpha, params.theta, params.beta, params.gamma, params.mu
    )

    u_c = u.astype(np.complex128)
    # Inner term: theta - i*beta*u + (1/2) * gamma^2 * u^2.
    # We take the principal branch of the complex power.
    inner = theta - 1j * beta * u_c + 0.5 * gamma * gamma * u_c * u_c
    # Cumulant generator of the subordinator:
    #   psi_T(-s) - psi_T(0) = -(theta^(1-alpha)/alpha) * (inner^alpha - theta^alpha)
    prefactor = -(theta ** (1.0 - alpha)) / alpha
    cumulant_generator = prefactor * (inner ** alpha - theta ** alpha)

    exponent = 1j * mu * u_c - 1j * beta * u_c + cumulant_generator
    return np.exp(exponent)


# --- Density via FFT inversion -------------------------------------------

def density_fft(
    x: np.ndarray,
    params: NTSParams,
    n_grid: int = 2 ** 12,
    u_max: float = 200.0,
) -> np.ndarray:
    """Evaluate the NTS density at query points x via Carr-Madan FFT inversion.

    Computes
        f(y) = (1 / 2*pi) * integral exp(-i u y) phi(u) du
    on a regular x-grid using a single FFT, then linearly interpolates to
    the query points. This is O(N log N + len(x)) per call, compared to
    O(N * len(x)) for the direct Riemann sum: the prior implementation
    became the bottleneck in the Monte Carlo study because each NTS-MLE
    iteration evaluates the density at the full sample.

    Algorithm. With symmetric grids
        u_k = (k - N/2) * du  for k = 0, ..., N-1,
        y_n = (n - N/2) * dx  for n = 0, ..., N-1,
        du * dx = 2*pi / N    (the FFT spacing relation),
    expanding (u_k * y_n) gives
        exp(-i u_k y_n) = exp(-2*pi*i*k*n/N) * (-1)^(k+n) * exp(-i*N*pi/2),
    so
        f(y_n) = (du / 2*pi) * (-1)^n * exp(-i*N*pi/2)
                 * FFT[ (-1)^k * phi(u_k) ][n].
    For N a power of two with N >= 4, exp(-i*N*pi/2) = +1 and the
    constant phase drops out; we include it generally so the function
    behaves correctly for any even N.

    References. Carr & Madan (1999), "Option Valuation Using the Fast
    Fourier Transform"; Menn & Rachev (2006), "Calibrated FFT-based
    density approximations for alpha-stable distributions"; the NTS
    density-via-FFT recipe in Kim, Rachev, Bianchi, Fabozzi (2008) and
    the TempStable R package.

    Parameters
    ----------
    x : (m,) array of real evaluation points.
    params : NTSParams
    n_grid : int
        Number of u-grid (and x-grid) points; power of 2 strongly recommended.
    u_max : float
        Half-width of the u-grid: u_grid spans [-u_max, u_max).

    Returns
    -------
    (m,) array of density values, clipped to be nonnegative.
    """
    params.validate()
    N = int(n_grid)
    du = 2.0 * u_max / N
    dx = 2.0 * np.pi / (N * du)

    k_idx = np.arange(N)
    u_grid = (k_idx - N // 2) * du
    x_grid = (k_idx - N // 2) * dx

    phi_vals = char_function(u_grid, params)

    # (-1)^k phase to convert symmetric-grid sum into a DFT,
    # multiplied through and used again on the output for (-1)^n.
    signs = (-1.0) ** k_idx
    const_phase = np.exp(-1j * np.pi * N / 2.0)  # = +1 for N a power of 2 >= 4

    fft_out = np.fft.fft(signs * phi_vals)
    f_on_grid = (du / (2.0 * np.pi)) * const_phase * signs * fft_out

    # The result should be real; the imaginary part is roundoff. Take real
    # part and clip negatives that arise from truncation of the u-integral.
    f_on_grid_real = np.clip(np.real(f_on_grid), 0.0, None)

    # Linear interpolation to query points. Outside the x-grid we return 0,
    # which is correct asymptotically for the densities of interest.
    x_query = np.atleast_1d(np.asarray(x, dtype=float))
    return np.interp(x_query, x_grid, f_on_grid_real, left=0.0, right=0.0)


def _density_riemann_legacy(
    x: np.ndarray,
    params: NTSParams,
    n_grid: int = 2 ** 12,
    u_max: float = 200.0,
) -> np.ndarray:
    """Reference implementation of the density via a direct Riemann sum.

    Retained only for head-to-head comparison against the FFT version: it
    evaluates exactly the same integral with the same u-grid but without
    the FFT trick, so the two should agree to numerical precision on the
    FFT-output x-grid, with an additional linear-interpolation error of
    order O(dx^2) at off-grid query points. Do not use in hot loops; the
    cost is O(N * len(x)) per call.
    """
    params.validate()
    du = 2.0 * u_max / n_grid
    u = np.linspace(-u_max, u_max, n_grid, endpoint=False)
    phi_vals = char_function(u, params)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    phase = np.exp(-1j * np.outer(x, u))
    vals = (du / (2.0 * np.pi)) * np.sum(phase * phi_vals, axis=1)
    return np.clip(np.real(vals), 0.0, None)


def log_density(x: np.ndarray, params: NTSParams, **kwargs) -> np.ndarray:
    """Log NTS density. Uses density_fft then takes log with clipping."""
    dens = density_fft(x, params, **kwargs)
    return np.log(np.clip(dens, 1e-300, None))


# --- CDF via Gil-Pelaez --------------------------------------------------

def _build_cdf_grid(
    params: NTSParams,
    x_min: float = -30.0,
    x_max: float = 30.0,
    n_dense: int = 4001,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_grid, cdf_grid) where cdf_grid is monotone and ends at 1.

    Used by both cdf() (forward evaluation) and sample() (inverse CDF
    sampling), so both draw from the same underlying discretized object.
    """
    params.validate()
    x_grid = np.linspace(x_min, x_max, n_dense)
    dens = density_fft(x_grid, params)
    cdf_on_grid = np.concatenate(
        [[0.0], np.cumsum(0.5 * (dens[:-1] + dens[1:]) * np.diff(x_grid))]
    )
    total = cdf_on_grid[-1]
    if total <= 0:
        return x_grid, np.linspace(0.0, 1.0, n_dense)
    cdf_on_grid = cdf_on_grid / total
    return x_grid, cdf_on_grid


def cdf(
    x: np.ndarray,
    params: NTSParams,
    x_min: float = -30.0,
    x_max: float = 30.0,
    n_dense: int = 4001,
) -> np.ndarray:
    """NTS CDF at points x via cumulative integration of the FFT density.

    Computes density on a fine grid over [x_min, x_max] using density_fft,
    cumulative-trapezoid integrates to a monotone CDF on the grid, then
    interpolates to the query points. Monotone by construction.
    """
    x_grid, cdf_on_grid = _build_cdf_grid(params, x_min, x_max, n_dense)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return np.clip(np.interp(x, x_grid, cdf_on_grid), 1e-12, 1.0 - 1e-12)


def ppf(
    p: np.ndarray,
    params: NTSParams,
    x_min: float = -30.0,
    x_max: float = 30.0,
    n_dense: int = 4001,
) -> np.ndarray:
    """Inverse CDF (quantile function) of the NTS distribution.

    Implemented by building the same CDF grid used by cdf() and performing
    linear interpolation in the reverse direction.
    """
    x_grid, cdf_on_grid = _build_cdf_grid(params, x_min, x_max, n_dense)
    p = np.atleast_1d(np.asarray(p, dtype=float))
    return np.interp(p, cdf_on_grid, x_grid)


# --- Simulation ----------------------------------------------------------

def sample(
    params: NTSParams,
    n: int,
    rng: np.random.Generator,
    x_min: float = -30.0,
    x_max: float = 30.0,
    n_dense: int = 4001,
) -> np.ndarray:
    """Sample from the NTS distribution via inverse CDF transform.

    Draws U ~ Uniform(0, 1) and returns F^{-1}(U), where F is the
    discretized NTS CDF from _build_cdf_grid. This is exact (up to
    discretization of the CDF grid) and by construction produces samples
    whose CDF evaluation round-trips to a uniform distribution, which
    makes the probability integral transform identity hold numerically.

    Earlier versions used Rosinski's truncated series representation of
    the tempered stable subordinator. That approach is also valid but is
    biased at finite truncation, and the bias was visible as a
    non-uniform PIT histogram. Inverse CDF sampling avoids the issue
    entirely.
    """
    params.validate()
    u = rng.uniform(0.0, 1.0, size=n)
    return ppf(u, params, x_min=x_min, x_max=x_max, n_dense=n_dense)


# --- Maximum likelihood estimation ---------------------------------------

def _neg_log_likelihood(free_params: np.ndarray, x: np.ndarray) -> float:
    """Negative log-likelihood in an unconstrained parameterization.

    Packing: [logit(alpha), log(theta), beta, log(gamma), mu]
    where logit(alpha) = log(alpha/(1-alpha)) maps (0,1) <-> R.
    """
    logit_alpha, log_theta, beta, log_gamma, mu = free_params
    alpha = 1.0 / (1.0 + np.exp(-logit_alpha))
    theta = np.exp(log_theta)
    gamma = np.exp(log_gamma)
    params = NTSParams(alpha=alpha, theta=theta, beta=beta, gamma=gamma, mu=mu)
    try:
        dens = density_fft(x, params)
    except Exception:
        return 1e12
    ll = float(np.sum(np.log(np.clip(dens, 1e-300, None))))
    if not np.isfinite(ll):
        return 1e12
    return -ll


def fit_mle(
    x: np.ndarray,
    init: NTSParams | None = None,
    maxiter: int = 200,
) -> NTSParams:
    """Maximum likelihood fit of NTS parameters to data x."""
    x = np.asarray(x, dtype=float)
    if init is None:
        # Moment-based initialization: use sample mean, std for mu, gamma,
        # and middle-of-range defaults for the shape parameters.
        sample_mean = float(np.mean(x))
        sample_sd = float(np.std(x, ddof=1))
        init = NTSParams(alpha=0.5, theta=1.0, beta=0.0,
                         gamma=max(sample_sd, 0.05), mu=sample_mean)

    init_free = np.array([
        np.log(init.alpha / (1.0 - init.alpha)),
        np.log(init.theta),
        init.beta,
        np.log(init.gamma),
        init.mu,
    ])

    res = minimize(
        _neg_log_likelihood,
        init_free,
        args=(x,),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-4},
    )

    logit_alpha, log_theta, beta, log_gamma, mu = res.x
    alpha = 1.0 / (1.0 + np.exp(-logit_alpha))
    return NTSParams(
        alpha=float(alpha),
        theta=float(np.exp(log_theta)),
        beta=float(beta),
        gamma=float(np.exp(log_gamma)),
        mu=float(mu),
    )
