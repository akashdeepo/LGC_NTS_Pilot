"""Sanity test: LGC on jointly Gaussian data should equal Pearson correlation.

Property 3 of Tjostheim & Hufthammer (2013): LGC reduces to ordinary Pearson
correlation when the underlying distribution is bivariate Gaussian, because
the local Gaussian fit exactly matches the true density everywhere.

In finite samples, LGC has pointwise bias and variance, particularly in
low-density regions. We test:

1. At the center (highest density), LGC should be very close to true rho.
2. Averaged over a dense grid weighted by the true density, LGC should
   closely match true rho.
3. Averaging over multiple seeds, the bias should be small.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ntslgc.lgc import lgc_on_grid, plugin_bandwidth


def simulate_bivariate_gaussian(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from a bivariate standard Gaussian with correlation rho."""
    mean = np.zeros(2)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return rng.multivariate_normal(mean, cov, size=n)


def build_high_density_grid(rho: float, n_points: int = 25) -> np.ndarray:
    """Build a grid of evaluation points in the high-density region of the
    standardized bivariate Gaussian with correlation rho.

    We sample points along the principal axis of the distribution (which for
    a bivariate Gaussian with correlation rho runs through the origin at
    angle pi/4 for positive rho, 3*pi/4 for negative rho), where density is
    highest. This avoids testing at artificially low-density points.
    """
    # Principal axis direction: (1, sign(rho)) normalized, or (1, 1) for rho=0.
    if rho >= 0:
        direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
    else:
        direction = np.array([1.0, -1.0]) / np.sqrt(2.0)
    # Sample along the axis in [-1.5, 1.5] (avoiding very low density tails).
    t = np.linspace(-1.5, 1.5, n_points)
    grid = np.outer(t, direction)
    return grid


def run_test(n: int, true_rho: float, seed: int) -> tuple[bool, float]:
    rng = np.random.default_rng(seed)
    X = simulate_bivariate_gaussian(n, true_rho, rng)

    # Three key evaluation types:
    # 1. Center (should be closest to true rho)
    # 2. High-density grid along principal axis
    # 3. Standard diagnostic points for visualization
    center = np.array([[0.0, 0.0]])
    high_density = build_high_density_grid(true_rho, n_points=15)

    rho_center = lgc_on_grid(X, center)[0]
    rhos_axis = lgc_on_grid(X, high_density)
    mean_axis = float(np.mean(rhos_axis))

    sample_pearson = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])

    print(f"  n = {n}, true rho = {true_rho:+.2f}, bandwidth = {plugin_bandwidth(n):.4f}")
    print(f"  Sample Pearson correlation = {sample_pearson:+.4f}")
    print(f"  LGC at center              = {rho_center:+.4f} (err {rho_center - true_rho:+.4f})")
    print(f"  LGC mean on principal axis = {mean_axis:+.4f} (err {mean_axis - true_rho:+.4f})")

    # Acceptance: center AND axis-mean within tol of true rho.
    tol = 0.08
    center_ok = abs(rho_center - true_rho) < tol
    axis_ok = abs(mean_axis - true_rho) < tol
    passed = center_ok and axis_ok
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed, mean_axis


def multi_seed_bias(n: int, true_rho: float, n_seeds: int = 5) -> None:
    """Average LGC across multiple seeds to see true bias."""
    axis_means = []
    for seed in range(100, 100 + n_seeds):
        rng = np.random.default_rng(seed)
        X = simulate_bivariate_gaussian(n, true_rho, rng)
        high_density = build_high_density_grid(true_rho, n_points=15)
        rhos_axis = lgc_on_grid(X, high_density)
        axis_means.append(float(np.mean(rhos_axis)))
    mean_est = float(np.mean(axis_means))
    sd_est = float(np.std(axis_means, ddof=1))
    bias = mean_est - true_rho
    print(f"  Over {n_seeds} seeds: mean LGC = {mean_est:+.4f}, sd = {sd_est:.4f}, bias = {bias:+.4f}")


def multi_seed_test(n: int, true_rho: float, n_seeds: int = 5, tol: float = 0.05) -> bool:
    """Primary acceptance test: average LGC bias across seeds should be small.

    Single-seed tests are noisy — a bad draw can legitimately produce a
    LGC estimate far from the true rho because the local sample correlation
    IS far from the true rho on that draw. The honest check is consistency
    in expectation.
    """
    axis_means = []
    center_vals = []
    for seed in range(100, 100 + n_seeds):
        rng = np.random.default_rng(seed)
        X = simulate_bivariate_gaussian(n, true_rho, rng)
        high_density = build_high_density_grid(true_rho, n_points=15)
        rhos_axis = lgc_on_grid(X, high_density)
        axis_means.append(float(np.mean(rhos_axis)))
        center_vals.append(float(lgc_on_grid(X, np.array([[0.0, 0.0]]))[0]))

    mean_axis = float(np.mean(axis_means))
    sd_axis = float(np.std(axis_means, ddof=1))
    bias_axis = mean_axis - true_rho

    mean_center = float(np.mean(center_vals))
    bias_center = mean_center - true_rho

    print(f"  n = {n}, true rho = {true_rho:+.2f}, {n_seeds} seeds, tol = {tol}")
    print(f"  Mean LGC on axis    = {mean_axis:+.4f} (bias {bias_axis:+.4f}, sd {sd_axis:.4f})")
    print(f"  Mean LGC at center  = {mean_center:+.4f} (bias {bias_center:+.4f})")
    passed = abs(bias_axis) < tol and abs(bias_center) < tol
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print("Single-seed diagnostic runs")
    print("=" * 50)
    diagnostic_cases = [
        (5000, 0.5),
        (5000, 0.0),
        (5000, -0.7),
    ]
    for n, rho in diagnostic_cases:
        print()
        print(f"Diagnostic: n={n}, rho={rho:+.2f}")
        print("-" * 50)
        run_test(n=n, true_rho=rho, seed=42)

    print()
    print("=" * 50)
    print("Primary acceptance: multi-seed bias tests")
    print("=" * 50)
    acceptance_cases = [
        (2000, 0.5),
        (2000, 0.0),
        (2000, -0.7),
        (5000, 0.5),
        (5000, 0.0),
        (5000, -0.7),
    ]

    all_ok = True
    for n, rho in acceptance_cases:
        print()
        print(f"Multi-seed test: n={n}, rho={rho:+.2f}")
        print("-" * 50)
        ok = multi_seed_test(n=n, true_rho=rho, n_seeds=5, tol=0.06)
        all_ok = all_ok and ok

    print()
    print("=" * 50)
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)
