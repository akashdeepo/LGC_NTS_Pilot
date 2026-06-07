"""Sanity tests for the NTS module.

Checks:

1. Density from FFT inversion integrates to approximately 1.
2. Density is positive everywhere.
3. Characteristic function satisfies phi(0) = 1 and |phi(u)| <= 1.
4. CDF is monotone increasing in x.
5. CDF approaches 0 and 1 at extreme x.
6. Round-trip: CDF(x) applied to sample data yields approximately U(0,1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ntslgc.nts import (
    NTSParams,
    cdf,
    char_function,
    density_fft,
    fit_mle,
    sample,
)


def test_char_function_basic(params: NTSParams) -> bool:
    """phi(0) = 1 and |phi(u)| <= 1 for all real u."""
    u = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, -1.0, -5.0])
    phi = char_function(u, params)
    phi_zero_ok = abs(phi[0] - 1.0) < 1e-10
    modulus_ok = np.all(np.abs(phi) <= 1.0 + 1e-10)
    print(f"  phi(0) = {phi[0]:.6f}  {'OK' if phi_zero_ok else 'FAIL'}")
    print(f"  max |phi(u)| = {np.max(np.abs(phi)):.6f}  {'OK' if modulus_ok else 'FAIL'}")
    return phi_zero_ok and modulus_ok


def test_density_integrates_to_one(params: NTSParams) -> bool:
    """Density integrates to ~1 on a wide support."""
    x = np.linspace(-20.0, 20.0, 4001)
    dens = density_fft(x, params)
    integral = float(np.trapezoid(dens, x))
    ok = abs(integral - 1.0) < 0.02
    print(f"  integral f(x) dx = {integral:.6f}  {'OK' if ok else 'FAIL'}")
    return ok


def test_density_nonneg(params: NTSParams) -> bool:
    """Density should be nonnegative everywhere."""
    x = np.linspace(-15.0, 15.0, 301)
    dens = density_fft(x, params)
    ok = np.all(dens >= 0.0)
    print(f"  min density = {np.min(dens):.6e}  {'OK' if ok else 'FAIL'}")
    return ok


def test_cdf_monotone(params: NTSParams) -> bool:
    """CDF is monotone increasing."""
    x = np.linspace(-10.0, 10.0, 101)
    F = cdf(x, params)
    diffs = np.diff(F)
    ok = np.all(diffs >= -1e-6)  # allow tiny numerical wobble
    min_diff = float(np.min(diffs))
    print(f"  min increment = {min_diff:.6e}  {'OK' if ok else 'FAIL'}")
    return ok


def test_cdf_limits(params: NTSParams) -> bool:
    """CDF -> 0 as x -> -inf, CDF -> 1 as x -> inf."""
    F_lo = cdf(np.array([-15.0]), params)[0]
    F_hi = cdf(np.array([15.0]), params)[0]
    print(f"  F(-15) = {F_lo:.6f}, F(+15) = {F_hi:.6f}")
    ok = F_lo < 0.02 and F_hi > 0.98
    return ok


def test_pit_uniform(params: NTSParams, n: int = 2000, seed: int = 0) -> bool:
    """Probability integral transform of NTS samples yields approximately U(0,1)."""
    rng = np.random.default_rng(seed)
    x = sample(params, n, rng)
    u = cdf(x, params)
    # Test uniform: sample mean ~ 0.5, sample variance ~ 1/12.
    mean_u = float(np.mean(u))
    var_u = float(np.var(u))
    mean_ok = abs(mean_u - 0.5) < 0.05
    var_ok = abs(var_u - 1.0 / 12.0) < 0.02
    print(f"  PIT sample mean = {mean_u:.4f} (target 0.5)  {'OK' if mean_ok else 'FAIL'}")
    print(f"  PIT sample var  = {var_u:.4f} (target {1/12:.4f})  {'OK' if var_ok else 'FAIL'}")
    return mean_ok and var_ok


def run_all_for_params(params: NTSParams, label: str) -> bool:
    print(f"\n=== {label} ===")
    print(f"  params: alpha={params.alpha}, theta={params.theta}, "
          f"beta={params.beta}, gamma={params.gamma}, mu={params.mu}")
    r1 = test_char_function_basic(params)
    r2 = test_density_integrates_to_one(params)
    r3 = test_density_nonneg(params)
    r4 = test_cdf_monotone(params)
    r5 = test_cdf_limits(params)
    r6 = test_pit_uniform(params)
    all_ok = r1 and r2 and r3 and r4 and r5 and r6
    print(f"  OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    # A symmetric mildly heavy-tailed case:
    p1 = NTSParams(alpha=0.5, theta=1.0, beta=0.0, gamma=1.0, mu=0.0)

    # An asymmetric heavier-tailed case:
    p2 = NTSParams(alpha=0.3, theta=0.5, beta=-0.4, gamma=0.9, mu=0.0)

    ok1 = run_all_for_params(p1, "Case 1: symmetric, alpha=0.5")
    ok2 = run_all_for_params(p2, "Case 2: skewed, alpha=0.3")

    all_pass = ok1 and ok2
    print()
    print("=" * 50)
    print("OVERALL:", "PASS" if all_pass else "FAIL")
    sys.exit(0 if all_pass else 1)
