"""Verify the new FFT-based density against the legacy Riemann-sum version.

For several NTS parameter sets, evaluate both implementations on a fixed
set of query points and report (a) the max absolute deviation and (b) the
typical relative error in the body of the distribution. Also report the
speedup of the FFT version over the legacy version at sample sizes
typical of the MC study (n in {500, 1000, 2500, 5000}).

Two implementations of the SAME integral with the SAME u-grid should
agree to numerical precision on the x-grid that the FFT uses internally;
the only added error in the FFT-then-interp pipeline is the linear
interpolation between adjacent x-grid points, which is O(dx^2) for
smooth densities. We expect agreement to ~4-6 digits in the body.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IMPL_ROOT = HERE.parent
sys.path.insert(0, str(IMPL_ROOT))

from ntslgc.nts import (
    NTSParams,
    _density_riemann_legacy,
    density_fft,
)


PARAM_CASES = [
    ("symmetric, gamma=1",
     NTSParams(alpha=0.5, theta=1.0, beta=0.0, gamma=1.0, mu=0.0)),
    ("skew-left, heavier",
     NTSParams(alpha=0.5, theta=0.8, beta=-0.4, gamma=1.0, mu=0.0)),
    ("heavy-tailed, alpha=0.3",
     NTSParams(alpha=0.3, theta=1.0, beta=0.0, gamma=1.0, mu=0.0)),
    ("near-Gaussian, alpha=0.9",
     NTSParams(alpha=0.9, theta=2.0, beta=0.0, gamma=1.0, mu=0.0)),
]

# Query points: a dense grid for accuracy comparison.
ACCURACY_QUERY = np.linspace(-10.0, 10.0, 401)

# Larger samples for timing.
TIMING_SAMPLE_SIZES = [500, 1000, 2500, 5000]


def accuracy_check() -> bool:
    print("\n[1] Accuracy: FFT vs. legacy Riemann sum on 401 query points in [-10, 10]")
    print(f"    {'parameter set':28s}  {'max |diff|':>12s}  {'rel err (body)':>16s}  [PASS/FAIL]")
    all_ok = True
    for label, params in PARAM_CASES:
        f_fft = density_fft(ACCURACY_QUERY, params)
        f_leg = _density_riemann_legacy(ACCURACY_QUERY, params)
        diff = np.abs(f_fft - f_leg)
        max_diff = float(diff.max())
        # Relative error in the body (where density > 1e-3).
        body = f_leg > 1e-3
        if body.any():
            rel_err = float(np.mean(diff[body] / f_leg[body]))
        else:
            rel_err = float("nan")
        # Tolerance: FFT introduces O(dx^2) interp error; with dx ~ 0.015 on a
        # smooth bell-shape, interp error in the body should be <~1e-3.
        ok = max_diff < 5e-3 and (np.isnan(rel_err) or rel_err < 1e-2)
        flag = "PASS" if ok else "FAIL"
        print(f"    {label:28s}  {max_diff:12.2e}  {rel_err:16.2e}  [{flag}]")
        all_ok = all_ok and ok
    return all_ok


def timing_check() -> None:
    print("\n[2] Speedup of FFT over legacy at typical MC sample sizes")
    print(f"    {'n':>6s}  {'legacy (ms)':>12s}  {'fft (ms)':>10s}  {'speedup':>10s}")
    params = PARAM_CASES[0][1]
    rng = np.random.default_rng(0)
    for n in TIMING_SAMPLE_SIZES:
        # Use a sample-like query: noise around 0, similar to fit_mle's calls.
        x = rng.standard_normal(n) * 1.5
        # warmup
        _ = density_fft(x, params)
        _ = _density_riemann_legacy(x[:50], params)

        t0 = time.perf_counter()
        for _ in range(20):
            _ = _density_riemann_legacy(x, params)
        t_leg = (time.perf_counter() - t0) / 20 * 1000.0

        t0 = time.perf_counter()
        for _ in range(20):
            _ = density_fft(x, params)
        t_fft = (time.perf_counter() - t0) / 20 * 1000.0

        speedup = t_leg / t_fft if t_fft > 0 else float("inf")
        print(f"    {n:6d}  {t_leg:12.2f}  {t_fft:10.2f}  {speedup:9.1f}x")


def main() -> None:
    ok = accuracy_check()
    timing_check()
    print()
    if ok:
        print("Density FFT verified against legacy. Proceed to MC.")
        sys.exit(0)
    else:
        print("Density FFT FAILED accuracy check. Investigate before MC.")
        sys.exit(1)


if __name__ == "__main__":
    main()
