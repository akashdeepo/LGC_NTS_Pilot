"""Sanity check before running the full Monte Carlo:

1. Each copula sampler reproduces its theoretical Kendall's tau
   to within a small Monte Carlo tolerance.

2. All four pipelines (raw, canonical, NTS-oracle, NTS-fitted) run
   end-to-end on a t-copula sample with NTS marginals and return
   finite local correlation estimates at a handful of grid points.

If either of these fails, the MC study below it is meaningless, so this
is the gating check before scaling up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
IMPL_ROOT = HERE.parent
sys.path.insert(0, str(IMPL_ROOT))

from ntslgc.copulas import (
    apply_nts_marginals,
    clayton_theta_for_tau,
    gaussian_rho_for_tau,
    gumbel_theta_for_tau,
    sample_clayton_copula,
    sample_gaussian_copula,
    sample_gumbel_copula,
    sample_t_copula,
)
from ntslgc.nts import NTSParams
from ntslgc.pipelines import (
    canonical_lgc,
    nts_lgc_fitted,
    nts_lgc_oracle,
    raw_lgc,
)


TARGET_TAU = 0.5
TAU_TOL = 0.03  # MC tolerance on Kendall's tau at n=5000


def kendall_tau_check() -> bool:
    """Each copula at theoretical tau = 0.5 should yield sample tau ~= 0.5."""
    rng = np.random.default_rng(42)
    n = 5000

    configs = [
        ("Gaussian", sample_gaussian_copula(n, gaussian_rho_for_tau(TARGET_TAU), rng)),
        ("Clayton",  sample_clayton_copula(n, clayton_theta_for_tau(TARGET_TAU), rng)),
        ("t (df=4)", sample_t_copula(n, gaussian_rho_for_tau(TARGET_TAU), 4, rng)),
        ("Gumbel",   sample_gumbel_copula(n, gumbel_theta_for_tau(TARGET_TAU), rng)),
    ]

    print(f"\n[1] Kendall's tau check (target = {TARGET_TAU:.2f}, tolerance = {TAU_TOL:.2f})")
    all_ok = True
    for label, U in configs:
        tau, _ = kendalltau(U[:, 0], U[:, 1])
        ok = abs(tau - TARGET_TAU) < TAU_TOL
        flag = "PASS" if ok else "FAIL"
        print(f"    {label:10s}  tau = {tau:+.4f}  [{flag}]")
        all_ok = all_ok and ok
    return all_ok


def pipeline_smoke_test() -> bool:
    """All four pipelines should run on a t-copula + NTS marginal sample
    and return finite estimates."""
    rng = np.random.default_rng(7)
    n = 1500

    # Sample a t-copula and push through symmetric, unit-variance NTS marginals.
    U = sample_t_copula(n, gaussian_rho_for_tau(TARGET_TAU), df=4, rng=rng)
    params = NTSParams(alpha=0.5, theta=1.0, beta=0.0, gamma=1.0, mu=0.0)
    X = apply_nts_marginals(U, params, params)

    # Grid points (in Z-space for the three transformed pipelines; the
    # raw pipeline uses the same coordinates in original X-space, which
    # is fine for a smoke test since the marginals are mean-zero).
    grid = np.array([
        [ 0.0,  0.0],
        [ 1.0,  1.0],
        [-1.0, -1.0],
        [ 1.5, -1.5],
    ])

    print(f"\n[2] Pipeline smoke test (t-copula + symmetric NTS marginals, n={n})")

    results = {}
    try:
        results["raw_lgc"]        = raw_lgc(X, grid)
        results["canonical_lgc"]  = canonical_lgc(X, grid)
        results["nts_lgc_oracle"] = nts_lgc_oracle(X, grid, params, params)
        results["nts_lgc_fitted"] = nts_lgc_fitted(X, grid)
    except Exception as e:
        print(f"    EXCEPTION: {e}")
        return False

    all_ok = True
    for name, rhos in results.items():
        finite = np.all(np.isfinite(rhos))
        in_range = np.all(np.abs(rhos) < 1.0)
        ok = finite and in_range
        flag = "PASS" if ok else "FAIL"
        rho_str = ", ".join(f"{r:+.3f}" for r in rhos)
        print(f"    {name:18s}  rho = [{rho_str}]  [{flag}]")
        all_ok = all_ok and ok
    return all_ok


def main() -> None:
    ok1 = kendall_tau_check()
    ok2 = pipeline_smoke_test()
    print()
    if ok1 and ok2:
        print("All sanity checks passed. The MC foundations are sound.")
        sys.exit(0)
    else:
        print("One or more sanity checks FAILED. Investigate before scaling up.")
        sys.exit(1)


if __name__ == "__main__":
    main()
