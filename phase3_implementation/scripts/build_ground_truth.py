"""Build cached ground-truth LGC surfaces for the MC study.

Run once before run_mc.py. Output goes to cache/ground_truth.npz.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IMPL_ROOT = HERE.parent
sys.path.insert(0, str(IMPL_ROOT))

from ntslgc.copulas import (
    clayton_theta_for_tau,
    gaussian_rho_for_tau,
    gumbel_theta_for_tau,
)
from ntslgc.ground_truth import (
    GRID_AXIS,
    QUANTILE_LEVELS,
    compute_ground_truth_surface,
    make_eval_grid,
    quantile_points,
)


TARGET_TAU = 0.5
N_GT = 20000
T_DF = 4

CONFIGS = {
    "gaussian": {"rho": gaussian_rho_for_tau(TARGET_TAU)},
    "clayton":  {"theta": clayton_theta_for_tau(TARGET_TAU)},
    "t":        {"rho": gaussian_rho_for_tau(TARGET_TAU), "df": T_DF},
    "gumbel":   {"theta": gumbel_theta_for_tau(TARGET_TAU)},
}

CACHE_PATH = IMPL_ROOT / "cache" / "ground_truth.npz"


def main() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building ground truth at n_gt = {N_GT}, target Kendall's tau = {TARGET_TAU}")
    print(f"Grid: 11x11 over [-3, 3]^2 plus 5 diagonal quantile points")
    print(f"Output: {CACHE_PATH}\n")

    out = {
        "grid": make_eval_grid(),
        "grid_axis": GRID_AXIS,
        "quantile_levels": QUANTILE_LEVELS,
        "quantile_points": quantile_points(),
    }

    for name, params in CONFIGS.items():
        t0 = time.time()
        print(f"  {name:10s}  params = {params}  ", end="", flush=True)
        rho_grid, rho_quant = compute_ground_truth_surface(
            name, params, n_gt=N_GT, seed=1234 + hash(name) % 10000
        )
        dt = time.time() - t0
        out[f"rho_grid_{name}"]  = rho_grid
        out[f"rho_quant_{name}"] = rho_quant
        # Diagonal sanity print: rho at quantile points
        qstr = ", ".join(f"{r:+.3f}" for r in rho_quant)
        print(f"  done in {dt:5.1f}s   rho(q) = [{qstr}]")

    np.savez(CACHE_PATH, **out)
    print(f"\nSaved ground truth to {CACHE_PATH}")


if __name__ == "__main__":
    main()
