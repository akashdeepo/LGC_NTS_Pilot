"""Analyze MC results and produce figures + a small summary table.

Reads:
  cache/ground_truth.npz   (from build_ground_truth.py)
  results/mc_raw.npz       (from run_mc.py)

Writes:
  figures/fig03_mc_ise.pdf
  figures/fig04_mc_pointwise.pdf
  results/mc_summary.csv

ISE per replication is computed as the sum of squared deviations between
the estimated and ground-truth LGC over the 11x11 grid, weighted by the
joint standard-normal density at each grid point. Density weighting is
the conventional choice in the LGC literature: it focuses error on
regions where data actually lives, rather than weighting empty tails
equally with the dense center.

Pointwise bias and variance are reported at each of the 5 diagonal
quantile points q in {0.01, 0.05, 0.5, 0.95, 0.99}.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
IMPL_ROOT = HERE.parent
sys.path.insert(0, str(IMPL_ROOT))

from ntslgc.ground_truth import GRID_AXIS, QUANTILE_LEVELS
from ntslgc.plots import use_paper_style


COPULAS    = ["gaussian", "clayton", "t", "gumbel"]
COPULA_NICE = {
    "gaussian": "Gaussian",
    "clayton":  "Clayton",
    "t":        "t (df=4)",
    "gumbel":   "Gumbel",
}
SAMPLE_SIZES = [500, 1000, 2500, 5000]
ESTIMATORS = ["canonical", "nts_oracle", "nts_fitted"]
ESTIMATOR_NICE = {
    "canonical":  "Canonical",
    "nts_oracle": "NTS-LGC (oracle)",
    "nts_fitted": "NTS-LGC (fitted)",
}
ESTIMATOR_COLOR = {
    "canonical":  "C0",
    "nts_oracle": "C2",
    "nts_fitted": "C3",
}
ESTIMATOR_MARKER = {
    "canonical":  "o",
    "nts_oracle": "s",
    "nts_fitted": "D",
}


def density_weights(grid_axis: np.ndarray) -> np.ndarray:
    """Joint standard-normal density weights on the 11x11 grid, normalized
    to sum to 1 (so the weighted-ISE has units of mean squared error
    relative to a probability average).
    """
    gx, gy = np.meshgrid(grid_axis, grid_axis)
    w = norm.pdf(gx) * norm.pdf(gy)
    w = w.ravel()
    return w / w.sum()


def main() -> None:
    use_paper_style()

    gt   = np.load(IMPL_ROOT / "cache"   / "ground_truth.npz")
    mc   = np.load(IMPL_ROOT / "results" / "mc_raw.npz")

    w = density_weights(GRID_AXIS)

    # --- ISE per (copula, n, estimator) ---------------------------------
    # Mean-and-SD of the density-weighted ISE across replications.
    ise_mean = {est: np.zeros((len(COPULAS), len(SAMPLE_SIZES))) for est in ESTIMATORS}
    ise_sd   = {est: np.zeros((len(COPULAS), len(SAMPLE_SIZES))) for est in ESTIMATORS}

    for i, copula in enumerate(COPULAS):
        rho_true_grid = gt[f"rho_grid_{copula}"]                  # (121,)
        for j, n in enumerate(SAMPLE_SIZES):
            for est in ESTIMATORS:
                arr = mc[f"rho_grid_{copula}_{n}_{est}"]          # (R, 121)
                err = arr - rho_true_grid[None, :]
                ise_per_rep = np.sum(w[None, :] * err * err, axis=1)
                ise_mean[est][i, j] = float(np.mean(ise_per_rep))
                ise_sd[est][i, j]   = float(np.std(ise_per_rep, ddof=1))

    # --- Pointwise bias/var at quantile points --------------------------
    # arrays of shape (n_copulas, n_sample_sizes, n_quantiles)
    pt_bias = {est: np.zeros((len(COPULAS), len(SAMPLE_SIZES), 5)) for est in ESTIMATORS}
    pt_var  = {est: np.zeros((len(COPULAS), len(SAMPLE_SIZES), 5)) for est in ESTIMATORS}

    for i, copula in enumerate(COPULAS):
        rho_true_q = gt[f"rho_quant_{copula}"]                    # (5,)
        for j, n in enumerate(SAMPLE_SIZES):
            for est in ESTIMATORS:
                arr = mc[f"rho_quant_{copula}_{n}_{est}"]         # (R, 5)
                pt_bias[est][i, j] = np.mean(arr - rho_true_q[None, :], axis=0)
                pt_var[est][i, j]  = np.var(arr, axis=0, ddof=1)

    # --- Figure 3: density-weighted ISE vs n, by copula -----------------
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for i, copula in enumerate(COPULAS):
        ax = axes[i]
        for est in ESTIMATORS:
            ax.plot(
                SAMPLE_SIZES, ise_mean[est][i, :],
                marker=ESTIMATOR_MARKER[est], color=ESTIMATOR_COLOR[est],
                label=ESTIMATOR_NICE[est], linestyle="-",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(SAMPLE_SIZES)
        ax.set_xticklabels([str(n) for n in SAMPLE_SIZES])
        ax.set_xlabel("$n$")
        if i == 0:
            ax.set_ylabel("Density-weighted ISE")
        ax.set_title(COPULA_NICE[copula])
    axes[0].legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig_path = IMPL_ROOT.parent / "figures" / "fig03_mc_ise.pdf"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved {fig_path}")

    # --- Figure 4: pointwise RMSE at quantile points (largest n) --------
    # Use the largest sample size so the picture is cleanest; rows = copulas,
    # x-axis = quantile, y-axis = RMSE. Per-panel y-axis (sharey=False) so
    # that the Clayton q=0.99 catastrophe at RMSE ~ 0.7 does not compress
    # the other three panels into a flat line.
    n_show = SAMPLE_SIZES[-1]
    j_show = SAMPLE_SIZES.index(n_show)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=False)
    q_labels = [f"{q:.2f}" for q in QUANTILE_LEVELS]
    for i, copula in enumerate(COPULAS):
        ax = axes[i]
        for est in ESTIMATORS:
            rmse = np.sqrt(pt_bias[est][i, j_show] ** 2 + pt_var[est][i, j_show])
            ax.plot(
                range(5), rmse,
                marker=ESTIMATOR_MARKER[est], color=ESTIMATOR_COLOR[est],
                label=ESTIMATOR_NICE[est], linestyle="-",
            )
        ax.set_xticks(range(5))
        ax.set_xticklabels(q_labels)
        ax.set_xlabel("Quantile $q$")
        if i == 0:
            ax.set_ylabel(rf"Pointwise RMSE at $n = {n_show}$")
        ax.set_title(COPULA_NICE[copula])
    axes[0].legend(loc="upper center", framealpha=0.9)
    fig.tight_layout()
    fig_path = IMPL_ROOT.parent / "figures" / "fig04_mc_pointwise.pdf"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Saved {fig_path}")

    # --- Summary CSV ----------------------------------------------------
    csv_path = IMPL_ROOT / "results" / "mc_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "copula", "n", "estimator",
            "ise_mean", "ise_sd",
            *[f"bias_q{q:.2f}" for q in QUANTILE_LEVELS],
            *[f"var_q{q:.2f}"  for q in QUANTILE_LEVELS],
        ])
        for i, copula in enumerate(COPULAS):
            for j, n in enumerate(SAMPLE_SIZES):
                for est in ESTIMATORS:
                    row = [
                        copula, n, est,
                        ise_mean[est][i, j], ise_sd[est][i, j],
                        *pt_bias[est][i, j].tolist(),
                        *pt_var[est][i, j].tolist(),
                    ]
                    writer.writerow(row)
    print(f"Saved {csv_path}")

    # Console summary so we can eyeball the headline result.
    print("\nDensity-weighted ISE (mean across reps), at largest sample size:")
    print(f"{'copula':10s}  " + "  ".join(f"{ESTIMATOR_NICE[e]:18s}" for e in ESTIMATORS))
    for i, copula in enumerate(COPULAS):
        cells = [f"{ise_mean[est][i, j_show]:.5f}" for est in ESTIMATORS]
        print(f"{COPULA_NICE[copula]:10s}  " + "  ".join(f"{c:18s}" for c in cells))


if __name__ == "__main__":
    main()
