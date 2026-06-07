"""Figure 1: LGC on Gaussian data recovers Pearson correlation.

Produces a two-panel figure for the paper demonstrating that the Python
LGC implementation correctly recovers the theoretical Gaussian property:
LGC equals the Pearson correlation everywhere on the support.

Panel (a): scatter of bivariate Gaussian samples with true rho = 0.6
overlaid with the estimated LGC surface as colored contours.

Panel (b): estimated LGC averaged over multiple seeds as a function of
the true rho, with +/- 1 standard deviation error bars, compared to the
y = x identity line.

Output: figures/fig01_lgc_gaussian_sanity.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
IMPL_ROOT = HERE.parent
sys.path.insert(0, str(IMPL_ROOT))

from ntslgc.lgc import lgc_on_grid, plugin_bandwidth
from ntslgc.plots import (
    plot_bias_vs_rho,
    plot_scatter_with_lgc_surface,
    use_paper_style,
)

FIGURE_PATH = IMPL_ROOT.parent / "figures" / "fig01_lgc_gaussian_sanity.pdf"


def simulate_bivariate_gaussian(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    cov = np.array([[1.0, rho], [rho, 1.0]])
    return rng.multivariate_normal(np.zeros(2), cov, size=n)


def panel_a_data(n: int = 8000, true_rho: float = 0.6, seed: int = 101):
    """Generate scatter + LGC surface for panel (a).

    Returns the effective sample size per grid point alongside the LGC
    surface so that downstream plotting can mask out regions where the
    estimator is unreliable due to sparse data.
    """
    rng = np.random.default_rng(seed)
    X = simulate_bivariate_gaussian(n, true_rho, rng)

    axis_lo, axis_hi = -3.0, 3.0
    axis_vals = np.linspace(axis_lo, axis_hi, 41)
    gx, gy = np.meshgrid(axis_vals, axis_vals)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])

    print(f"Panel (a): computing LGC on {grid_points.shape[0]} grid points "
          f"(n={n}, true rho={true_rho})")
    rhos, eff_n = lgc_on_grid(X, grid_points, return_effective_n=True)
    rho_grid = rhos.reshape(gx.shape)
    eff_n_grid = eff_n.reshape(gx.shape)

    return X, axis_vals, axis_vals, rho_grid, eff_n_grid, true_rho


def panel_b_data(
    n: int = 2000,
    true_rhos: tuple[float, ...] = (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9),
    n_seeds: int = 10,
):
    """Generate mean +/- sd LGC as a function of true rho for panel (b)."""
    print(f"Panel (b): computing LGC across {len(true_rhos)} true rho values "
          f"x {n_seeds} seeds (n={n})")
    means = []
    sds = []
    for rho in true_rhos:
        axis_estimates = []
        for seed in range(200, 200 + n_seeds):
            rng = np.random.default_rng(seed)
            X = simulate_bivariate_gaussian(n, rho, rng)
            # Evaluate along a small high-density grid.
            if rho >= 0:
                direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
            else:
                direction = np.array([1.0, -1.0]) / np.sqrt(2.0)
            t = np.linspace(-1.2, 1.2, 11)
            grid = np.outer(t, direction)
            rhos_here = lgc_on_grid(X, grid)
            axis_estimates.append(float(np.mean(rhos_here)))
        means.append(float(np.mean(axis_estimates)))
        sds.append(float(np.std(axis_estimates, ddof=1)))
        print(f"  rho = {rho:+.2f}: mean = {means[-1]:+.4f}, sd = {sds[-1]:.4f}")
    return list(true_rhos), means, sds


def main() -> None:
    use_paper_style()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))

    # Panel (a): surface with sparse-region mask.
    X, gx, gy, rho_grid, eff_n_grid, true_rho = panel_a_data()
    im = plot_scatter_with_lgc_surface(
        axes[0], X, gx, gy, rho_grid,
        title=f"(a) LGC surface, true $\\rho = {true_rho}$",
        vmin=-1.0, vmax=1.0,
        eff_n_grid=eff_n_grid,
        eff_n_threshold=40.0,
    )
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    cbar = fig.colorbar(im, ax=axes[0], shrink=0.85, pad=0.02,
                        ticks=np.linspace(-1.0, 1.0, 5))
    cbar.set_label(r"$\hat{\rho}(x_1, x_2)$")

    # Panel (b)
    true_rhos, means, sds = panel_b_data()
    plot_bias_vs_rho(
        axes[1], true_rhos, means, sds,
        label="LGC mean on principal axis",
        color="C0",
    )
    axes[1].set_title("(b) LGC consistency on Gaussian data")
    axes[1].legend(loc="upper left", framealpha=0.9)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH)
    print(f"Saved figure to {FIGURE_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
