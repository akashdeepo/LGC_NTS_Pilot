"""Plotting utilities for the NTS-LGC pilot study.

Thin wrappers around matplotlib for the figures that go into both the
diagnostic notebooks and the paper. Style is kept minimal so that figures
are legible in both black-and-white and color.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

# --- Global style --------------------------------------------------------
PAPER_STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}


def use_paper_style() -> None:
    """Apply paper-quality matplotlib defaults."""
    plt.rcParams.update(PAPER_STYLE)


def plot_scatter_with_lgc_surface(
    ax: plt.Axes,
    X: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    rho_grid: np.ndarray,
    title: str = "",
    sample_alpha: float = 0.25,
    vmin: float = -1.0,
    vmax: float = 1.0,
    eff_n_grid: np.ndarray | None = None,
    eff_n_threshold: float = 20.0,
) -> "matplotlib.image.AxesImage":
    """Scatter of samples with a smooth image (pcolormesh) of the LGC surface.

    Uses pcolormesh with a fixed colormap range [vmin, vmax] so the
    colorbar reflects the theoretical range of the LGC, not the data range
    within the plot.

    Parameters
    ----------
    eff_n_grid : 2D array, optional
        Effective sample size at each grid point, same shape as rho_grid.
        When supplied, cells with eff_n below eff_n_threshold are masked
        out of the surface to indicate regions where LGC cannot be
        estimated reliably. This is the honest way to show the LGC surface
        --- nonparametric estimators have a domain where they are
        meaningful and we should not paint over sparse regions.
    eff_n_threshold : float
        Minimum effective sample size to display the LGC estimate.
    """
    rho_plot = np.array(rho_grid, copy=True)
    if eff_n_grid is not None:
        mask = eff_n_grid < eff_n_threshold
        rho_plot = np.ma.array(rho_plot, mask=mask)

    levels_for_contour = np.linspace(-1.0, 1.0, 21)
    im = ax.pcolormesh(
        grid_x, grid_y, rho_plot,
        cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto",
    )
    # Only contour where the underlying surface is present.
    if eff_n_grid is None:
        ax.contour(
            grid_x, grid_y, rho_grid,
            levels=levels_for_contour, colors="black",
            linewidths=0.3, alpha=0.4,
        )
    ax.scatter(X[:, 0], X[:, 1], s=4, alpha=sample_alpha, color="black",
               rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal", adjustable="box")
    return im


# Keep the old name as an alias to not break existing scripts mid-refactor.
plot_scatter_with_lgc_contour = plot_scatter_with_lgc_surface


def plot_bias_vs_rho(
    ax: plt.Axes,
    true_rhos: Sequence[float],
    mean_estimates: Sequence[float],
    sd_estimates: Sequence[float],
    label: str = "LGC",
    color: str = "C0",
    marker: str = "o",
) -> None:
    """Plot estimated LGC mean and +/- 1 sd versus true rho."""
    true_rhos = np.asarray(true_rhos)
    mean_estimates = np.asarray(mean_estimates)
    sd_estimates = np.asarray(sd_estimates)
    ax.errorbar(
        true_rhos, mean_estimates, yerr=sd_estimates,
        label=label, color=color, marker=marker,
        linestyle="-", capsize=3, markersize=5,
    )
    # Reference line y = x.
    lo = min(true_rhos.min(), mean_estimates.min()) - 0.1
    hi = max(true_rhos.max(), mean_estimates.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=0.8, label=None)
    ax.set_xlabel(r"True $\rho$")
    ax.set_ylabel(r"Estimated $\rho$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
