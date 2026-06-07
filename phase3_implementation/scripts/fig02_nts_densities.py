"""Figure 2: NTS density shapes, tail comparison, and PIT diagnostic.

Panel (a): NTS densities for three parameter settings, rescaled to unit
variance for a fair comparison against a standard Gaussian. Without this
rescaling, differences between NTS parameterizations are dominated by
differences in overall variance, not tail shape.

Panel (b): Log-scale right-tail comparison. This is where NTS diverges
visibly from the Gaussian: heavy-tailed NTS retains visible density out
to several standard deviations while the Gaussian decays to essentially
zero. This is the key property that motivates the NTS-LGC synthesis.

Panel (c): Probability integral transform diagnostic. Sample from an
NTS distribution via inverse-CDF transform, then apply the NTS CDF. A
correctly implemented CDF and simulator together yield an approximately
uniform output.

Output: figures/fig02_nts_densities.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
IMPL_ROOT = HERE.parent
sys.path.insert(0, str(IMPL_ROOT))

from ntslgc.nts import NTSParams, cdf, density_fft, sample
from ntslgc.plots import use_paper_style

FIGURE_PATH = IMPL_ROOT.parent / "figures" / "fig02_nts_densities.pdf"


def rescale_density_to_unit_variance(
    params: NTSParams,
    x_grid: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return (f_y, sigma) where f_y is the density of Y = X / sigma
    with sigma chosen so Var(Y) = 1.

    Computes Var(X) numerically from the NTS density, then uses the
    change of variables f_Y(y) = sigma * f_X(sigma * y).
    """
    dense = np.linspace(-30.0, 30.0, 6001)
    fx = density_fft(dense, params)
    mean_x = float(np.trapezoid(dense * fx, dense))
    var_x = float(np.trapezoid((dense - mean_x) ** 2 * fx, dense))
    sigma = np.sqrt(var_x)
    # f_Y(y) = sigma * f_X(sigma*y + mean_x) for Y = (X - mean_x)/sigma.
    f_y = sigma * density_fft(mean_x + sigma * x_grid, params)
    return f_y, sigma


def main() -> None:
    use_paper_style()

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))

    # -----------------------------------------------------------------
    # Common: density grids, parameter list
    # -----------------------------------------------------------------
    x_grid = np.linspace(-6.0, 6.0, 601)
    param_list = [
        (NTSParams(alpha=0.5, theta=1.0, beta=0.0, gamma=1.0, mu=0.0),
         r"NTS symmetric ($\alpha=0.5$)", "C0"),
        (NTSParams(alpha=0.5, theta=1.0, beta=-0.6, gamma=1.0, mu=0.0),
         r"NTS skew-left ($\beta=-0.6$)", "C1"),
        (NTSParams(alpha=0.3, theta=0.5, beta=0.0, gamma=1.0, mu=0.0),
         r"NTS heavier ($\alpha=0.3$)", "C2"),
    ]

    # -----------------------------------------------------------------
    # Panel (a): unit-variance densities on linear scale
    # -----------------------------------------------------------------
    for params, label, color in param_list:
        f_y, sigma = rescale_density_to_unit_variance(params, x_grid)
        axes[0].plot(x_grid, f_y, color=color, label=label, linewidth=1.5)
        print(f"  {label}: sigma = {sigma:.4f}")

    axes[0].plot(x_grid, norm.pdf(x_grid), color="black", linestyle="--",
                 label=r"$\mathcal{N}(0, 1)$", linewidth=1.0, alpha=0.7)
    axes[0].set_xlabel("$y$ (unit variance)")
    axes[0].set_ylabel("density $f_Y(y)$")
    axes[0].set_title("(a) NTS densities, unit variance")
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=8)
    axes[0].set_xlim(-5, 5)

    # -----------------------------------------------------------------
    # Panel (b): log-scale right tail (after unit-variance rescaling)
    # -----------------------------------------------------------------
    tail_grid = np.linspace(0.5, 8.0, 501)
    for params, label, color in param_list:
        f_y, sigma = rescale_density_to_unit_variance(params, tail_grid)
        axes[1].semilogy(tail_grid, np.clip(f_y, 1e-15, None),
                         color=color, label=label, linewidth=1.5)

    axes[1].semilogy(tail_grid, norm.pdf(tail_grid), color="black",
                     linestyle="--", label=r"$\mathcal{N}(0, 1)$",
                     linewidth=1.0, alpha=0.7)
    axes[1].set_xlabel("$y$ (unit variance)")
    axes[1].set_ylabel(r"density $f_Y(y)$ (log scale)")
    axes[1].set_title("(b) Right-tail comparison")
    axes[1].set_xlim(0.5, 7.0)
    axes[1].set_ylim(1e-10, 1.0)
    axes[1].legend(loc="upper right", framealpha=0.9, fontsize=8)

    # -----------------------------------------------------------------
    # Panel (c): PIT diagnostic with the inverse-CDF simulator
    # -----------------------------------------------------------------
    params_pit = NTSParams(alpha=0.4, theta=0.8, beta=-0.3, gamma=1.0, mu=0.0)
    rng = np.random.default_rng(42)
    n = 5000
    samples = sample(params_pit, n, rng)
    u = cdf(samples, params_pit)

    axes[2].hist(u, bins=30, density=True, alpha=0.7, color="C0",
                 edgecolor="black", linewidth=0.5)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.0,
                    label="Uniform reference")
    axes[2].set_xlabel("PIT value $F(X)$")
    axes[2].set_ylabel("density")
    axes[2].set_title(f"(c) PIT of $n = {n}$ NTS samples")
    axes[2].legend(loc="lower center", framealpha=0.9)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1.5)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH)
    print(f"Saved figure to {FIGURE_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
