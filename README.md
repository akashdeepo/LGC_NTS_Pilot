# NTS-LGC Pilot

A Python implementation of Normal Tempered Stable (NTS) marginals
combined with bivariate Local Gaussian Correlation (LGC), together with
a Monte Carlo harness comparing three LGC estimation pipelines
(canonical, NTS-oracle, NTS-fitted) on four copulas paired with NTS
marginals.

## Layout

```
.
├── README.md
├── requirements.txt
├── figures/                          # generated figure PDFs
└── phase3_implementation/
    ├── ntslgc/                       # the package
    │   ├── nts.py                    # NTS distribution + Carr-Madan FFT density
    │   ├── lgc.py                    # bivariate Tjostheim-Hufthammer LGC
    │   ├── copulas.py                # Gaussian / Clayton / t / Gumbel samplers
    │   ├── pipelines.py              # canonical / oracle / fitted estimators
    │   ├── ground_truth.py           # large-N reference LGC surfaces
    │   └── plots.py                  # matplotlib helpers
    ├── scripts/
    │   ├── sanity_check_mc_foundations.py  # Kendall-tau + pipeline smoke test
    │   ├── verify_density_fft.py     # FFT density vs Riemann legacy
    │   ├── build_ground_truth.py     # writes cache/ground_truth.npz
    │   ├── run_mc.py                 # the MC harness; writes results/mc_raw.npz
    │   ├── analyze_mc.py             # produces fig03, fig04, summary CSV
    │   ├── fig01_lgc_gaussian_sanity.py
    │   └── fig02_nts_densities.py
    ├── tests/                        # unit tests for NTS and LGC
    ├── cache/                        # cached ground-truth surfaces
    └── results/                      # MC raw outputs and summary CSV
```

## Install

Requires Python 3.11+. Three dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing the Monte Carlo

```bash
cd phase3_implementation

# Sanity checks (~1 min each)
python scripts/sanity_check_mc_foundations.py
python scripts/verify_density_fft.py

# Ground-truth LGC surfaces per copula (~1 min)
python scripts/build_ground_truth.py

# Full MC: 4 copulas x 4 sample sizes x 50 reps x 3 estimators (~30 min).
# Parallelized across cores; checkpoints every 25 tasks so the run resumes
# cleanly after any interruption (re-issue the same command).
python scripts/run_mc.py

# Figures and summary CSV
python scripts/analyze_mc.py
```

Outputs land in `phase3_implementation/cache/`,
`phase3_implementation/results/`, and `figures/`.

## Notes on the implementation

- **NTS density** uses Carr--Madan FFT inversion of the characteristic
  function (`density_fft` in `nts.py`). A reference Riemann-sum
  implementation is retained as `_density_riemann_legacy` for direct
  comparison; `scripts/verify_density_fft.py` checks the two agree.
- **Per-worker BLAS threading is capped at 1** in `run_mc.py` so that
  multiprocessing across cores does not oversubscribe via internal
  numpy thread pools.
- **MC reproducibility:** each task's seed is derived deterministically
  from `(copula, n, rep_id)`, so re-running the harness reproduces the
  same per-rep outputs.
