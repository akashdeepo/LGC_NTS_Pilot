"""Main Monte Carlo harness for the NTS-LGC pilot.

For each (copula, sample size, replication) we draw a fresh sample with
NTS marginals and a known copula, run three LGC estimators on it
(canonical, NTS-oracle, NTS-fitted), and store the resulting local
correlation estimates at the standard grid plus the five diagonal
quantile points.

Parallelizes across (copula, sample size, replication) triples using
multiprocessing.

Output: results/mc_raw.npz, a single npz with arrays keyed by
    rho_grid_<copula>_<n>_<estimator>   shape (R, 121)
    rho_quant_<copula>_<n>_<estimator>  shape (R, 5)
"""

from __future__ import annotations

import os
# Cap BLAS / OpenMP threads at 1 per worker BEFORE importing numpy.
# Without this, each multiprocessing worker spawns its own BLAS thread pool,
# producing massive oversubscription (e.g. 11 workers x ~11 BLAS threads on 12
# cores). With these caps in place, the multiprocessing.Pool gets clean
# per-core parallelism instead of contending with itself.
for _var in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import sys
import time
from pathlib import Path

import numpy as np

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
from ntslgc.ground_truth import make_eval_grid, quantile_points
from ntslgc.nts import NTSParams
from ntslgc.pipelines import canonical_lgc, nts_lgc_fitted, nts_lgc_oracle


# --- Experiment configuration -------------------------------------------

TARGET_TAU = 0.5
T_DF = 4
NTS_MARGINAL = NTSParams(alpha=0.5, theta=1.0, beta=0.0, gamma=1.0, mu=0.0)

COPULA_CONFIGS = {
    "gaussian": {"rho": gaussian_rho_for_tau(TARGET_TAU)},
    "clayton":  {"theta": clayton_theta_for_tau(TARGET_TAU)},
    "t":        {"rho": gaussian_rho_for_tau(TARGET_TAU), "df": T_DF},
    "gumbel":   {"theta": gumbel_theta_for_tau(TARGET_TAU)},
}

SAMPLE_SIZES = [500, 1000, 2500, 5000]
N_REPS = 50  # restored after the FFT density rewrite collapsed wall-clock
ESTIMATORS = ["canonical", "nts_oracle", "nts_fitted"]


# --- Worker function (must be top-level for multiprocessing) ------------

def _sample_in_x_space(copula: str, params: dict, n: int, rng):
    """Draw uniforms from the copula and push through NTS marginals."""
    if copula == "gaussian":
        U = sample_gaussian_copula(n, params["rho"], rng)
    elif copula == "clayton":
        U = sample_clayton_copula(n, params["theta"], rng)
    elif copula == "t":
        U = sample_t_copula(n, params["rho"], params["df"], rng)
    elif copula == "gumbel":
        U = sample_gumbel_copula(n, params["theta"], rng)
    else:
        raise ValueError(f"unknown copula '{copula}'")
    return apply_nts_marginals(U, NTS_MARGINAL, NTS_MARGINAL)


def run_one_rep(args):
    """Run all three estimators on a single replication.

    Parameters
    ----------
    args : tuple
        (copula, n, rep_id) where rep_id is the replication index used for seeding.

    Returns
    -------
    (copula, n, rep_id, results_dict)
        results_dict has keys "canonical", "nts_oracle", "nts_fitted",
        each mapping to a tuple (rho_grid, rho_quant) of arrays.
    """
    copula, n, rep_id = args
    params = COPULA_CONFIGS[copula]

    # Deterministic seed per (copula, n, rep) so reruns are reproducible.
    seed_str = f"{copula}_{n}_{rep_id}"
    seed = abs(hash(seed_str)) % (2**31 - 1)
    rng = np.random.default_rng(seed)

    X = _sample_in_x_space(copula, params, n, rng)
    grid = make_eval_grid()
    qpts = quantile_points()
    all_pts = np.vstack([grid, qpts])
    n_grid = grid.shape[0]

    out = {}
    rho_can = canonical_lgc(X, all_pts)
    out["canonical"] = (rho_can[:n_grid], rho_can[n_grid:])

    rho_orc = nts_lgc_oracle(X, all_pts, NTS_MARGINAL, NTS_MARGINAL)
    out["nts_oracle"] = (rho_orc[:n_grid], rho_orc[n_grid:])

    rho_fit = nts_lgc_fitted(X, all_pts)
    out["nts_fitted"] = (rho_fit[:n_grid], rho_fit[n_grid:])

    return copula, n, rep_id, out


# --- Driver --------------------------------------------------------------

CHECKPOINT_EVERY = 25  # save progress every N completed tasks


def _save_checkpoint(out_path, grid_arrays, quant_arrays):
    """Persist current state of all per-estimator arrays to npz.

    Called periodically during the run so that an interrupted process can
    resume from the last completed checkpoint instead of starting over.
    """
    save_kwargs = {}
    for (copula, n, est), arr in grid_arrays.items():
        save_kwargs[f"rho_grid_{copula}_{n}_{est}"] = arr
    for (copula, n, est), arr in quant_arrays.items():
        save_kwargs[f"rho_quant_{copula}_{n}_{est}"] = arr
    # Write atomically: write to a temp path, then replace.
    # NB: np.savez auto-appends ".npz" if the path does not end in .npz, so the
    # temp filename must already end in .npz to avoid a surprise rename target.
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(tmp, **save_kwargs)
    os.replace(tmp, out_path)


def main() -> None:
    out_path = IMPL_ROOT / "results" / "mc_raw.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Allocate output arrays keyed by (copula, n, estimator).
    grid_arrays = {}
    quant_arrays = {}
    for copula in COPULA_CONFIGS:
        for n in SAMPLE_SIZES:
            for est in ESTIMATORS:
                grid_arrays[(copula, n, est)]  = np.full((N_REPS, 121), np.nan)
                quant_arrays[(copula, n, est)] = np.full((N_REPS, 5),  np.nan)

    # Resume from any prior checkpoint.
    resumed = 0
    if out_path.exists():
        prior = np.load(out_path)
        for (copula, n, est) in grid_arrays.keys():
            grid_key  = f"rho_grid_{copula}_{n}_{est}"
            quant_key = f"rho_quant_{copula}_{n}_{est}"
            if grid_key in prior.files:
                grid_arrays[(copula, n, est)]  = prior[grid_key]
                quant_arrays[(copula, n, est)] = prior[quant_key]

        # A (copula, n, rep_id) task is "done" iff the canonical entry is finite.
        # We use canonical as the witness because it is the first estimator to
        # populate per rep; if it is finite, the others will be too (the
        # worker fills all three before returning).
        for copula in COPULA_CONFIGS:
            for n in SAMPLE_SIZES:
                arr = grid_arrays[(copula, n, "canonical")]
                resumed += int(np.sum(np.isfinite(arr[:, 0])))
        if resumed > 0:
            print(f"Resuming from prior checkpoint: {resumed} tasks already completed.\n")

    # Build the task list, skipping completed tasks.
    tasks = []
    for copula in COPULA_CONFIGS:
        for n in SAMPLE_SIZES:
            for rep_id in range(N_REPS):
                done_row = grid_arrays[(copula, n, "canonical")][rep_id]
                if np.isfinite(done_row[0]):
                    continue
                tasks.append((copula, n, rep_id))
    remaining = len(tasks)
    total = len(COPULA_CONFIGS) * len(SAMPLE_SIZES) * N_REPS
    print(f"MC harness: {total} tasks total, {remaining} remaining "
          f"({len(COPULA_CONFIGS)} copulas x {len(SAMPLE_SIZES)} sample sizes "
          f"x {N_REPS} reps), 3 estimators each.\n")

    if remaining == 0:
        print("Nothing to do -- all tasks already complete.")
        return

    n_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Parallelizing across {n_workers} worker processes. "
          f"Checkpointing every {CHECKPOINT_EVERY} completed tasks.\n")

    t0 = time.time()
    done = 0

    from multiprocessing import Pool

    with Pool(processes=n_workers) as pool:
        for copula, n, rep_id, out in pool.imap_unordered(run_one_rep, tasks, chunksize=1):
            for est in ESTIMATORS:
                grid_arrays[(copula, n, est)][rep_id]  = out[est][0]
                quant_arrays[(copula, n, est)][rep_id] = out[est][1]
            done += 1

            if done % CHECKPOINT_EVERY == 0 or done == remaining:
                _save_checkpoint(out_path, grid_arrays, quant_arrays)

            if done % 10 == 0 or done == remaining:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (remaining - done) / rate if rate > 0 else float("inf")
                print(f"  {done:4d} / {remaining:4d}  ({100*done/remaining:5.1f}%)  "
                      f"elapsed = {elapsed/60:5.1f} min  ETA = {eta/60:5.1f} min")

    _save_checkpoint(out_path, grid_arrays, quant_arrays)
    total_t = time.time() - t0
    print(f"\nDone in {total_t/60:.1f} min. Results saved to {out_path}")


if __name__ == "__main__":
    main()
