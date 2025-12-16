# causal_forest_benchmark.py

import gc
from contextlib import contextmanager
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector, StrVector

# R package
grf = importr("grf")


# -------------------------
# Memory cleanup utilities
# -------------------------

def r_gc() -> None:
    """Trigger R garbage collection."""
    ro.r("gc()")


def cleanup_cf_model(cf_model: Optional[Dict], *, run_r_gc: bool = True) -> None:
    """
    Best-effort cleanup for long loops:
    - drop Python references to R objects
    - trigger Python GC
    - optionally trigger R gc()
    """
    if cf_model is None:
        return

    # Drop reference to R external pointer(s)
    if isinstance(cf_model, dict):
        if "forest" in cf_model:
            cf_model["forest"] = None

    # Python GC
    gc.collect()

    # R GC
    if run_r_gc:
        r_gc()


@contextmanager
def fitted_multiarm_causal_forest(
    X_pilot: np.ndarray,
    y_pilot: np.ndarray,
    D_pilot: np.ndarray,
    *,
    action_levels: Optional[Sequence[int]],
    num_trees: int,
    seed: int = 0,
    cleanup: bool = True,
    run_r_gc: bool = True,
):
    """
    Context manager wrapper to ensure cleanup in long-running loops.

    Usage:
        with fitted_multiarm_causal_forest(X, y, D, seed=seed) as cf_model:
            a_hat, mu_hat = predict_best_action_multiarm(cf_model, X_test)
    """
    cf_model = fit_multiarm_causal_forest(
        X_pilot,
        y_pilot,
        D_pilot,
        action_levels=action_levels,
        num_trees=num_trees,
        seed=seed,
    )
    try:
        yield cf_model
    finally:
        if cleanup:
            cleanup_cf_model(cf_model, run_r_gc=run_r_gc)


# -------------------------
# Core API
# -------------------------

def fit_multiarm_causal_forest(
    X_pilot: np.ndarray,
    y_pilot: np.ndarray,
    D_pilot: np.ndarray,
    *,
    action_levels: Optional[Sequence[int]],
    num_trees: int,
    seed: int,
) -> Dict:
    """
    Fit grf::multi_arm_causal_forest on pilot data.

    Parameters
    ----------
    X_pilot : (n, d)
    y_pilot : (n,)
    D_pilot : (n,) int actions (0..K-1)
    action_levels : list[int] or None
        R factor 的 levels 顺序；None 则用 sorted(unique(D_pilot))
    num_trees : int
    seed : int

    Returns
    -------
    model : dict
        {
          "forest": R multi_arm_causal_forest object,
          "levels": np.ndarray of int (arm labels in column order),
        }
    """
    X_pilot = np.asarray(X_pilot, dtype=float)
    y_pilot = np.asarray(y_pilot, dtype=float).ravel()
    D_pilot = np.asarray(D_pilot).astype(int).ravel()

    if action_levels is None:
        levels = np.array(sorted(np.unique(D_pilot).tolist()), dtype=int)
    else:
        levels = np.array(list(action_levels), dtype=int)

    # R wants factor W with levels (strings)
    level_str = [str(a) for a in levels]
    D_str = [str(a) for a in D_pilot]

    with localconverter(default_converter + numpy2ri.converter):
        X_r = ro.conversion.py2rpy(X_pilot)

    y_r = FloatVector(y_pilot.tolist())
    W_r = ro.r["factor"](StrVector(D_str), levels=StrVector(level_str))

    forest = grf.multi_arm_causal_forest(
        X_r,
        y_r,
        W_r,
        num_trees=num_trees,
        seed=seed,
        min_node_size=2,
    )

    return {
        "forest": forest,
        "levels": levels,   # column order for mu_hat
    }


def predict_mu_hat_matrix_multiarm(cf_model: Dict, X: np.ndarray) -> np.ndarray:
    """
    Return a relative mu_hat matrix (n, K):
      column 0 = 0 baseline (reference arm)
      column a = tau_hat for arm a (a=1..K-1)
    This is sufficient for argmax policy selection.
    """
    X = np.asarray(X, dtype=float)
    forest = cf_model["forest"]
    K = len(cf_model["levels"])

    with localconverter(default_converter + numpy2ri.converter):
        X_r = ro.conversion.py2rpy(X)

    pred = grf.predict_multi_arm_causal_forest(forest, X_r)

    with localconverter(default_converter + numpy2ri.converter):
        tau_hat = np.asarray(ro.conversion.rpy2py(pred.rx2("predictions")), dtype=float)

    tau_hat = np.squeeze(tau_hat)
    if tau_hat.ndim == 1:
        tau_hat = tau_hat.reshape(-1, 1)

    # build relative mu_hat: baseline=0, others=tau
    if tau_hat.shape[1] != K - 1:
        raise ValueError(f"Expected tau_hat with {K-1} columns, got {tau_hat.shape}")

    mu_hat_rel = np.column_stack([np.zeros(tau_hat.shape[0]), tau_hat])
    return mu_hat_rel

def predict_best_action_multiarm(cf_model: Dict, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return best action per individual by argmax over mu_hat columns.

    Returns
    -------
    a_hat : (n,) int action labels (same labels as cf_model["levels"])
    mu_hat : (n, K) float
    """
    levels = np.asarray(cf_model["levels"], dtype=int)
    if levels[0] != 0:
        raise ValueError(f"Baseline arm is levels[0]={levels[0]}, expected 0. "
                        "Either pass action_levels=np.arange(K) or adjust evaluation mapping.")

    mu_hat_rel = predict_mu_hat_matrix_multiarm(cf_model, X)

    idx = np.argmax(mu_hat_rel, axis=1).astype(int)
    a_hat = levels[idx]
    return a_hat, mu_hat_rel
