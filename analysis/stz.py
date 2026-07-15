"""STZ (Simester, Timoshenko, and Zoumpoulis) advantage evaluator."""

from __future__ import annotations

import numpy as np


def STZ_evaluator(
    run: dict,
    algo_dast: str,
    algo_comp: str,
) -> float:
    """
    DAS advantage ratio for one simulation run using logged implementation data.

    Returns np.nan if implementation data or factual subsets are missing.
    """
    impl = run.get("implementation")
    if impl is None:
        return np.nan

    actions = impl.get("actions", {})
    if algo_dast not in actions or algo_comp not in actions:
        return np.nan

    D = np.asarray(impl["D"], dtype=int)
    y = np.asarray(impl["y"], dtype=float)
    a_dast = np.asarray(actions[algo_dast], dtype=int)
    a_comp = np.asarray(actions[algo_comp], dtype=int)

    n = len(D)
    if n == 0:
        return np.nan

    disagree_mask = a_dast != a_comp
    n_disagree = int(disagree_mask.sum())
    if n_disagree == 0:
        return 0.0
    weight = n_disagree / n

    dast_factual = disagree_mask & (D == a_dast)
    if dast_factual.sum() == 0:
        return np.nan
    v_dast = float(y[dast_factual].mean())

    comp_factual = disagree_mask & (D == a_comp)
    if comp_factual.sum() == 0:
        return np.nan
    v_comp = float(y[comp_factual].mean())

    if not np.isfinite(v_comp) or abs(v_comp) < 1e-12:
        return np.nan
    return float(weight * (v_dast - v_comp) / abs(v_comp) * 100.0)
