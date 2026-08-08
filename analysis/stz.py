"""STZ (Simester, Timoshenko, and Zoumpoulis) advantage evaluators.

Two Y-only variants on logged implementation data:

  STZ_basic  (V_dast - V_comp) / |V_comp| * 100 on the full implementation set
             (factual means, no disagreement filter).

  STZ_VR     Δ_STZ = w (V_d - V_c) on the disagreement set, then
             Δ_STZ / |V_comp,overall| * 100
             (V_comp,overall = factual mean on the full set).
"""

from __future__ import annotations

import numpy as np


def _impl_arrays(run: dict, algo_dast: str, algo_comp: str):
    """Return (D, y, a_dast, a_comp) or None if implementation data is missing."""
    impl = run.get("implementation")
    if impl is None:
        return None

    actions = impl.get("actions", {})
    if algo_dast not in actions or algo_comp not in actions:
        return None

    D = np.asarray(impl["D"], dtype=int)
    y = np.asarray(impl["y"], dtype=float)
    a_dast = np.asarray(actions[algo_dast], dtype=int)
    a_comp = np.asarray(actions[algo_comp], dtype=int)

    if len(D) == 0:
        return None
    return D, y, a_dast, a_comp


def _pct_of_comp(numer: float, v_comp: float) -> float:
    if not np.isfinite(numer) or not np.isfinite(v_comp) or abs(v_comp) < 1e-12:
        return float("nan")
    return float(numer / abs(v_comp) * 100.0)


def STZ_basic(
    run: dict,
    algo_dast: str,
    algo_comp: str,
) -> float:
    """
    Basic STZ advantage (no disagreement filter).

    On the full implementation set:
      V_dast = mean(y | D == a_dast)
      V_comp = mean(y | D == a_comp)
      advantage = (V_dast - V_comp) / |V_comp| * 100

    Returns np.nan if implementation data or factual subsets are missing.
    """
    arrays = _impl_arrays(run, algo_dast, algo_comp)
    if arrays is None:
        return np.nan

    D, y, a_dast, a_comp = arrays

    dast_factual = D == a_dast
    if dast_factual.sum() == 0:
        return np.nan
    v_dast = float(y[dast_factual].mean())

    comp_factual = D == a_comp
    if comp_factual.sum() == 0:
        return np.nan
    v_comp = float(y[comp_factual].mean())

    return _pct_of_comp(v_dast - v_comp, v_comp)


def STZ_VR(
    run: dict,
    algo_dast: str,
    algo_comp: str,
) -> float:
    """
    STZ variance-reduction advantage (disagreement-restricted).

    On the disagreement set {a_dast != a_comp}, with w = n_disagree / n:
      V_d = mean(y | disagree, D == a_dast)
      V_c = mean(y | disagree, D == a_comp)
      Δ_STZ = w * (V_d - V_c)

    Normalize by the comparator factual mean on the full population:
      V_comp,overall = mean(y | D == a_comp)
      advantage = Δ_STZ / |V_comp,overall| * 100

    Returns np.nan if implementation data or factual subsets are missing.
    """
    arrays = _impl_arrays(run, algo_dast, algo_comp)
    if arrays is None:
        return np.nan

    D, y, a_dast, a_comp = arrays
    n = len(D)

    disagree_mask = a_dast != a_comp
    n_disagree = int(disagree_mask.sum())
    if n_disagree == 0:
        return 0.0
    weight = n_disagree / n

    dast_factual = disagree_mask & (D == a_dast)
    if dast_factual.sum() == 0:
        return np.nan
    v_d = float(y[dast_factual].mean())

    comp_factual_disagree = disagree_mask & (D == a_comp)
    if comp_factual_disagree.sum() == 0:
        return np.nan
    v_c = float(y[comp_factual_disagree].mean())

    delta_stz = weight * (v_d - v_c)

    comp_factual_overall = D == a_comp
    if comp_factual_overall.sum() == 0:
        return np.nan
    v_comp_overall = float(y[comp_factual_overall].mean())

    return _pct_of_comp(delta_stz, v_comp_overall)


# Backward-compatible alias: historical STZ_evaluator == STZ_VR.
STZ_evaluator = STZ_VR
