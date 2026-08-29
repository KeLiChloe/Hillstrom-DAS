"""STZ (Simester, Timoshenko, and Zoumpoulis) advantage evaluators.

Two Y-only variants on logged implementation data:

  STZ_basic  (V_dast - V_comp) / |V_comp| * 100 on the full implementation set
             (factual means, no disagreement filter).

  STZ_VR     Δ_STZ = w (V_d - V_c) on the disagreement set, then
             Δ_STZ / |V_comp,overall| * 100
             (V_comp,overall = factual mean on the full set).

Also: factual net profit and treatment rate for cost-sweep analysis.
"""

from __future__ import annotations

import numpy as np

REFERENCE_ACTION = 0


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


def _factual_net_mean(y, a_policy, D, treatment_cost, reference_action=REFERENCE_ACTION):
    factual = D == a_policy
    if factual.sum() == 0:
        return float("nan")
    c = float(treatment_cost)
    ref = int(reference_action)
    if c > 0:
        net = y[factual] - c * (a_policy[factual] != ref)
    else:
        net = y[factual]
    return float(net.mean())


def STZ_basic_net(
    run: dict,
    algo_dast: str,
    algo_comp: str,
    treatment_cost: float,
    *,
    reference_action: int = REFERENCE_ACTION,
) -> float:
    """STZ_basic on factual net outcome (y - c·𝟙[treat])."""
    arrays = _impl_arrays(run, algo_dast, algo_comp)
    if arrays is None:
        return np.nan

    D, y, a_dast, a_comp = arrays
    v_dast = _factual_net_mean(y, a_dast, D, treatment_cost, reference_action)
    v_comp = _factual_net_mean(y, a_comp, D, treatment_cost, reference_action)
    if not np.isfinite(v_dast) or not np.isfinite(v_comp):
        return np.nan
    return _pct_of_comp(v_dast - v_comp, v_comp)


def STZ_VR_net(
    run: dict,
    algo_dast: str,
    algo_comp: str,
    treatment_cost: float,
    *,
    reference_action: int = REFERENCE_ACTION,
) -> float:
    """STZ-VR on factual net outcome."""
    arrays = _impl_arrays(run, algo_dast, algo_comp)
    if arrays is None:
        return np.nan

    D, y, a_dast, a_comp = arrays
    n = len(D)
    c = float(treatment_cost)
    ref = int(reference_action)

    disagree_mask = a_dast != a_comp
    n_disagree = int(disagree_mask.sum())
    if n_disagree == 0:
        return 0.0
    weight = n_disagree / n

    dast_factual = disagree_mask & (D == a_dast)
    if dast_factual.sum() == 0:
        return np.nan
    if c > 0:
        net_d = y[dast_factual] - c * (a_dast[dast_factual] != ref)
    else:
        net_d = y[dast_factual]
    v_d = float(net_d.mean())

    comp_factual_disagree = disagree_mask & (D == a_comp)
    if comp_factual_disagree.sum() == 0:
        return np.nan
    if c > 0:
        net_c = y[comp_factual_disagree] - c * (a_comp[comp_factual_disagree] != ref)
    else:
        net_c = y[comp_factual_disagree]
    v_c = float(net_c.mean())

    delta_stz = weight * (v_d - v_c)

    comp_factual_overall = D == a_comp
    if comp_factual_overall.sum() == 0:
        return np.nan
    if c > 0:
        net_overall = y[comp_factual_overall] - c * (a_comp[comp_factual_overall] != ref)
    else:
        net_overall = y[comp_factual_overall]
    v_comp_overall = float(net_overall.mean())

    return _pct_of_comp(delta_stz, v_comp_overall)


def _impl_action_array(run: dict, algo: str):
    """Return (D, y, a) from implementation block, or None."""
    impl = run.get("implementation")
    if impl is None:
        return None
    actions = impl.get("actions", {})
    if algo not in actions:
        return None
    D = np.asarray(impl["D"], dtype=int)
    y = np.asarray(impl["y"], dtype=float)
    a = np.asarray(actions[algo], dtype=int)
    if len(D) == 0:
        return None
    return D, y, a


def treatment_rate(
    run: dict,
    algo: str,
    *,
    reference_action: int = REFERENCE_ACTION,
) -> float:
    """
    Fraction of implementation users assigned a non-reference action.

    Returns a value in [0, 1]; multiply by 100 for percent.
    """
    arrays = _impl_action_array(run, algo)
    if arrays is None:
        return float("nan")
    _, _, a = arrays
    return float(np.mean(a != int(reference_action)))


def factual_net_profit(
    run: dict,
    algo: str,
    treatment_cost: float,
    *,
    reference_action: int = REFERENCE_ACTION,
) -> float:
    """
    Factual mean net outcome on users where logged D matches policy action.

    net_i = y_i - c * 1[a_i != reference_action]
    V = mean(net_i | D_i == a_i)

    treatment_cost must be in the same units as y (e.g. c = 10/50 when y is 0/1).
    """
    arrays = _impl_action_array(run, algo)
    if arrays is None:
        return float("nan")
    D, y, a = arrays
    c = float(treatment_cost)
    ref = int(reference_action)
    factual = D == a
    if factual.sum() == 0:
        return float("nan")
    net = y[factual] - c * (a[factual] != ref)
    return float(net.mean())


def net_profit_advantage(
    run: dict,
    algo_dast: str,
    algo_comp: str,
    treatment_cost: float,
    *,
    reference_action: int = REFERENCE_ACTION,
) -> float:
    """
    (net_profit_dast - net_profit_comp) / |net_profit_comp| * 100
    using factual_net_profit on each algorithm's recommended action.
    """
    v_dast = factual_net_profit(
        run, algo_dast, treatment_cost, reference_action=reference_action
    )
    v_comp = factual_net_profit(
        run, algo_comp, treatment_cost, reference_action=reference_action
    )
    return _pct_of_comp(v_dast - v_comp, v_comp)


# Backward-compatible alias: historical STZ_evaluator == STZ_VR.
STZ_evaluator = STZ_VR
