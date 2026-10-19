#!/usr/bin/env python3
"""Compute DAS improvement ratios from experiment pickles.

Mirrors the synthetic-experiment summary script, adapted to this repo's
``{params, results[i][algo][eval]}`` layout (and optional ``*_stz.pkl``).

Improvement (OPE, same as plot_trend.py)::

    (V_dast - V_comp) / V_comp * 100

STZ / STZ-VR use analysis/stz.py (needs implementation sidecar).

Examples::

  python analysis/compute_ratio.py \\
    exp_aug_larger_M/hillstrom/conversion/lightgbm_reg/sample_frac_with_fixed_0.40_pilot_frac/sample_frac_100.pkl

  python analysis/compute_ratio.py \\
    --exp-dir exp_aug_larger_M/hillstrom/conversion/lightgbm_reg/sample_frac_with_fixed_0.40_pilot_frac \\
    --eval-method dual_dr

  python analysis/compute_ratio.py PATH.pkl --metric stz
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_ANALYSIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ANALYSIS_DIR.parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plot_style import PREFERRED_ORDER, baseline_label  # noqa: E402
from stz import STZ_VR, STZ_basic  # noqa: E402

DAST_ALGO = "dast"

REQUESTED_COMPARATORS = [
    "kmeans",
    "kmeans_dams",
    "gmm",
    "gmm_dams",
    "clr",
    "clr_dams",
    "mst",
    "causal_forest",
    "t_learner",
    "s_learner",
    "x_learner",
    "dr_learner",
    "policytree",
]

OPE_EVAL_METHODS = ("dual_dr", "dr", "ipw", "dm")
_METRICS = ("improvement", "stz", "stz_vr")

# Same spirit as the synthetic script: drop a few extremes per comparator.
_APPLY_REMOVE_EXTREME = {
    "gmm": True,
    "kmeans": True,
    "mst": True,
    "clr": True,
    "t_learner": True,
    "x_learner": True,
    "dr_learner": True,
    "s_learner": True,
    "causal_forest": True,
    "policytree": True,
    "kmeans_dams": True,
    "gmm_dams": True,
    "clr_dams": True,
}


def _pkl_load(path: Path):
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as f:
            return pickle.load(f)


def _is_main_pkl(path: Path) -> bool:
    return path.suffix == ".pkl" and not path.stem.endswith("_stz")


def _stz_path(main: Path) -> Path:
    return main.with_name(main.stem + "_stz" + main.suffix)


def merge_stz_sidecar(results: list[dict], main_pkl: Path) -> int:
    stz = _stz_path(main_pkl)
    if not stz.is_file():
        return 0
    stz_data = _pkl_load(stz)
    by_seed = {
        run["seed"]: run["implementation"]
        for run in stz_data.get("results", [])
        if isinstance(run, dict) and "implementation" in run and "seed" in run
    }
    n = 0
    for run in results:
        impl = by_seed.get(run.get("seed"))
        if impl is None:
            continue
        run["implementation"] = impl
        n += 1
    return n


def print_params(params: dict) -> None:
    keys = [
        "dataset",
        "target_col",
        "sample_frac",
        "pilot_frac",
        "treatment_cost",
        "N_sim",
        "seed_sequence",
        "mu_model_type",
        "meta_learner_mu_model_type",
        "value_type_dast",
        "value_type_dams",
        "action_method",
        "n_folds_dams",
        "min_leaf_size",
        "attempts_used",
        "ALGO_LIST",
        "POLICYTREE_DEPTH",
        "POLICYTREE_SEARCH_DEPTH",
    ]
    print("=== Experiment Parameters ===")
    for key in keys:
        if key in params:
            print(f"{key:>28}: {params[key]}")
    seeds = params.get("seeds")
    if isinstance(seeds, list):
        print(f"{'n_seeds':>28}: {len(seeds)}")


def safe_get_value(run: dict, algo: str, ev: str) -> float:
    block = run.get(algo)
    if not isinstance(block, dict):
        return float("nan")
    v = block.get(ev, np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def das_improvement_ratio(vt: float, vb: float) -> float:
    """(V_dast - V_comp) / V_comp * 100  — same as plot_trend.py."""
    if not np.isfinite(vt) or not np.isfinite(vb) or abs(vb) < 1e-12:
        return float("nan")
    return float((vt - vb) / vb * 100.0)


def discover_comparators(results: list[dict]) -> list[str]:
    """Include a comparator if it appears in *any* run (partial backfills OK)."""
    keys: set[str] = set()
    for run in results:
        if isinstance(run, dict):
            keys |= set(run.keys())
    return [b for b in REQUESTED_COMPARATORS if b in keys]


def compute_ope_improvement(
    results: list[dict],
    comparators: list[str],
    eval_method: str,
    *,
    das_algo: str = DAST_ALGO,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {c: [] for c in comparators}
    for run in results:
        vt = safe_get_value(run, das_algo, eval_method)
        if not np.isfinite(vt):
            continue
        for comp in comparators:
            vb = safe_get_value(run, comp, eval_method)
            if not np.isfinite(vb):
                continue
            ratio = das_improvement_ratio(vt, vb)
            if np.isfinite(ratio):
                out[comp].append(ratio)
    return out


def compute_stz_improvement(
    results: list[dict],
    comparators: list[str],
    *,
    stz_fn,
    das_algo: str = DAST_ALGO,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {c: [] for c in comparators}
    for run in results:
        for comp in comparators:
            adv = stz_fn(run, das_algo, comp)
            if np.isfinite(adv):
                out[comp].append(float(adv))
    return out


def filter_ratios(
    improvement_ratios: dict[str, list[float]],
    apply_remove_extreme: dict[str, bool],
    apply_sigma_clip: bool,
    *,
    sigma: float = 1.5,
    n_trim: int = 3,
) -> dict[str, np.ndarray]:
    filtered: dict[str, np.ndarray] = {}
    for comp, ratios in improvement_ratios.items():
        ratios_np = np.asarray(ratios, dtype=float)
        ratios_np = ratios_np[np.isfinite(ratios_np)]
        if ratios_np.size == 0:
            filtered[comp] = ratios_np
            continue

        if apply_remove_extreme.get(comp, False) and len(ratios_np) > 2 * n_trim:
            min_indices = np.argsort(ratios_np)[:n_trim]
            mask = np.ones_like(ratios_np, dtype=bool)
            mask[min_indices] = False
            ratios_np = ratios_np[mask]

            max_indices = np.argsort(ratios_np)[-n_trim:]
            mask = np.ones_like(ratios_np, dtype=bool)
            mask[max_indices] = False
            ratios_np = ratios_np[mask]

        if apply_sigma_clip and len(ratios_np) >= 4:
            mean = float(np.mean(ratios_np))
            std = float(np.std(ratios_np, ddof=1)) if len(ratios_np) > 1 else 0.0
            if std > 0:
                ratios_np = ratios_np[np.abs(ratios_np - mean) <= sigma * std]

        filtered[comp] = ratios_np
    return filtered


def print_coverage(
    n_results: int,
    raw_counts: dict[str, int],
    expected_n: int | None,
) -> None:
    target = int(expected_n) if expected_n is not None else n_results
    print("\nCoverage (usable DAS vs comparator pairs):")
    for comp in REQUESTED_COMPARATORS:
        n = int(raw_counts.get(comp, 0))
        label = baseline_label(comp)
        flag = "" if n >= target else "  [incomplete]"
        print(f"{label:>20}: {n:3d}/{target}{flag}")


def print_summary(
    filtered_ratios: dict[str, np.ndarray],
    raw_counts: dict[str, int],
    n_results: int,
    title: str,
    *,
    expected_n: int | None = None,
) -> None:
    target = int(expected_n) if expected_n is not None else n_results
    label_to_key = {baseline_label(k): k for k in filtered_ratios}
    ordered_labels = [x for x in PREFERRED_ORDER if x in label_to_key]
    remaining = sorted(
        baseline_label(k) for k in filtered_ratios if baseline_label(k) not in ordered_labels
    )
    print(f"\n{title}:")
    n_printed = 0
    for label in ordered_labels + remaining:
        key = label_to_key[label]
        ratios = np.asarray(filtered_ratios[key], dtype=float)
        raw_n = int(raw_counts.get(key, len(ratios)))
        if len(ratios) == 0:
            continue
        n = len(ratios)
        mean = float(np.mean(ratios))
        se = float(stats.sem(ratios, ddof=1)) if n > 1 else 0.0
        ci = float(se * stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
        flag = "  [incomplete]" if raw_n < target else ""
        print(
            f"{label:>20}: {n:3d} used / {raw_n:3d} avail / {target} total | "
            f"Avg: {mean:7.2f}% | 95% CI: [{mean - ci:7.2f}%, {mean + ci:7.2f}%]{flag}"
        )
        n_printed += 1
    if n_printed == 0:
        print("  (no comparator ratios available yet)")


def load_main_pkl(path: Path) -> tuple[dict, list[dict]]:
    data = _pkl_load(path)
    if not isinstance(data, dict) or "results" not in data:
        raise RuntimeError(f"Unexpected pickle format: {path}")
    params = dict(data.get("params", {}))
    results = [r for r in data["results"] if isinstance(r, dict)]
    return params, results


def summarize_results(
    results: list[dict],
    *,
    metric: str,
    eval_method: str,
    sigma_clip: bool,
    remove_extreme: bool,
    title_prefix: str,
    expected_n: int | None = None,
    params: dict | None = None,
) -> None:
    present = discover_comparators(results)
    if params is not None:
        print_params(params)
    print(f"n_results: {len(results)}")
    print(f"Comparators with ≥1 run present: {present}")

    apply_extreme = _APPLY_REMOVE_EXTREME if remove_extreme else {}

    if metric == "improvement":
        ratios = compute_ope_improvement(results, present, eval_method)
        title = f"{title_prefix} DAS improvement ratio (%)  eval={eval_method}".strip()
    else:
        stz_fn = STZ_basic if metric == "stz" else STZ_VR
        ratios = compute_stz_improvement(results, present, stz_fn=stz_fn)
        title = f"{title_prefix} DAS improvement ratio (%)  metric={metric}".strip()

    raw_counts = {c: len(v) for c, v in ratios.items()}
    for c in REQUESTED_COMPARATORS:
        raw_counts.setdefault(c, 0)

    print_coverage(len(results), raw_counts, expected_n)
    filtered = filter_ratios(ratios, apply_extreme, sigma_clip)
    filtered_print = {c: filtered[c] for c in present if c in filtered and len(ratios[c]) > 0}
    print_summary(
        filtered_print,
        raw_counts,
        len(results),
        title=title,
        expected_n=expected_n,
    )


def run_one(
    pkl_path: Path,
    *,
    metric: str,
    eval_method: str,
    sigma_clip: bool,
    remove_extreme: bool,
) -> list[dict]:
    params, results = load_main_pkl(pkl_path)
    print(f"\n{'=' * 60}")
    print(f"File: {pkl_path}")

    if metric != "improvement":
        n_merged = merge_stz_sidecar(results, pkl_path)
        print(f"Merged STZ implementation blocks: {n_merged}/{len(results)}")

    expected = params.get("N_sim")
    try:
        expected_n = int(expected) if expected is not None else len(results)
    except (TypeError, ValueError):
        expected_n = len(results)

    summarize_results(
        results,
        metric=metric,
        eval_method=eval_method,
        sigma_clip=sigma_clip,
        remove_extreme=remove_extreme,
        title_prefix="",
        expected_n=expected_n,
        params=params,
    )
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute DAS improvement ratios from experiment pickles."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("file", nargs="?", default=None, help="Main experiment .pkl")
    g.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Directory of main experiment pickles",
    )
    p.add_argument(
        "--metric",
        choices=_METRICS,
        default="improvement",
        help=(
            "improvement: OPE (V_dast-V_comp)/V_comp*100 [default]; "
            "stz / stz_vr: Matched-Action variants (need *_stz.pkl)"
        ),
    )
    p.add_argument(
        "--eval-method",
        choices=OPE_EVAL_METHODS,
        default="dual_dr",
        help="OPE field for --metric improvement (default: dual_dr)",
    )
    p.add_argument(
        "--sigma-clip",
        action="store_true",
        help="After extreme trim, keep values within 1.5 sample SD of the mean.",
    )
    p.add_argument(
        "--no-remove-extreme",
        action="store_true",
        help="Disable dropping 3 lowest + 3 highest ratios per comparator.",
    )
    p.add_argument(
        "--no-pool",
        action="store_true",
        help="With --exp-dir, skip the pooled summary across all pickles.",
    )
    p.add_argument(
        "--pool-only",
        action="store_true",
        help="With --exp-dir, only print the pooled summary (no per-file tables).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.exp_dir:
        exp_dir = Path(args.exp_dir).expanduser().resolve()
        pkls = sorted(p for p in exp_dir.glob("*.pkl") if _is_main_pkl(p))
        if not pkls:
            raise SystemExit(f"No main .pkl files in {exp_dir}")
    else:
        if not args.file:
            raise SystemExit("Provide a .pkl path or --exp-dir")
        pkls = [Path(args.file).expanduser().resolve()]
        if not _is_main_pkl(pkls[0]):
            raise SystemExit(f"Pass a main .pkl, not *_stz.pkl: {pkls[0].name}")

    all_results: list[dict] = []
    for pkl in pkls:
        if args.pool_only and args.exp_dir:
            params, results = load_main_pkl(pkl)
            if args.metric != "improvement":
                merge_stz_sidecar(results, pkl)
            all_results.extend(results)
            continue
        results = run_one(
            pkl,
            metric=args.metric,
            eval_method=args.eval_method,
            sigma_clip=args.sigma_clip,
            remove_extreme=not args.no_remove_extreme,
        )
        all_results.extend(results)

    if args.exp_dir and not args.no_pool:
        print(f"\n{'=' * 60}")
        print(f"POOLED over {len(pkls)} pickle(s)  ({len(all_results)} runs total)")
        print("(includes incomplete backfills; n avail may be < n total)")
        summarize_results(
            all_results,
            metric=args.metric,
            eval_method=args.eval_method,
            sigma_clip=args.sigma_clip,
            remove_extreme=not args.no_remove_extreme,
            title_prefix="[pooled]",
            expected_n=len(all_results),
            params=None,
        )


if __name__ == "__main__":
    main()
