#!/usr/bin/env python3
"""Backfill policytree into existing experiment pickles (merge, do not re-run others).

For each main ``*.pkl`` under an experiment directory (or a single pkl), reuse the
stored successful seeds and all saved hyper-parameters, run **only** policytree,
then merge:

  - ``results[i]["policytree"]``  (OPE scalars + depth / time)
  - ``implementation.actions["policytree"]`` in the ``*_stz.pkl`` sidecar
  - append ``"policytree"`` to ``params["ALGO_LIST"]`` if missing

Example (Hillstrom sample_frac sweep)::

  /home/5880.li/miniconda3/envs/hillstrom/bin/python scripts/backfill_policytree.py \\
    --exp-dir exp_aug_larger_M/hillstrom/conversion/lightgbm_reg/sample_frac_with_fixed_0.40_pilot_frac

Example (Criteo sample_frac sweep)::

  /home/5880.li/miniconda3/envs/hillstrom/bin/python scripts/backfill_policytree.py \\
    --exp-dir exp_aug_larger_M/criteo/conversion/lightgbm_reg/sample_frac_with_fixed_0.40_pilot_frac

Options::

  --force     recompute even if policytree already present
  --dry-run   print plan only
  --pkl PATH  backfill one main pkl instead of a whole directory
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gzip
import pickle

import numpy as np


def _pkl_load(path: Path):
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as f:
            return pickle.load(f)


def _is_main_pkl(path: Path) -> bool:
    return path.suffix == ".pkl" and not path.stem.endswith("_stz")


def discover_main_pkls(exp_dir: Path) -> list[Path]:
    return sorted(p for p in exp_dir.glob("*.pkl") if _is_main_pkl(p))


def _stz_path_for(main_pkl: Path) -> Path:
    return main_pkl.with_name(main_pkl.stem + "_stz" + main_pkl.suffix)


def backfill_one_pkl(
    pkl_path: Path,
    *,
    force: bool,
    dry_run: bool,
) -> tuple[int, int]:
    """Return (n_skipped, n_ran_or_planned)."""
    # Lazy import: keeps --dry-run usable even if rpy2/R is unhappy.
    import run_sims
    from run_sims import (
        POLICYTREE_DEPTH,
        POLICYTREE_SEARCH_DEPTH,
        _merge_stz_into_results,
        _save_experiment_checkpoints,
        normalize_main_pkl_path,
        resolve_stz_pkl_path,
        run_single_experiment,
    )

    pkl_path = Path(normalize_main_pkl_path(str(pkl_path))).resolve()
    stz_path = Path(resolve_stz_pkl_path(str(pkl_path)))

    data = _pkl_load(pkl_path)
    if not isinstance(data, dict) or "params" not in data or "results" not in data:
        raise RuntimeError(f"Unexpected pickle format: {pkl_path}")

    params = data["params"]
    results = data["results"]
    if stz_path.is_file():
        _merge_stz_into_results(data, _pkl_load(stz_path))

    algo_list = list(params.get("ALGO_LIST") or [])
    if "policytree" not in algo_list:
        algo_list.append("policytree")

    n_skip = 0
    n_run = 0
    print(f"\n=== {pkl_path.name}  ({len(results)} runs) ===")
    print(
        f"policytree config: depth={POLICYTREE_DEPTH}, "
        f"search_depth={POLICYTREE_SEARCH_DEPTH}"
    )

    # Restrict run_single_experiment to policytree only.
    run_sims.ALGO_LIST[:] = ["policytree"]

    for i, run in enumerate(results):
        seed = int(run["seed"])
        has_pt = (
            isinstance(run.get("policytree"), dict)
            and run["policytree"].get("dual_dr") is not None
        )
        if (not force) and has_pt:
            n_skip += 1
            print(f"  [{i+1}/{len(results)}] seed={seed} SKIP (already has policytree)")
            continue

        print(f"  [{i+1}/{len(results)}] seed={seed} RUN policytree ...", flush=True)
        if dry_run:
            n_run += 1
            continue

        t0 = time.perf_counter()
        fresh = run_single_experiment(
            sample_frac=float(params["sample_frac"]),
            pilot_frac=float(params["pilot_frac"]),
            train_frac=float(params.get("train_frac", 0.7)),
            dataset=str(params["dataset"]),
            target_col=str(params["target_col"]),
            mu_model_type=str(params["mu_model_type"]),
            meta_learner_mu_model_type=str(params["meta_learner_mu_model_type"]),
            value_type_dast=str(params["value_type_dast"]),
            value_type_dams=str(params["value_type_dams"]),
            seed=seed,
            save_offline_data=True,
            action_method=str(params.get("action_method", "diff_in_means")),
            n_folds_dams=int(params.get("n_folds_dams", 5)),
            ope_mu_hparams=params.get("ope_mu_hparams"),
            meta_mu_hparams=params.get("meta_mu_hparams"),
            min_leaf_size=int(params["min_leaf_size"]),
            treatment_cost=float(params.get("treatment_cost", 0.0)),
        )
        if "policytree" not in fresh:
            raise RuntimeError(f"policytree missing from fresh run (seed={seed})")

        from action_encoding import align_fresh_algo_actions_to_sidecar

        run["policytree"] = fresh["policytree"]
        target_ipw = float(fresh["policytree"]["ipw"])
        if "implementation" not in run:
            run["implementation"] = fresh["implementation"]
        else:
            aligned, perm = align_fresh_algo_actions_to_sidecar(
                run["implementation"],
                fresh["implementation"],
                "policytree",
                target_ipw,
            )
            if not np.array_equal(perm, np.arange(len(perm))):
                print(
                    f"  [{i+1}/{len(results)}] seed={seed} "
                    f"aligned action labels via perm={perm.tolist()} "
                    f"(sidecar D encoding ≠ current loader)",
                    flush=True,
                )
            actions = run["implementation"].setdefault("actions", {})
            actions["policytree"] = np.asarray(aligned, dtype=np.int8)

        params["ALGO_LIST"] = list(algo_list)
        params["POLICYTREE_DEPTH"] = POLICYTREE_DEPTH
        params["POLICYTREE_SEARCH_DEPTH"] = POLICYTREE_SEARCH_DEPTH
        data["params"] = params

        _save_experiment_checkpoints(
            str(pkl_path),
            str(stz_path),
            data,
            save_offline_data=True,
        )
        n_run += 1
        print(
            f"  [{i+1}/{len(results)}] seed={seed} OK "
            f"({time.perf_counter() - t0:.1f}s) → merged + saved",
            flush=True,
        )

    return n_skip, n_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill policytree into existing experiment pickles."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--exp-dir",
        type=str,
        help="Directory with sample_frac_*.pkl / treatment_cost_*.pkl",
    )
    g.add_argument(
        "--pkl",
        type=str,
        help="Single main experiment .pkl (not *_stz.pkl)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute policytree even if already present",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which seeds would run, without computing or writing",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.pkl:
        pkls = [Path(args.pkl).expanduser().resolve()]
        if not _is_main_pkl(pkls[0]):
            raise SystemExit(f"Pass a main .pkl, not a STZ sidecar: {pkls[0].name}")
    else:
        exp_dir = Path(args.exp_dir).expanduser().resolve()
        if not exp_dir.is_dir():
            raise SystemExit(f"Not a directory: {exp_dir}")
        pkls = discover_main_pkls(exp_dir)
        if not pkls:
            raise SystemExit(f"No main .pkl files in {exp_dir}")

    # Peek depth settings without importing R stacks on dry-run when possible.
    depth, search_depth = 3, 2
    try:
        from run_sims import POLICYTREE_DEPTH, POLICYTREE_SEARCH_DEPTH

        depth, search_depth = POLICYTREE_DEPTH, POLICYTREE_SEARCH_DEPTH
    except Exception:
        pass

    print(f"Backfill policytree into {len(pkls)} pickle(s)")
    print(f"Defaults from run_sims: depth={depth}, search_depth={search_depth}")
    if args.dry_run:
        print("[dry-run] no writes")

    total_skip = total_run = 0
    for pkl in pkls:
        if args.dry_run:
            # Lightweight plan without importing run_sims / rpy2.
            data = _pkl_load(pkl)
            results = data["results"]
            n_skip = n_run = 0
            print(f"\n=== {pkl.name}  ({len(results)} runs) ===")
            for i, run in enumerate(results):
                seed = int(run["seed"])
                has_pt = (
                    isinstance(run.get("policytree"), dict)
                    and run["policytree"].get("dual_dr") is not None
                )
                if (not args.force) and has_pt:
                    n_skip += 1
                    print(
                        f"  [{i+1}/{len(results)}] seed={seed} "
                        "SKIP (already has policytree)"
                    )
                else:
                    n_run += 1
                    print(f"  [{i+1}/{len(results)}] seed={seed} RUN policytree ...")
            total_skip += n_skip
            total_run += n_run
            continue

        n_skip, n_run = backfill_one_pkl(
            pkl, force=args.force, dry_run=False
        )
        total_skip += n_skip
        total_run += n_run

    print(
        f"\nDone. skipped={total_skip}, ran/planned={total_run}, "
        f"pkls={len(pkls)}"
    )


if __name__ == "__main__":
    main()
