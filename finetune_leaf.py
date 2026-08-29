#!/usr/bin/env python3
"""
Sequential DAST min_leaf_size finetune (standalone).

Requires an existing hparams/{dataset}/{target}.json from finetune_mu.py.
Fits frozen OPE μ on the full sample, builds Gamma, then selects min_leaf_size
via DAMS CV (same select_dast_M_via_dams used in experiments). Writes/updates
only the top-level \"dast\" block.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_utils import load_criteo, load_hillstrom
from mu_hparams import (
    DEFAULT_LEAF_GRID,
    default_mu_hparams_path,
    get_dast_entry,
    load_hparams_payload,
    merge_hparams_payload,
    save_hparams_payload,
    upsert_dast_entry,
    dast_config_key,
    _params_for_type,
    _has_type,
)
from outcome_model import fit_mu_models, predict_mu_values
from segmentation import select_dast_M_via_dams

DATASET_LOADERS = {
    "hillstrom": load_hillstrom,
    "criteo": load_criteo,
}

DEFAULT_SEED = 42
DEFAULT_SAMPLE_FRAC = 1.0
DEFAULT_M_CANDIDATES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
DEFAULT_N_FOLDS_DAMS = 5


def _parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return list(default)
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Empty list: {raw!r}")
    return vals


def build_gamma_matrix(X, D, y, mu_models) -> np.ndarray:
    """Same DR Gamma construction as prepare_pilot_impl (no pilot/impl split)."""
    X = np.asarray(X)
    D = np.asarray(D).astype(int)
    y = np.asarray(y, dtype=float)
    K = int(max(D.max(), max(mu_models.keys()))) + 1
    actions = np.arange(K, dtype=int)
    Gamma = np.zeros((X.shape[0], K), dtype=float)
    for a in actions:
        if a not in mu_models:
            raise ValueError(f"mu_models missing action {a}")
        mask_a = D == a
        e_a = max(float(mask_a.mean()), 1e-6)
        mu_a_hat = predict_mu_values(mu_models[a], X)
        Gamma[:, a] = mu_a_hat + (mask_a.astype(float) / e_a) * (y - mu_a_hat)
    return Gamma


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tune DAST min_leaf_size via DAMS CV under frozen μ; "
            "update dast block in hparams/{dataset}/{target}.json"
        )
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS))
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--mu_model_type",
        required=True,
        help="OPE μ type whose frozen hparams are used to build Gamma",
    )
    parser.add_argument("--value_type_dast", default="hybrid")
    parser.add_argument("--value_type_dams", default="hybrid")
    parser.add_argument(
        "--action_method",
        required=True,
        choices=["diff_in_means", "gamma", "logistic"],
    )
    parser.add_argument("--sample_frac", type=float, default=DEFAULT_SAMPLE_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_folds_dams", type=int, default=DEFAULT_N_FOLDS_DAMS)
    parser.add_argument(
        "--M_candidates",
        type=str,
        default=None,
        help=f"Comma-separated M grid (default: {DEFAULT_M_CANDIDATES})",
    )
    parser.add_argument(
        "--leaf_grid",
        type=str,
        default=None,
        help=f"Comma-separated min_leaf_size grid (default: {list(DEFAULT_LEAF_GRID)})",
    )
    parser.add_argument(
        "--treatment_cost",
        type=float,
        default=0.0,
        help="Per-unit implementation cost c for action selection and DAST/DAMS (default: 0)",
    )
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    if args.treatment_cost < 0:
        parser.error("--treatment_cost must be >= 0")

    out_path = Path(
        args.out_json
        if args.out_json
        else default_mu_hparams_path(args.dataset, args.target)
    )
    existing = load_hparams_payload(out_path)
    if existing is None:
        raise SystemExit(
            f"[finetune_leaf] missing μ hparams file: {out_path.resolve()}\n"
            f"  Run finetune_mu.py first for {args.dataset}/{args.target}."
        )
    if not _has_type(existing, args.mu_model_type):
        raise SystemExit(
            f"[finetune_leaf] hparams file has no models[{args.mu_model_type!r}].\n"
            f"  Re-run finetune_mu.py."
        )
    cfg_key = dast_config_key(
        mu_model_type=args.mu_model_type,
        action_method=args.action_method,
        value_type_dast=args.value_type_dast,
        value_type_dams=args.value_type_dams,
        treatment_cost=args.treatment_cost,
    )
    if get_dast_entry(
        existing.get("dast"),
        mu_model_type=args.mu_model_type,
        action_method=args.action_method,
        value_type_dast=args.value_type_dast,
        value_type_dams=args.value_type_dams,
        treatment_cost=args.treatment_cost,
        fallback_to_cost_zero=False,
    ) is not None:
        print(
            f"[finetune_leaf] existing dast[{cfg_key!r}] will be overwritten "
            f"(other configs kept)"
        )

    M_candidates = _parse_int_list(args.M_candidates, DEFAULT_M_CANDIDATES)
    leaf_grid = _parse_int_list(args.leaf_grid, list(DEFAULT_LEAF_GRID))
    ope_params = _params_for_type(existing, args.mu_model_type)

    loader = DATASET_LOADERS[args.dataset]
    X, y, D = loader(
        sample_frac=args.sample_frac, seed=args.seed, target_col=args.target
    )
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    D = np.asarray(D).astype(int)
    print(
        f"[finetune_leaf] n={len(y)} sample_frac={args.sample_frac} "
        f"mu={args.mu_model_type} leaf_grid={leaf_grid}"
    )

    print("[finetune_leaf] fitting frozen OPE μ on full cohort …")
    mu_models = fit_mu_models(
        X,
        D,
        y,
        mu_model_type=args.mu_model_type,
        random_state=args.seed,
        mu_hparams=ope_params,
    )
    Gamma = build_gamma_matrix(X, D, y, mu_models)
    print(f"[finetune_leaf] Gamma shape={Gamma.shape}")

    results = []
    best_leaf = None
    best_score = -np.inf
    best_M = None
    protocol = {
        "dataset": args.dataset,
        "target": args.target,
        "sample_frac": float(args.sample_frac),
        "n_folds_dams": int(args.n_folds_dams),
        "M_candidates": list(M_candidates),
        "leaf_grid": list(leaf_grid),
        "value_type_dast": args.value_type_dast,
        "value_type_dams": args.value_type_dams,
        "action_method": args.action_method,
        "mu_model_type": args.mu_model_type,
        "treatment_cost": float(args.treatment_cost),
        "seed": int(args.seed),
    }

    def _checkpoint(*, status: str) -> None:
        nonlocal existing
        dast_entry = {
            "min_leaf_size": best_leaf,
            "best_M": best_M,
            "cv_score": None if best_leaf is None else best_score,
            "grid_results": list(results),
            "status": status,
            "protocol": dict(protocol),
        }
        dast_store = upsert_dast_entry(
            existing.get("dast"),
            dast_entry,
            mu_model_type=args.mu_model_type,
            action_method=args.action_method,
            value_type_dast=args.value_type_dast,
            value_type_dams=args.value_type_dams,
            treatment_cost=args.treatment_cost,
        )
        payload = merge_hparams_payload(existing, {"dast": dast_store})
        saved = save_hparams_payload(out_path, payload)
        existing = payload
        tqdm.write(f"[finetune_leaf] checkpoint ({status}) → {saved}")

    leaf_bar = tqdm(leaf_grid, desc="leaf grid")
    for L in leaf_bar:
        leaf_bar.set_postfix(L=L, best=best_leaf)
        tqdm.write("\n" + "=" * 60)
        tqdm.write(f"min_leaf_size={L}")
        tqdm.write("=" * 60)
        M_star, score, _H = select_dast_M_via_dams(
            X,
            D,
            y,
            Gamma,
            M_candidates=M_candidates,
            min_leaf_size=int(L),
            value_type_dast=args.value_type_dast,
            value_type_dams=args.value_type_dams,
            action_method=args.action_method,
            n_folds=args.n_folds_dams,
            cv_random_state=args.seed,
            treatment_cost=args.treatment_cost,
        )
        results.append(
            {
                "min_leaf_size": int(L),
                "best_M": int(M_star),
                "cv_score": float(score),
            }
        )
        # Maximize DAMS score; ties → smaller leaf (more regularization).
        if (score > best_score) or (
            np.isclose(score, best_score) and (best_leaf is None or L < best_leaf)
        ):
            best_score = float(score)
            best_leaf = int(L)
            best_M = int(M_star)
        leaf_bar.set_postfix(L=L, best=best_leaf, DAMS=f"{best_score:.4g}")
        _checkpoint(status="in_progress")

    _checkpoint(status="complete")
    print(
        f"\n[finetune_leaf] selected min_leaf_size={best_leaf} "
        f"(best_M={best_M}, DAMS={best_score:.6f})"
    )
    print(f"[finetune_leaf] wrote dast[{cfg_key!r}] → {out_path.resolve()}")


if __name__ == "__main__":
    main()
