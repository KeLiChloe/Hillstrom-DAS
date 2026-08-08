#!/usr/bin/env python3
"""
One-shot μ hyperparameter finetune (standalone; not part of experiment sweeps).

For each (dataset, target), tune ALL μ model types on the full sample and write
one permanent JSON under hparams/{dataset}/{target}.json. run_sims.py later
picks the entry matching --mu_model_type / --meta_learner_mu_model_type.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold

from data_utils import load_criteo, load_hillstrom, load_lenta
from mu_hparams import ALL_MU_MODEL_TYPES, default_mu_hparams_path
from outcome_model import is_classifier_mu_type, make_mu_model

DATASET_LOADERS = {
    "hillstrom": load_hillstrom,
    "criteo": load_criteo,
    "lenta": load_lenta,
}

N_FOLDS = 5
DEFAULT_SEED = 42
DEFAULT_SAMPLE_FRAC = 1.0


def _param_grid(mu_model_type: str) -> list[dict]:
    """
    Small grids matched to Hillstrom / Criteo-style cohorts in this repo:
      - low dimension (d ≈ 9–12 after encoding)
      - rare binary conversion (~0.3%–1%) or skewed continuous spend
      - per-arm n from ~20k (Hillstrom) to 100k+ (Criteo percent10)

    Prefer shallower / better-regularized μ heads over huge capacity.
    """
    if mu_model_type in ("lightgbm_reg", "lightgbm_clf"):
        keys = ("n_estimators", "learning_rate", "num_leaves", "min_child_samples")
        values = [
            [100, 300],
            [0.03, 0.05],
            [8, 15, 31],
            [20, 50],
        ]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    if mu_model_type in ("mlp_reg", "mlp_clf"):
        keys = ("hidden_layer_sizes", "alpha")
        values = [
            [(32,), (64,), (64, 32)],
            [1e-3, 1e-2],
        ]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    if mu_model_type == "logistic":
        return [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]

    # linear / unknown: keep factory defaults
    return [{}]


def _fold_score(y_true, y_pred, *, classifier: bool) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if classifier:
        y_pred = np.clip(y_pred, 1e-6, 1.0 - 1e-6)
        return float(log_loss(y_true, y_pred, labels=[0.0, 1.0]))
    return float(mean_squared_error(y_true, y_pred))


def cv_score_for_params(
    X: np.ndarray,
    D: np.ndarray,
    y: np.ndarray,
    mu_model_type: str,
    params: dict,
    *,
    n_folds: int,
    seed: int,
) -> float:
    """
    Weighted average of within-action K-fold CV loss (lower is better).
    Weight = number of samples in that action.
    """
    X = np.asarray(X)
    D = np.asarray(D).astype(int)
    y = np.asarray(y, dtype=float)
    classifier = is_classifier_mu_type(mu_model_type)
    actions = np.unique(D)

    total_w = 0.0
    total_loss = 0.0

    for a in actions:
        mask = D == a
        Xa, ya = X[mask], y[mask]
        n_a = int(len(ya))
        if n_a < max(n_folds, 4):
            continue
        if classifier and len(np.unique(ya)) < 2:
            continue

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + int(a))
        fold_losses = []
        for fold_i, (tr, te) in enumerate(kf.split(Xa)):
            if classifier and (len(np.unique(ya[tr])) < 2 or len(np.unique(ya[te])) < 2):
                continue
            model = make_mu_model(
                mu_model_type,
                random_state=seed + 1000 * int(a) + fold_i,
                y=ya[tr] if mu_model_type == "lightgbm_clf" else None,
                params=params,
            )
            model.fit(Xa[tr], ya[tr])
            if classifier:
                pred = model.predict_proba(Xa[te])[:, 1]
            else:
                pred = model.predict(Xa[te])
            fold_losses.append(_fold_score(ya[te], pred, classifier=classifier))

        if not fold_losses:
            continue
        total_loss += float(np.mean(fold_losses)) * n_a
        total_w += n_a

    if total_w <= 0:
        return float("inf")
    return total_loss / total_w


def tune_mu_model(
    X: np.ndarray,
    D: np.ndarray,
    y: np.ndarray,
    mu_model_type: str,
    *,
    n_folds: int,
    seed: int,
) -> dict:
    grid = _param_grid(mu_model_type)
    metric = "log_loss" if is_classifier_mu_type(mu_model_type) else "mse"
    best_params: dict = {}
    best_score = float("inf")

    print(f"[finetune] type={mu_model_type} grid_size={len(grid)} metric={metric}")
    for i, params in enumerate(grid):
        score = cv_score_for_params(
            X, D, y, mu_model_type, params, n_folds=n_folds, seed=seed
        )
        printable = {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()
        }
        print(f"  [{i + 1}/{len(grid)}] score={score:.6g} params={printable}")
        if score < best_score:
            best_score = score
            best_params = dict(params)

    best_params_json = {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()
    }
    return {
        "mu_model_type": mu_model_type,
        "params": best_params_json,
        "cv_score": None if not np.isfinite(best_score) else float(best_score),
        "metric": metric,
    }


def load_tuning_cohort(
    *,
    dataset: str,
    target: str,
    sample_frac: float,
    seed: int,
):
    """Load full available sample; no pilot/impl split."""
    if dataset not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {dataset}")
    X, y, D = DATASET_LOADERS[dataset](
        sample_frac=sample_frac, seed=seed, target_col=target
    )
    return (
        np.asarray(X),
        np.asarray(D).astype(int),
        np.asarray(y, dtype=float),
    )


def _y_is_binary(y: np.ndarray) -> bool:
    u = set(np.unique(y).tolist())
    return u.issubset({0.0, 1.0, 0, 1})


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tune ALL μ model types once per dataset/target; "
            "save permanently under hparams/{dataset}/{target}.json."
        )
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS))
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=DEFAULT_SAMPLE_FRAC,
        help="Fraction of full dataset used for CV (default: 1.0)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Override output path (default: hparams/{dataset}/{target}.json)",
    )
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing permanent hparams file",
    )
    args = parser.parse_args()

    out_path = Path(
        args.out_json
        if args.out_json
        else default_mu_hparams_path(args.dataset, args.target)
    )
    if out_path.is_file() and not args.force:
        raise SystemExit(
            f"[finetune] refuse to overwrite existing {out_path}\n"
            f"  Re-run with --force, or delete the file first."
        )

    X, D, y = load_tuning_cohort(
        dataset=args.dataset,
        target=args.target,
        sample_frac=args.sample_frac,
        seed=args.seed,
    )
    binary_y = _y_is_binary(y)
    types_to_tune = list(ALL_MU_MODEL_TYPES)
    if not binary_y:
        # Classifiers require {0,1} labels (e.g. spend is continuous).
        types_to_tune = [t for t in types_to_tune if not is_classifier_mu_type(t)]
        print(
            f"[finetune] y is not binary → skipping classifier types; "
            f"tuning {types_to_tune}"
        )

    print(
        f"[finetune] n={len(y)} sample_frac={args.sample_frac} "
        f"seed={args.seed} types={types_to_tune}"
    )

    models = {}
    for i, mtype in enumerate(types_to_tune):
        print("\n" + "=" * 60)
        print(f"[{i + 1}/{len(types_to_tune)}] {mtype}")
        print("=" * 60)
        models[mtype] = tune_mu_model(
            X,
            D,
            y,
            mtype,
            n_folds=args.n_folds,
            seed=args.seed + 17 * i,
        )

    payload = {
        "models": models,
        "protocol": {
            "dataset": args.dataset,
            "target": args.target,
            "sample_frac": float(args.sample_frac),
            "tuning_on": "full_sample",
            "n_folds": int(args.n_folds),
            "seed": int(args.seed),
            "types_tuned": types_to_tune,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\n[finetune] wrote permanent hparams → {out_path}")


if __name__ == "__main__":
    main()
