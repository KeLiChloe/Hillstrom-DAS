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
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

from data_utils import load_criteo, load_hillstrom
from mu_hparams import ALL_MU_MODEL_TYPES, default_mu_hparams_path, load_hparams_payload, merge_hparams_payload, save_hparams_payload
from outcome_model import is_classifier_mu_type, make_mu_model

DATASET_LOADERS = {
    "hillstrom": load_hillstrom,
    "criteo": load_criteo,
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
        # KFold requires n_samples >= n_splits.
        if n_a < n_folds:
            continue
        if classifier and len(np.unique(ya)) < 2:
            continue

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + int(a))
        fold_losses = []
        for fold_i, (tr, te) in enumerate(kf.split(Xa)):
            if classifier and (len(np.unique(ya[tr])) < 2 or len(np.unique(ya[te])) < 2):
                continue
            fold_params = dict(params)
            model = make_mu_model(
                mu_model_type,
                random_state=seed + 1000 * int(a) + fold_i,
                y=ya[tr] if mu_model_type == "lightgbm_clf" else None,
                params=fold_params,
            )
            try:
                model.fit(Xa[tr], ya[tr])
            except ValueError:
                # MLP early_stopping's internal stratified val split can fail on
                # rare-event folds; retry once without it, else skip fold.
                if mu_model_type not in ("mlp_reg", "mlp_clf"):
                    continue
                if fold_params.get("early_stopping") is False:
                    continue
                fold_params = {**fold_params, "early_stopping": False}
                model = make_mu_model(
                    mu_model_type,
                    random_state=seed + 1000 * int(a) + fold_i,
                    y=ya[tr] if mu_model_type == "lightgbm_clf" else None,
                    params=fold_params,
                )
                try:
                    model.fit(Xa[tr], ya[tr])
                except ValueError:
                    continue
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
    pbar = tqdm(grid, desc=f"grid:{mu_model_type}", leave=False)
    for i, params in enumerate(pbar):
        score = cv_score_for_params(
            X, D, y, mu_model_type, params, n_folds=n_folds, seed=seed
        )
        printable = {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()
        }
        tqdm.write(f"  [{i + 1}/{len(grid)}] score={score:.6g} params={printable}")
        if score < best_score:
            best_score = score
            best_params = dict(params)
        pbar.set_postfix(best=f"{best_score:.4g}" if np.isfinite(best_score) else "inf")

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
    args = parser.parse_args()

    out_path = Path(
        args.out_json
        if args.out_json
        else default_mu_hparams_path(args.dataset, args.target)
    )
    if out_path.is_file():
        print(f"[finetune] existing {out_path} will be updated (models overwritten; dast preserved if present)")

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

    # Incremental save: keep prior models/dast; overwrite one type at a time.
    existing = load_hparams_payload(out_path) or {}
    models = dict(existing.get("models") or {})
    protocol = {
        "dataset": args.dataset,
        "target": args.target,
        "sample_frac": float(args.sample_frac),
        "tuning_on": (
            "full_sample"
            if float(args.sample_frac) >= 1.0 - 1e-12
            else "subsample"
        ),
        "n_folds": int(args.n_folds),
        "seed": int(args.seed),
        "types_planned": types_to_tune,
        "types_completed": [
            t for t in types_to_tune if t in models
        ],
    }

    type_bar = tqdm(types_to_tune, desc="μ types")
    for i, mtype in enumerate(type_bar):
        type_bar.set_postfix_str(mtype)
        tqdm.write("\n" + "=" * 60)
        tqdm.write(f"[{i + 1}/{len(types_to_tune)}] {mtype}")
        tqdm.write("=" * 60)
        models[mtype] = tune_mu_model(
            X,
            D,
            y,
            mtype,
            n_folds=args.n_folds,
            seed=args.seed + 17 * i,
        )
        protocol["types_completed"] = [
            t for t in types_to_tune if t in models
        ]
        protocol["types_tuned"] = list(protocol["types_completed"])
        payload = merge_hparams_payload(
            existing,
            {"models": models, "protocol": dict(protocol)},
        )
        saved = save_hparams_payload(out_path, payload)
        existing = payload
        block = models[mtype]
        tqdm.write(
            f"[finetune] checkpoint → {saved} "
            f"({mtype}: {block.get('metric')}={block.get('cv_score')})"
        )

    print("\n[finetune] CV summary (lower is better):")
    for mtype in types_to_tune:
        block = models.get(mtype) or {}
        print(
            f"  {mtype:16s}  {block.get('metric', '?'):8s}  "
            f"{block.get('cv_score')}"
        )
    print(f"\n[finetune] finished all types → {out_path.resolve()}")


if __name__ == "__main__":
    main()
