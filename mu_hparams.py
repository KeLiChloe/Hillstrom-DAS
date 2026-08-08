"""Canonical on-disk μ hyperparameters (tuned once per dataset/target)."""

from __future__ import annotations

import json
from pathlib import Path

# Repo-local permanent store (not under exp_* sweep outdirs).
HPARAMS_ROOT = Path("hparams")

# All μ factory types we support; finetune writes one entry per type.
ALL_MU_MODEL_TYPES = (
    "linear",
    "mlp_reg",
    "lightgbm_reg",
    "logistic",
    "mlp_clf",
    "lightgbm_clf",
)


def default_mu_hparams_path(dataset: str, target: str) -> Path:
    """hparams/{dataset}/{target}.json — one file for all μ model types."""
    return (HPARAMS_ROOT / dataset / f"{target}.json").resolve()


def _params_for_type(payload: dict, mu_model_type: str) -> dict | None:
    """
    Support current format (models[type].params) and legacy
    ({ope: {params}, meta: {params}}) for one-off migration.
    """
    models = payload.get("models")
    if isinstance(models, dict) and mu_model_type in models:
        params = (models[mu_model_type] or {}).get("params")
        if isinstance(params, dict) and not params:
            return None
        return params if isinstance(params, dict) else None

    # legacy single-pair files
    for key in ("ope", "meta"):
        block = payload.get(key) or {}
        if block.get("mu_model_type") == mu_model_type:
            params = block.get("params")
            if isinstance(params, dict) and not params:
                return None
            return params if isinstance(params, dict) else None
    return None


def load_mu_hparams_payload(path: str | Path | None) -> dict | None:
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_mu_hparams(
    *,
    dataset: str,
    target: str,
    mu_model_type: str,
    meta_learner_mu_model_type: str,
    explicit_path: str | None = None,
) -> tuple[str | None, dict | None, dict | None]:
    """
    Prefer explicit_path; else canonical hparams/{dataset}/{target}.json.
    Returns (path_used_or_None, ope_params, meta_params) for the requested types.
    """
    path = Path(explicit_path) if explicit_path else default_mu_hparams_path(dataset, target)
    payload = load_mu_hparams_payload(path)
    if payload is None:
        if explicit_path:
            raise FileNotFoundError(f"μ hparams file not found: {explicit_path}")
        return None, None, None

    ope = _params_for_type(payload, mu_model_type)
    meta = _params_for_type(payload, meta_learner_mu_model_type)
    return str(path.resolve()), ope, meta
