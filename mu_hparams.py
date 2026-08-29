"""Canonical on-disk hyperparameters (tuned once per dataset/target)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Always relative to repo root (this file's directory), not process cwd.
_REPO_ROOT = Path(__file__).resolve().parent
HPARAMS_ROOT = _REPO_ROOT / "hparams"

# All μ factory types we support; finetune writes one entry per type.
ALL_MU_MODEL_TYPES = (
    "linear",
    "mlp_reg",
    "lightgbm_reg",
    "logistic",
    "mlp_clf",
    "lightgbm_clf",
)

DEFAULT_LEAF_GRID = (5, 10, 20, 50)
DAST_PROTOCOL_KEYS = (
    "value_type_dast",
    "value_type_dams",
    "action_method",
    "mu_model_type",
    "treatment_cost",
)


def normalize_treatment_cost(value: Any) -> float:
    if value is None:
        return 0.0
    cost = float(value)
    if cost < 0:
        raise ValueError(f"treatment_cost must be >= 0, got {cost}")
    return cost


def dast_config_key(
    *,
    mu_model_type: str,
    action_method: str,
    value_type_dast: str,
    value_type_dams: str,
    treatment_cost: float = 0.0,
) -> str:
    """Stable key so different leaf protocols coexist in one JSON."""
    base = "|".join(
        [
            str(mu_model_type),
            str(action_method),
            str(value_type_dast),
            str(value_type_dams),
        ]
    )
    cost = normalize_treatment_cost(treatment_cost)
    if cost == 0.0:
        return base
    return f"{base}|cost={cost:g}"


def _is_legacy_flat_dast(dast: dict) -> bool:
    """Old format: single block with top-level min_leaf_size (+ protocol)."""
    return isinstance(dast, dict) and "min_leaf_size" in dast


def normalize_dast_store(dast: dict | None) -> dict[str, dict]:
    """
    Return keyed store: {config_key: entry}.

    Migrates legacy flat dast (one global block) into a single keyed entry when
    protocol.mu_model_type / method fields are present.
    """
    if not isinstance(dast, dict) or not dast:
        return {}

    if not _is_legacy_flat_dast(dast):
        # Already keyed: keep only dict entries that look like leaf results.
        out: dict[str, dict] = {}
        for key, entry in dast.items():
            if isinstance(entry, dict) and "min_leaf_size" in entry:
                out[str(key)] = entry
        return out

    protocol = dast.get("protocol") or {}
    try:
        key = dast_config_key(
            mu_model_type=str(protocol["mu_model_type"]),
            action_method=str(protocol["action_method"]),
            value_type_dast=str(protocol["value_type_dast"]),
            value_type_dams=str(protocol["value_type_dams"]),
        )
    except KeyError:
        # Incomplete legacy protocol — keep under a placeholder so data is not lost.
        key = "_legacy_unkeyed"
    return {key: dict(dast)}


def upsert_dast_entry(
    existing_dast: dict | None,
    entry: dict,
    *,
    mu_model_type: str,
    action_method: str,
    value_type_dast: str,
    value_type_dams: str,
    treatment_cost: float = 0.0,
) -> dict[str, dict]:
    """Insert/overwrite one config entry; preserve other configs."""
    store = normalize_dast_store(existing_dast)
    key = dast_config_key(
        mu_model_type=mu_model_type,
        action_method=action_method,
        value_type_dast=value_type_dast,
        value_type_dams=value_type_dams,
        treatment_cost=treatment_cost,
    )
    store[key] = entry
    return store


def get_dast_entry(
    dast: dict | None,
    *,
    mu_model_type: str,
    action_method: str,
    value_type_dast: str,
    value_type_dams: str,
    treatment_cost: float = 0.0,
    fallback_to_cost_zero: bool = True,
) -> dict | None:
    """
    Look up a leaf-tuning entry.

    When fallback_to_cost_zero is True, treatment_cost > 0 with no cost-specific
    entry reuses the cost=0 min_leaf_size (same grid for all cost sweeps).
    """
    store = normalize_dast_store(dast)
    key = dast_config_key(
        mu_model_type=mu_model_type,
        action_method=action_method,
        value_type_dast=value_type_dast,
        value_type_dams=value_type_dams,
        treatment_cost=treatment_cost,
    )
    entry = store.get(key)
    if isinstance(entry, dict):
        return entry

    if not fallback_to_cost_zero:
        return None

    cost = normalize_treatment_cost(treatment_cost)
    if cost != 0.0:
        fallback_key = dast_config_key(
            mu_model_type=mu_model_type,
            action_method=action_method,
            value_type_dast=value_type_dast,
            value_type_dams=value_type_dams,
            treatment_cost=0.0,
        )
        entry = store.get(fallback_key)
        if isinstance(entry, dict):
            return entry
    return None


def default_mu_hparams_path(dataset: str, target: str) -> Path:
    """hparams/{dataset}/{target}.json — μ models + optional dast block."""
    return HPARAMS_ROOT / dataset / f"{target}.json"


def load_hparams_payload(path: str | Path | None) -> dict | None:
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Backwards-compatible alias
load_mu_hparams_payload = load_hparams_payload


def save_hparams_payload(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path.resolve()


def merge_hparams_payload(existing: dict | None, updates: dict) -> dict:
    """Shallow-merge top-level keys; nested dicts under known keys are replaced by update."""
    out: dict[str, Any] = dict(existing or {})
    for key, value in updates.items():
        out[key] = value
    return out


def _params_for_type(payload: dict, mu_model_type: str) -> dict | None:
    """
    Support current format (models[type].params) and legacy
    ({ope: {params}, meta: {params}}) for one-off migration.
    Empty params dict → None (factory defaults for that type, e.g. linear).
    """
    models = payload.get("models")
    if isinstance(models, dict) and mu_model_type in models:
        params = (models[mu_model_type] or {}).get("params")
        if isinstance(params, dict) and not params:
            return None
        return params if isinstance(params, dict) else None

    for key in ("ope", "meta"):
        block = payload.get(key) or {}
        if block.get("mu_model_type") == mu_model_type:
            params = block.get("params")
            if isinstance(params, dict) and not params:
                return None
            return params if isinstance(params, dict) else None
    return None


def _has_type(payload: dict, mu_model_type: str) -> bool:
    models = payload.get("models")
    if isinstance(models, dict) and mu_model_type in models:
        return True
    for key in ("ope", "meta"):
        if (payload.get(key) or {}).get("mu_model_type") == mu_model_type:
            return True
    return False


def _require_models(
    payload: dict,
    path: Path,
    mu_model_type: str,
    meta_learner_mu_model_type: str,
    *,
    dataset: str,
    target: str,
) -> tuple[dict | None, dict | None]:
    missing = [
        mtype
        for mtype in (mu_model_type, meta_learner_mu_model_type)
        if not _has_type(payload, mtype)
    ]
    if missing:
        available = sorted((payload.get("models") or {}).keys()) or [
            (payload.get("ope") or {}).get("mu_model_type"),
            (payload.get("meta") or {}).get("mu_model_type"),
        ]
        raise KeyError(
            f"μ hparams file {path.resolve()} missing type(s) {missing}; "
            f"available={available}. Re-run finetune_mu.py for {dataset}/{target}."
        )
    return (
        _params_for_type(payload, mu_model_type),
        _params_for_type(payload, meta_learner_mu_model_type),
    )


def _require_dast_leaf(
    payload: dict,
    path: Path,
    *,
    dataset: str,
    target: str,
    value_type_dast: str,
    value_type_dams: str,
    action_method: str,
    mu_model_type: str,
    treatment_cost: float = 0.0,
) -> int:
    entry = get_dast_entry(
        payload.get("dast"),
        mu_model_type=mu_model_type,
        action_method=action_method,
        value_type_dast=value_type_dast,
        value_type_dams=value_type_dams,
        treatment_cost=treatment_cost,
    )
    lookup_key = dast_config_key(
        mu_model_type=mu_model_type,
        action_method=action_method,
        value_type_dast=value_type_dast,
        value_type_dams=value_type_dams,
        treatment_cost=treatment_cost,
    )
    resolved_key = lookup_key
    if entry is not None and normalize_treatment_cost(treatment_cost) != 0.0:
        store = normalize_dast_store(payload.get("dast"))
        if lookup_key not in store:
            resolved_key = dast_config_key(
                mu_model_type=mu_model_type,
                action_method=action_method,
                value_type_dast=value_type_dast,
                value_type_dams=value_type_dams,
                treatment_cost=0.0,
            )
    key = resolved_key
    if entry is None or "min_leaf_size" not in entry:
        available = sorted(normalize_dast_store(payload.get("dast")).keys())
        raise KeyError(
            f"hparams file {path.resolve()} missing dast entry for {key!r}.\n"
            f"  available={available}\n"
            f"  Run: python finetune_leaf.py --dataset {dataset} --target {target} "
            f"--mu_model_type {mu_model_type} --action_method {action_method} ..."
        )
    if entry.get("min_leaf_size") is None:
        raise KeyError(
            f"dast[{key!r}] in {path.resolve()} has no min_leaf_size yet "
            f"(leaf finetune interrupted before first grid point)."
        )
    if entry.get("status") == "in_progress":
        raise ValueError(
            f"dast[{key!r}] in {path.resolve()} is still status=in_progress.\n"
            f"  Re-run finetune_leaf.py to completion before starting experiments."
        )

    # Entry is already keyed by protocol fields; still verify embedded protocol
    # matches (guards against hand-edited JSON).
    protocol = entry.get("protocol") or {}
    expected = {
        "value_type_dast": value_type_dast,
        "value_type_dams": value_type_dams,
        "action_method": action_method,
        "mu_model_type": mu_model_type,
    }
    mismatches = []
    for k, want in expected.items():
        got = protocol.get(k)
        if got is None:
            mismatches.append(f"{k}: missing in dast[].protocol (expected {want!r})")
        elif str(got) != str(want):
            mismatches.append(f"{k}: protocol={got!r}, run={want!r}")
    if mismatches:
        raise ValueError(
            f"dast[{key!r}] protocol in {path.resolve()} does not match this run:\n  "
            + "\n  ".join(mismatches)
            + "\n  Re-run finetune_leaf.py with the same method settings, or change the run."
        )

    return int(entry["min_leaf_size"])


def resolve_mu_hparams(
    *,
    dataset: str,
    target: str,
    mu_model_type: str,
    meta_learner_mu_model_type: str,
    explicit_path: str | None = None,
) -> tuple[str, dict | None, dict | None]:
    """μ-only resolve (legacy). Prefer resolve_hparams for experiments."""
    path, ope, meta, _leaf = resolve_hparams(
        dataset=dataset,
        target=target,
        mu_model_type=mu_model_type,
        meta_learner_mu_model_type=meta_learner_mu_model_type,
        value_type_dast="hybrid",
        value_type_dams="hybrid",
        action_method="diff_in_means",
        explicit_path=explicit_path,
        require_leaf=False,
    )
    return path, ope, meta


def resolve_hparams(
    *,
    dataset: str,
    target: str,
    mu_model_type: str,
    meta_learner_mu_model_type: str,
    value_type_dast: str,
    value_type_dams: str,
    action_method: str,
    explicit_path: str | None = None,
    require_leaf: bool = True,
    treatment_cost: float = 0.0,
) -> tuple[str, dict | None, dict | None, int | None]:
    """
    Load permanent hparams/{dataset}/{target}.json.

    Returns (path, ope_params, meta_params, min_leaf_size).
    Raises if file/models missing; if require_leaf, also requires dast + protocol match.
    """
    path = Path(explicit_path) if explicit_path else default_mu_hparams_path(dataset, target)
    payload = load_hparams_payload(path)
    if payload is None:
        raise FileNotFoundError(
            f"Required hparams file not found: {path.resolve()}\n"
            f"  Run: python finetune_mu.py --dataset {dataset} --target {target}"
        )

    ope, meta = _require_models(
        payload,
        path,
        mu_model_type,
        meta_learner_mu_model_type,
        dataset=dataset,
        target=target,
    )

    leaf: int | None = None
    if require_leaf:
        leaf = _require_dast_leaf(
            payload,
            path,
            dataset=dataset,
            target=target,
            value_type_dast=value_type_dast,
            value_type_dams=value_type_dams,
            action_method=action_method,
            mu_model_type=mu_model_type,
            treatment_cost=treatment_cost,
        )

    return str(path.resolve()), ope, meta, leaf
