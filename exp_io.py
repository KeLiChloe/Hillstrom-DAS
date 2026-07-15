"""Small helpers for experiment artifact I/O (no heavy ML / R imports)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def json_safe(obj):
    """Convert params values to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    return obj


def write_run_params_json(out_path: str | os.PathLike, params: dict) -> str:
    """
    Write experiment params to <outdir>/run_params.json next to the pkl.

    Returns the JSON path written.
    """
    out_dir = Path(out_path).expanduser().resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "run_params.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(dict(params)), f, indent=2, sort_keys=True)
        f.write("\n")
    return str(json_path)
