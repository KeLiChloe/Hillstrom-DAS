"""Shared node/plane fitting for MST and CLR (binary y, logistic)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def _deviance_sum(y, proba, eps: float = 1e-15) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(proba, dtype=float), eps, 1.0 - eps)
    return float(-2.0 * np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def per_sample_loss(y, proba) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(proba, dtype=float), 1e-15, 1.0 - 1e-15)
    return -2.0 * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _max_binary_deviance(n: int) -> float:
    """Penalize degenerate nodes (single-class y or failed fit)."""
    return float(n * np.log(2.0))


def node_impurity(X_design: np.ndarray, y: np.ndarray) -> float:
    """Node/plane impurity for tree splitting (lower is better): logistic deviance."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return 0.0

    if np.unique(y).size <= 1:
        return _max_binary_deviance(n)

    model = LogisticRegression(fit_intercept=False, max_iter=500, solver="lbfgs")
    try:
        model.fit(X_design, y)
        proba = model.predict_proba(X_design)[:, 1]
        loss = _deviance_sum(y, proba)
    except (ValueError, np.linalg.LinAlgError):
        loss = _max_binary_deviance(n)

    return loss + float(np.random.rand() * 1e-9)


def plane_predict_loss(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-sample logistic deviance for CLR cluster assignment."""
    y = np.asarray(y, dtype=float)
    return per_sample_loss(y, model.predict_proba(X)[:, 1])


def fit_plane_model(model, X: np.ndarray, y: np.ndarray) -> bool:
    """Fit one CLR plane model. Returns False if fit was skipped (degenerate y)."""
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return False
    if np.unique(y).size <= 1:
        return False
    try:
        model.fit(X, y)
    except (ValueError, np.linalg.LinAlgError):
        return False
    return True


def _n_model_params(model) -> int:
    if hasattr(model, "coef_"):
        n = int(np.size(model.coef_))
        if hasattr(model, "intercept_") and model.intercept_ is not None:
            n += int(np.size(model.intercept_))
        return n
    return 0


def segmentation_bic(
    X_D: np.ndarray,
    y: np.ndarray,
    cluster_labels: np.ndarray,
    models: list,
) -> float:
    """BIC for piecewise logistic models."""
    y = np.asarray(y, dtype=float)
    n, _ = X_D.shape
    k = len(models)

    logL = 0.0
    p = 0
    for cl_idx in range(k):
        mask = cluster_labels == cl_idx
        if mask.sum() == 0:
            continue
        proba = models[cl_idx].predict_proba(X_D[mask])[:, 1]
        ym = y[mask]
        proba = np.clip(proba, 1e-15, 1.0 - 1e-15)
        logL += float(np.sum(ym * np.log(proba) + (1.0 - ym) * np.log(1.0 - proba)))
        p += _n_model_params(models[cl_idx])
    return -2.0 * logL + p * np.log(n)
