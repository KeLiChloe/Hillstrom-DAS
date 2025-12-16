# dr_learner.py
"""
DR-learner (multi-action, K-action) for your Hillstrom pipeline.

Key idea:
  - You already computed Gamma_pilot: shape (N, K)
      Gamma_{i,a} is a DR pseudo-outcome for E[Y(a) | X_i].
  - DR-learner fits a supervised model per action:
      f_a(x) ≈ E[ Gamma_{i,a} | X=x ]
  - Policy: pi(x) = argmax_a f_a(x)

Important:
  - This module ONLY learns the policy (via Gamma labels).
  - Offline evaluation should still use your unified evaluator:
        evaluate_policy_dual_dr(..., mu_pilot_models, action_identity, log_y=log_y)
    to keep ALL comparators on the same OPE.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False


# =========================================================
# Models
# =========================================================

def _make_regressor(model_type: str):
    """
    Return a fresh regressor instance.

    model_type:
      - "ridge"
      - "mlp"
      - "lgbm" (requires lightgbm)
    """
    model_type = str(model_type).lower()

    if model_type == "ridge":
        # Strong baseline, stable, low variance.
        return Ridge(alpha=1.0, random_state=0)

    if model_type == "mlp":
        # Similar spirit to your MLPRegressor outcome model.
        # Keep it modest to avoid overfitting Gamma noise.
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            random_state=0,
        )

    if model_type == "lgbm":
        if not _HAS_LGBM:
            raise ImportError("model_type='lgbm' requires lightgbm to be installed.")
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=0,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


# =========================================================
# Fit DR-learner per-action models on Gamma labels
# =========================================================

def fit_dr_learner_models(
    X_pilot: np.ndarray,
    Gamma_pilot: np.ndarray,
    model_type: str,
    actions: np.ndarray | None = None,
):
    """
    Fit f_a(x) ~ Gamma_{.,a} for each action a.

    Parameters
    ----------
    X_pilot : (N, d)
    Gamma_pilot : (N, K)
    model_type : {"ridge","mlp","lgbm"}
    actions : optional array of actions to fit (subset of 0..K-1).
              Default fits all actions 0..K-1.

    Returns
    -------
    dr_models : dict[int, regressor]
        dr_models[a] predicts E[Gamma_a | X]
    """
    X_pilot = np.asarray(X_pilot)
    Gamma_pilot = np.asarray(Gamma_pilot, dtype=float)

    if Gamma_pilot.ndim != 2:
        raise ValueError("Gamma_pilot must be 2D: shape (N, K)")
    N, K = Gamma_pilot.shape
    if X_pilot.shape[0] != N:
        raise ValueError("X_pilot and Gamma_pilot must have the same N")

    if actions is None:
        actions = np.arange(K, dtype=int)
    else:
        actions = np.asarray(actions, dtype=int)

    dr_models = {}
    for a in actions:
        if a < 0 or a >= K:
            continue
        y_a = Gamma_pilot[:, a]

        # If Gamma is degenerate for some action (rare but possible), fail loudly.
        if np.unique(y_a).size <= 1:
            raise ValueError(
                f"DR-learner: Gamma_pilot[:, {a}] is degenerate (single unique value). "
                "This usually means that action has too few samples or identical outcomes."
            )

        model = _make_regressor(model_type)
        model.fit(X_pilot, y_a)
        dr_models[int(a)] = model

    return dr_models


def predict_dr_values(
    dr_models: dict[int, object],
    X: np.ndarray,
    K: int,
):
    """
    Predict f_a(x) for all actions a and all samples.

    Returns
    -------
    V_hat : (n, K) where V_hat[i,a] = f_a(X_i)
    """
    X = np.asarray(X)
    n = X.shape[0]
    V_hat = np.zeros((n, K), dtype=float)

    for a, model in dr_models.items():
        a = int(a)
        if 0 <= a < K:
            V_hat[:, a] = model.predict(X)

    return V_hat


def dr_learner_policy(
    dr_models: dict[int, object],
    X: np.ndarray,
    K: int,
):
    """
    Return individual-level actions a_hat(x) = argmax_a f_a(x).

    Returns
    -------
    a_hat : (n,) int
    """
    V_hat = predict_dr_values(dr_models, X, K)  # (n,K)
    return np.argmax(V_hat, axis=1).astype(int)
