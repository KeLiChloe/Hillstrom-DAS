# outcome_model.py

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from lightgbm import LGBMRegressor, LGBMClassifier

CLASSIFIER_MU_TYPES = frozenset({"logistic", "mlp_clf", "lightgbm_clf"})

# Default meta-learner μ head (t/s/x/dr). Override via --meta_learner_mu_model_type
# or META_LEARNER_MU_MODEL_TYPE in run_pilot_frac.sh / run_sample_frac.sh.
# Choices: linear, mlp_reg, lightgbm_reg, logistic, mlp_clf, lightgbm_clf
META_LEARNER_MU_MODEL_TYPE = "lightgbm_reg"


def is_classifier_mu_type(mu_model_type: str) -> bool:
    return mu_model_type in CLASSIFIER_MU_TYPES


def tau_model_type_from_mu(mu_model_type: str) -> str:
    """
    Map classifier mu types to regression analogues for continuous tau/CATE heads
    (DR-learner second stage, X-learner effect models).
    """
    return {
        "logistic": "linear",
        "mlp_clf": "mlp_reg",
        "lightgbm_clf": "lightgbm_reg",
    }.get(mu_model_type, mu_model_type)


def _coerce_hparams(params: dict | None) -> dict:
    """JSON may store tuples as lists (e.g. hidden_layer_sizes)."""
    if not params:
        return {}
    out = dict(params)
    hls = out.get("hidden_layer_sizes")
    if isinstance(hls, list):
        out["hidden_layer_sizes"] = tuple(hls)
    return out


def make_mu_model(mu_model_type: str, *, random_state: int = 42, y=None, params: dict | None = None):
    """
    Shared sklearn estimator factory for all mu heads (OPE + meta-learners).

    Pass y when building lightgbm_clf (for scale_pos_weight).
    Optional ``params`` overrides constructor defaults (from finetune JSON).
    ``random_state`` always comes from the caller (per-sim seed), not from params.
    """
    p = _coerce_hparams(params)

    if mu_model_type == "linear":
        # LinearRegression has no useful tunable params in our grid.
        return LinearRegression()

    if mu_model_type == "mlp_reg":
        kwargs = dict(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=5000,
            early_stopping=True,
            random_state=random_state,
        )
        kwargs.update(p)
        kwargs["random_state"] = random_state
        return MLPRegressor(**kwargs)

    if mu_model_type == "lightgbm_reg":
        kwargs = dict(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            verbosity=-1,
            random_state=random_state,
        )
        kwargs.update(p)
        kwargs["random_state"] = random_state
        kwargs.setdefault("verbosity", -1)
        return LGBMRegressor(**kwargs)

    if mu_model_type == "logistic":
        kwargs = dict(max_iter=500, random_state=random_state)
        kwargs.update(p)
        kwargs["random_state"] = random_state
        return LogisticRegression(**kwargs)

    if mu_model_type == "mlp_clf":
        kwargs = dict(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=5000,
            early_stopping=True,
            random_state=random_state,
        )
        kwargs.update(p)
        kwargs["random_state"] = random_state
        return MLPClassifier(**kwargs)

    if mu_model_type == "lightgbm_clf":
        if y is None:
            raise ValueError("lightgbm_clf requires y to set scale_pos_weight.")
        y = np.asarray(y)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(f"y is degenerate (n_pos={n_pos}, n_neg={n_neg})")
        pos_weight = n_neg / n_pos
        kwargs = dict(
            objective="binary",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            scale_pos_weight=pos_weight,
            verbosity=-1,
            random_state=random_state,
        )
        kwargs.update(p)
        kwargs["random_state"] = random_state
        kwargs.setdefault("verbosity", -1)
        # Always recompute scale_pos_weight from the fit slice unless explicitly set.
        if "scale_pos_weight" not in p:
            kwargs["scale_pos_weight"] = pos_weight
        return LGBMClassifier(**kwargs)

    raise ValueError(f"Unknown mu_model_type: {mu_model_type}")


def predict_mu_values(model, X) -> np.ndarray:
    """Return mu(x)=E[Y|X]: regressor predict or classifier P(y=1|X)."""
    X = np.asarray(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(float)
    return model.predict(X).astype(float)


def _safe_fit(model, X, y, min_pos=1):
    """
    - y 只有一个取值 => 报错
    - 若是 {0,1} 二分类 => 正类数 < min_pos 报错（默认 1：至少要有一个正例）
    """
    y = np.asarray(y)
    uniq = np.unique(y)
    if uniq.size <= 1:
        raise ValueError(f"y has only one unique value: {uniq[0]}")

    if set(uniq.tolist()).issubset({0, 1}):
        n_pos = int((y == 1).sum())
        if n_pos < int(min_pos):
            raise ValueError(f"too few positives: n_pos={n_pos} < {min_pos}")

    model.fit(X, y)
    return model


def fit_mu_models(X, D, y, mu_model_type, random_state=42, mu_hparams=None):
    """
    对每个 action a 拟合 μ_a(x) = E[Y|X,D=a].
    Classifiers return P(y=1|x); regressors return E[Y|x].
    """
    X = np.asarray(X)
    D = np.asarray(D)
    y = np.asarray(y)

    actions = np.unique(D)
    mu_models = {}

    for a in actions:
        mask_a = (D == a)
        Xa, ya = X[mask_a], y[mask_a]
        if Xa.shape[0] == 0:
            raise ValueError(f"No samples for action {a}")

        model = make_mu_model(
            mu_model_type,
            random_state=random_state,
            y=ya if mu_model_type == "lightgbm_clf" else None,
            params=mu_hparams,
        )
        model = _safe_fit(model, Xa, ya, min_pos=1)
        mu_models[int(a)] = model

    return mu_models
