# dr_learner_true.py
import numpy as np
from sklearn.base import clone
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

from outcome_model import predict_mu


# --------------------------------------------------
# utilities
# --------------------------------------------------

def _make_regressor(model_type: str):
    model_type = model_type.lower()
    if model_type == "ridge":
        return Ridge(alpha=1.0)
    if model_type == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            random_state=0,
        )
    if model_type == "lgbm":
        return LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=0,
        )
    raise ValueError(model_type)


# --------------------------------------------------
# fit
# --------------------------------------------------

def fit_dr_learner_models(
    X_pilot,
    D_pilot,
    y_pilot,
    mu_pilot_models,     # dict[a] -> μ_a(x)
    *,
    K: int,
    baseline: int = 0,
    model_type: str = "lgbm",
):
    """
    True DR-learner (Chernozhukov-style), multi-action one-vs-baseline.
    """

    X_pilot = np.asarray(X_pilot)
    D_pilot = np.asarray(D_pilot).astype(int)
    y_pilot = np.asarray(y_pilot, float)

    n = X_pilot.shape[0]

    # ---- marginal outcome m(x) ----
    # use mixture over actions (RCT => consistent)
    m_hat = np.zeros(n)
    for a, mu_a in mu_pilot_models.items():
        m_hat += predict_mu(mu_a, X_pilot)
    m_hat /= len(mu_pilot_models)

    # ---- known propensities (RCT) ----
    pi = np.array([(D_pilot == a).mean() for a in range(K)])
    pi = np.clip(pi, 1e-6, 1.0)

    tau_models = {}

    for a in range(K):
        if a == baseline:
            continue

        pseudo = (
            ((D_pilot == a).astype(float) - pi[a])
            / (pi[a] * (1 - pi[a]))
            * (y_pilot - m_hat)
        )

        model = _make_regressor(model_type)
        model.fit(X_pilot, pseudo)
        tau_models[a] = model

    return {
        "baseline": baseline,
        "tau_models": tau_models,
        "m_hat_model": mu_pilot_models[baseline],
        "K": K,
    }


# --------------------------------------------------
# predict policy
# --------------------------------------------------

def dr_learner_policy(
    dr_model,
    X,
):
    X = np.asarray(X)
    n = X.shape[0]
    K = dr_model["K"]
    baseline = dr_model["baseline"]

    mu0 = predict_mu(dr_model["m_hat_model"], X)

    mu_hat = np.zeros((n, K))
    mu_hat[:, baseline] = mu0

    for a, tau_model in dr_model["tau_models"].items():
        mu_hat[:, a] = mu0 + tau_model.predict(X)

    a_hat = np.argmax(mu_hat, axis=1)
    return a_hat, mu_hat
