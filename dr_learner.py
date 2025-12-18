#dr_learner.py 
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

from outcome_model import predict_mu


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


def _as_1d(x, name: str):
    x = np.asarray(x).reshape(-1)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return x


def _clip_prob(p, eps=1e-6):
    return np.clip(p, eps, 1.0 - eps)



def fit_dr_learner_binary(
    X,
    D,
    y,
    mu_models,          # dict {0: m0_model, 1: m1_model}
    e: float,           # known RCT propensity P(D=1)
    model_type: str,
):
    X = np.asarray(X)
    D = _as_1d(D, "D").astype(int)
    y = _as_1d(y, "y").astype(float)

    if set(np.unique(D)) - {0, 1}:
        raise ValueError("Binary version expects D in {0,1}.")
    if 0 not in mu_models or 1 not in mu_models:
        raise ValueError("mu_models must contain keys 0 and 1.")

    e = float(_clip_prob(e))

    m0 = predict_mu(mu_models[0], X)
    m1 = predict_mu(mu_models[1], X)
    mD = D * m1 + (1 - D) * m0

    pseudo = ((D - e) / (e * (1 - e))) * (y - mD) + (m1 - m0)

    tau_model = _make_regressor(model_type)
    tau_model.fit(X, pseudo)

    return {
        "type": "binary",
        "tau_model": tau_model,
        "mu_models": mu_models,
        "e": e,
    }



def dr_learner_policy_binary(dr_model, X):
    """
    Returns action in {0,1} maximizing estimated outcome.
    Uses mu0(x) and tau(x): mu1(x) = mu0(x) + tau(x)
    """
    X = np.asarray(X)
    mu0 = predict_mu(dr_model["mu_models"][0], X)
    tau = dr_model["tau_model"].predict(X)
    mu1 = mu0 + tau
    a_hat = (mu1 > mu0).astype(int)
    mu_hat = np.vstack([mu0, mu1]).T
    return a_hat, mu_hat



def fit_dr_learner_k_armed(
    X,
    D,
    y,
    mu_models,              # dict[a] -> m_a_model for all a in {0..K-1}
    K: int,
    pi,                     # known propensities: array-like length K (sum ~ 1)
    baseline: int,
    model_type: str,
):
    X = np.asarray(X)
    D = _as_1d(D, "D").astype(int)
    y = _as_1d(y, "y").astype(float)

    if K < 2:
        raise ValueError("K must be >= 2.")
    if baseline < 0 or baseline >= K:
        raise ValueError("baseline out of range.")
    if any(a not in mu_models for a in range(K)):
        missing = [a for a in range(K) if a not in mu_models]
        raise ValueError(f"mu_models missing actions: {missing}")

    pi = _clip_prob(np.asarray(pi, float).reshape(-1))
    if pi.shape[0] != K:
        raise ValueError("pi must have length K.")

    # Precompute outcome predictions for all arms on X
    mu_hat = {a: predict_mu(mu_models[a], X) for a in range(K)}
    mub = mu_hat[baseline]
    pib = float(pi[baseline])

    tau_models = {}
    for a in range(K):
        if a == baseline:
            continue

        mua = mu_hat[a]
        pia = float(pi[a])

        Ia = (D == a).astype(float)
        Ib = (D == baseline).astype(float)

        pseudo = (mua - mub) + (Ia / pia) * (y - mua) - (Ib / pib) * (y - mub)

        model = _make_regressor(model_type)
        model.fit(X, pseudo)
        tau_models[a] = model

    return {
        "type": "k_armed",
        "K": K,
        "baseline": baseline,
        "pi": pi,
        "mu_models": mu_models,     # keep all m_a models
        "tau_models": tau_models,   # dict[a!=baseline] -> model for tau_a(x)
    }




def dr_learner_policy_k_armed(dr_model, X):
    """
    Policy: choose argmax_a \hat m_a(x).
    We form \hat m_baseline(x) then add \hat tau_a(x) to get \hat m_a(x).
    """
    X = np.asarray(X)
    n = X.shape[0]
    K = dr_model["K"]
    baseline = dr_model["baseline"]

    mub = predict_mu(dr_model["mu_models"][baseline], X)

    mu_hat = np.zeros((n, K), float)
    mu_hat[:, baseline] = mub

    for a, tau_model in dr_model["tau_models"].items():
        mu_hat[:, a] = mub + tau_model.predict(X)

    a_hat = np.argmax(mu_hat, axis=1)
    return a_hat, mu_hat
