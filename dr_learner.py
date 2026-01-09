import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

from outcome_model import predict_mu


def _make_regressor(model_type: str):
    model_type = model_type.lower()
    if model_type == "ridge":
        return Ridge(alpha=1.0)
    if model_type == "mlp_reg":
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            random_state=0,
        )
    if model_type == "lightgbm_reg":
        return LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=0,
        )
    raise ValueError(model_type)


def _clip_prob(p, eps=1e-6):
    return np.clip(p, eps, 1.0 - eps)


def _fit_mu_models_by_action(X_train, D_train, y_train, K, model_type):
    """Fit mu_a(x)=E[Y|X,D=a] on a training split."""
    mu_models = {}
    for a in range(K):
        mask = (D_train == a)
        if mask.sum() == 0:
            raise ValueError(f"No training samples for action {a} in this fold.")
        m = _make_regressor(model_type)
        m.fit(X_train[mask], y_train[mask])
        mu_models[a] = m
    return mu_models


# =========================


def fit_dr_learner_binary(
    X, D, y,
    *,
    e: float,
    n_folds: int,
    mu_model_type: str,
    tau_model_type: str,
):
    X = np.asarray(X)
    D = np.asarray(D).astype(int).reshape(-1)
    y = np.asarray(y, float).reshape(-1)

    if set(np.unique(D)) - {0, 1}:
        raise ValueError("Binary expects D in {0,1}.")
    n = X.shape[0]
    if D.shape[0] != n or y.shape[0] != n:
        raise ValueError("X, D, y must have same length.")

    e = float(_clip_prob(float(e)))

    kf = KFold(n_splits=n_folds, shuffle=True)

    tau_models = []
    mu_models_full = None  # optional final nuisances for better mu0 at prediction

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, D_tr, y_tr = X[train_idx], D[train_idx], y[train_idx]
        X_te, D_te, y_te = X[test_idx], D[test_idx], y[test_idx]

        # nuisances on TRAIN folds
        mu_models = _fit_mu_models_by_action(X_tr, D_tr, y_tr, K=2, model_type=mu_model_type)

        # pseudo on HELD-OUT fold
        m0 = predict_mu(mu_models[0], X_te)
        m1 = predict_mu(mu_models[1], X_te)
        mD = D_te * m1 + (1 - D_te) * m0
        pseudo = ((D_te - e) / (e * (1 - e))) * (y_te - mD) + (m1 - m0)

        # second-stage model trained on THIS fold's pseudo
        tau_f = _make_regressor(tau_model_type)
        tau_f.fit(X_te, pseudo)
        tau_models.append(tau_f)

    # optional: fit nuisance on all data for better outcome prediction in policy
    mu_models_full = _fit_mu_models_by_action(X, D, y, K=2, model_type=mu_model_type)

    return {
        "type": "binary_crossfit_ensemble",
        "e": e,
        "n_folds": n_folds,
        "mu_models": mu_models_full,
        "tau_models": tau_models,  # list of estimators
    }


def dr_learner_predict_binary(dr_model, X):
    X = np.asarray(X)
    preds = np.column_stack([m.predict(X) for m in dr_model["tau_models"]])
    return preds.mean(axis=1)


def dr_learner_policy_binary(dr_model, X):
    X = np.asarray(X)
    mu0 = predict_mu(dr_model["mu_models"][0], X)
    tau = dr_learner_predict_binary(dr_model, X)
    mu1 = mu0 + tau
    mu_hat = np.vstack([mu0, mu1]).T
    a_hat = np.argmax(mu_hat, axis=1)
    return a_hat, mu_hat



def fit_dr_learner_k_armed(
    X, D, y,
    *,
    K: int,
    pi,
    baseline: int,
    n_folds: int,
    mu_model_type: str,
    tau_model_type: str,
):
    X = np.asarray(X)
    D = np.asarray(D).astype(int).reshape(-1)
    y = np.asarray(y, float).reshape(-1)

    n = X.shape[0]
    if D.shape[0] != n or y.shape[0] != n:
        raise ValueError("X, D, y must have same length.")
    if K < 2:
        raise ValueError("K must be >= 2.")
    if baseline < 0 or baseline >= K:
        raise ValueError("baseline out of range.")
    if np.any((D < 0) | (D >= K)):
        raise ValueError("D contains invalid action ids outside [0, K).")

    pi = np.asarray(pi, float).reshape(-1)
    if pi.shape[0] != K:
        raise ValueError("pi must have length K.")
    pi = _clip_prob(pi)
    pib = float(pi[baseline])

    kf = KFold(n_splits=n_folds, shuffle=True)

    # For each action a!=baseline, keep a list of tau estimators across folds
    tau_models_by_a = {a: [] for a in range(K) if a != baseline}

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, D_tr, y_tr = X[train_idx], D[train_idx], y[train_idx]
        X_te, D_te, y_te = X[test_idx], D[test_idx], y[test_idx]

        # nuisances on TRAIN folds
        mu_models = _fit_mu_models_by_action(X_tr, D_tr, y_tr, K=K, model_type=mu_model_type)

        # pseudo on HELD-OUT fold
        mu_hat = {a: predict_mu(mu_models[a], X_te) for a in range(K)}
        mub = mu_hat[baseline]
        Ib = (D_te == baseline).astype(float)

        for a in range(K):
            if a == baseline:
                continue
            mua = mu_hat[a]
            pia = float(pi[a])
            Ia = (D_te == a).astype(float)

            pseudo = (mua - mub) + (Ia / pia) * (y_te - mua) - (Ib / pib) * (y_te - mub)

            tau_f = _make_regressor(tau_model_type)
            tau_f.fit(X_te, pseudo)
            tau_models_by_a[a].append(tau_f)

    # optional: fit nuisance on all data for better baseline prediction in policy
    mu_models_full = _fit_mu_models_by_action(X, D, y, K=K, model_type=mu_model_type)

    return {
        "type": "k_armed_crossfit_ensemble",
        "K": K,
        "baseline": baseline,
        "pi": pi,
        "n_folds": n_folds,
        "mu_models": mu_models_full,
        "tau_models_by_a": tau_models_by_a,  # dict[a] -> list of estimators
    }


def dr_learner_predict_k_armed(dr_model, X):
    X = np.asarray(X)
    n = X.shape[0]
    K = dr_model["K"]
    baseline = dr_model["baseline"]

    tau_hat = np.zeros((n, K), float)
    for a, models in dr_model["tau_models_by_a"].items():
        preds = np.column_stack([m.predict(X) for m in models])
        tau_hat[:, a] = preds.mean(axis=1)
    tau_hat[:, baseline] = 0.0
    return tau_hat


def dr_learner_policy_k_armed(dr_model, X):
    X = np.asarray(X)
    K = dr_model["K"]
    baseline = dr_model["baseline"]

    mub = predict_mu(dr_model["mu_models"][baseline], X)
    tau_hat = dr_learner_predict_k_armed(dr_model, X)

    mu_hat = mub[:, None] + tau_hat
    mu_hat[:, baseline] = mub  # baseline = mub + 0

    a_hat = np.argmax(mu_hat, axis=1)
    return a_hat, mu_hat
