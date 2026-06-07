# t_learner.py
import numpy as np
from outcome_model import make_mu_model, predict_mu_values


def fit_t_learner(
    X: np.ndarray,
    D: np.ndarray,
    y: np.ndarray,
    K: int,
    model_type: str,
    random_state: int,
):
    """
    训练 multi-action T-learner：
    对每个 action a，单独拟合 mu_a(x) = E[Y | X, D=a]

    model_type follows --mu_model_type (see outcome_model.make_mu_model).
    """
    X = np.asarray(X)
    D = np.asarray(D).astype(int).ravel()
    y = np.asarray(y).astype(float).ravel()

    models = []

    for a in range(K):
        idx = D == a
        if idx.sum() == 0:
            raise ValueError(f"No samples for action {a}")

        Xa = X[idx]
        ya = y[idx]

        model = make_mu_model(
            model_type,
            random_state=random_state,
            y=ya if model_type == "lightgbm_clf" else None,
        )
        model.fit(Xa, ya)
        models.append(model)

    return models


def predict_mu_t_learner_matrix(
    t_models,
    X: np.ndarray,
):
    """
    返回 mu_mat: (n, K)，
    mu_mat[i,a] = E[Y | X_i, D=a] 的预测
    """
    X = np.asarray(X)
    n = X.shape[0]
    K = len(t_models)

    mu_mat = np.zeros((n, K), dtype=float)

    for a, model in enumerate(t_models):
        mu_mat[:, a] = predict_mu_values(model, X)

    return mu_mat
