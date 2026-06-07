# s_learner.py
import numpy as np
from outcome_model import make_mu_model, predict_mu_values


def _one_hot_actions(D: np.ndarray, K: int) -> np.ndarray:
    D = np.asarray(D).astype(int).ravel()
    if D.min() < 0 or D.max() >= K:
        raise ValueError(f"Action D must be in [0, {K-1}], got min={D.min()}, max={D.max()}")
    return np.eye(K, dtype=float)[D]


def _build_slearner_features(X: np.ndarray, D: np.ndarray, K: int) -> np.ndarray:
    """
    S-learner 的特征： [X, onehot(D)]
    """
    X = np.asarray(X)
    D_oh = _one_hot_actions(D, K)
    return np.hstack([X, D_oh])


def fit_s_learner(
    X: np.ndarray,
    D: np.ndarray,
    y: np.ndarray,
    K: int,
    model_type: str,
    random_state: int,
):
    """
    训练 multi-action S-learner：一个模型拟合 mu(x,a)=E[Y|X,D=a]

    model_type follows --mu_model_type (see outcome_model.make_mu_model).
    """
    X = np.asarray(X)
    D = np.asarray(D).astype(int).ravel()
    y = np.asarray(y).astype(float).ravel()

    X_s = _build_slearner_features(X, D, K)

    model = make_mu_model(
        model_type,
        random_state=random_state,
        y=y if model_type == "lightgbm_clf" else None,
    )
    model.fit(X_s, y)
    return model


def predict_mu_s_learner_matrix(
    s_model,
    X: np.ndarray,
    K: int,
):
    """
    返回 mu_mat: (n, K)，其中 mu_mat[i,a] = E[Y|X_i, D=a] 的预测
    """
    X = np.asarray(X)
    n = X.shape[0]
    mu_mat = np.zeros((n, K), dtype=float)

    for a in range(K):
        D_a = np.full(n, a, dtype=int)
        X_s = _build_slearner_features(X, D_a, K)
        mu_mat[:, a] = predict_mu_values(s_model, X_s)

    return mu_mat
