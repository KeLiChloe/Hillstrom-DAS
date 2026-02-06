# s_learner.py
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor


def _one_hot_actions(D: np.ndarray, K: int) -> np.ndarray:
    D = np.asarray(D).astype(int).ravel()
    if D.min() < 0 or D.max() >= K:
        raise ValueError(f"Action D must be in [0, {K-1}], got min={D.min()}, max={D.max()}")
    return np.eye(K, dtype=float)[D]  # (n, K)


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

    参数
    ----
    K: action 数量（Hillstrom=3）
    """
    X = np.asarray(X)
    D = np.asarray(D).astype(int).ravel()
    y = np.asarray(y).astype(float).ravel()

    X_s = _build_slearner_features(X, D, K)

    def make_model():
        if model_type == "linear":
            return LinearRegression()
        if model_type == "ridge":
            return Ridge(alpha=1e-2, random_state=random_state)
        if model_type == "mlp_reg":
            return MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=300,
                early_stopping=True,
                random_state=random_state,
            )
        if model_type == "lightgbm_reg":
            return LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
                random_state=random_state,
            )
        raise ValueError(f"Unknown model_type: {model_type}")

    model = make_model()
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
        pred = s_model.predict(X_s).astype(float)
        mu_mat[:, a] = pred

    return mu_mat
