# t_learner.py
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor


def fit_t_learner(
    X: np.ndarray,
    D: np.ndarray,
    y: np.ndarray,
    K: int,
    model_type: str,
    log_y: bool,
    random_state: int,
):
    """
    训练 multi-action T-learner：
    对每个 action a，单独拟合 mu_a(x) = E[Y | X, D=a]

    返回
    ----
    models: list，长度为 K，第 a 个模型对应 action a
    """
    X = np.asarray(X)
    D = np.asarray(D).astype(int).ravel()
    y = np.asarray(y).astype(float).ravel()

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

    models = []

    for a in range(K):
        idx = D == a
        if idx.sum() == 0:
            raise ValueError(f"No samples for action {a}")

        Xa = X[idx]
        ya = y[idx]
        ya_train = np.log1p(ya) if log_y else ya

        model = make_model()
        model.fit(Xa, ya_train)
        models.append(model)

    return models


def predict_mu_t_learner_matrix(
    t_models,
    X: np.ndarray,
    log_y: bool,
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
        pred = model.predict(X).astype(float)
        mu_mat[:, a] = np.expm1(pred) if log_y else pred

    return mu_mat
