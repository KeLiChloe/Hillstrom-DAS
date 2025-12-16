# outcome_model.py

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor


# =====================================================================
#  Utility: strict fit for regression（单一取值时直接报错）
# =====================================================================

def safe_fit_reg(model, X, y):
    """
    严格版 fit:
      - 如果 y 只有一个取值，直接报错（提醒该 action 数据有问题）
      - 否则正常 fit
    """
    y = np.asarray(y)
    unique_vals = np.unique(y)
    if unique_vals.size <= 1:
        raise ValueError(
            f"safe_fit_reg: y has only a single unique value ({unique_vals[0]}). "
            "This indicates degenerate data for this action; please check your split / sampling."
        )
    model.fit(X, y)
    return model


# =====================================================================
#  Fit μ_a(x) for Hillstrom (multi-action, regression)
# =====================================================================

def fit_mu_models(X, D, y, model_type):
    """
    Hillstrom 专用版：为每个 action a 拟合回归模型

        μ_a(x) = E[Y | X, D = a]

    不区分 binary / continuous y，统统当作回归问题。
    对于 y∈{0,1} 时，μ_a(x) 仍然是 P(Y=1|X,D=a) 的近似。
    """
    X = np.asarray(X)
    D = np.asarray(D)
    y = np.asarray(y)

    actions = np.unique(D)

    def make_model():
        if model_type == "linear":
            return LinearRegression()

        elif model_type == "mlp_reg":
            return MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=300,
                early_stopping=True,
            )

        elif model_type == "lightgbm_reg":
            return LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=-1,
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    mu_models = {}
    for a in actions:
        mask_a = (D == a)
        if mask_a.sum() == 0:
            continue
        model_a = make_model()
        # 这里如果该 action 下 y 全一样，会直接抛 ValueError
        mu_models[int(a)] = safe_fit_reg(model_a, X[mask_a], y[mask_a])

    return mu_models


# =====================================================================
#  Predict μ(x) = E[Y|X] from a given model
# =====================================================================

def predict_mu(mu_model, X):
    """
    回归版 predict：
      - 直接调用 model.predict(X)，得到 E[Y|X]
      - 对 y 二元 / 连续都适用
    """
    return mu_model.predict(X)
