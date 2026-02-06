# outcome_model.py

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMRegressor, LGBMClassifier


def _is_binary01(y):
    uniq = np.unique(y)
    return uniq.size == 2 and set(uniq.tolist()).issubset({0, 1})


def _safe_fit(model, X, y, min_pos=10):
    """
    - y 只有一个取值 => 报错
    - 若是 {0,1} 二分类 => 正类数 < min_pos 报错（否则后面分层/阈值都不稳）
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


def _pick_threshold_max_f1(y_true, y_prob):
    """
    简单扫阈值找 max F1
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, 1001):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t


def fit_mu_models(X, D, y, mu_model_type, val_size=0.2, random_state=42):
    """
    对每个 action a 拟合 μ_a(x) = E[Y|X,D=a]
    - 回归模型：返回 (model, None)
    - 分类模型（lightgbm_clf / logistic）：自动选 threshold，返回 (model, threshold)
    """
    X = np.asarray(X)
    D = np.asarray(D)
    y = np.asarray(y)

    actions = np.unique(D)
    mu_model_tuples = {}

    for a in actions:
        mask_a = (D == a)
        Xa, ya = X[mask_a], y[mask_a]
        if Xa.shape[0] == 0:
            raise ValueError(f"No samples for action {a}")

        # -------- build model --------
        if mu_model_type == "linear":
            model = LinearRegression()
        elif mu_model_type == "mlp_reg":
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=10000,
                early_stopping=True,
            )
        elif mu_model_type == "lightgbm_reg":
            model = LGBMRegressor(n_estimators=200, learning_rate=0.05)

        elif mu_model_type == "logistic":
            model = LogisticRegression(max_iter=500)

        elif mu_model_type == "lightgbm_clf":
            n_pos = int((ya == 1).sum())
            n_neg = int((ya == 0).sum())
            if n_pos == 0 or n_neg == 0:
                raise ValueError(f"action={a}: y is degenerate (n_pos={n_pos}, n_neg={n_neg})")
            pos_weight = n_neg / n_pos
            model = LGBMClassifier(
                objective="binary",
                n_estimators=300,
                learning_rate=0.05,
                scale_pos_weight=pos_weight,
            )
        else:
            raise ValueError(f"Unknown mu_model_type: {mu_model_type}")

        # -------- fit + threshold if classifier --------
        is_clf = hasattr(model, "predict_proba") and set(np.unique(ya).tolist()).issubset({0, 1})

        if is_clf:
            # 要能做 stratify split：两类都得有、且至少各 2 个更稳
            if not _is_binary01(ya):
                raise ValueError(f"action={a}: classifier needs y in {{0,1}} with both classes present")

            # 分层切 val
            Xtr, Xva, ytr, yva = train_test_split(
                Xa, ya, test_size=val_size, random_state=random_state, stratify=ya
            )

            model = _safe_fit(model, Xtr, ytr, min_pos=2)
            val_prob = model.predict_proba(Xva)[:, 1]
            thr = _pick_threshold_max_f1(yva, val_prob)

            # 用全量再 fit 一次，threshold 用刚才选的（更常见的做法）
            model_full = model.__class__(**model.get_params())
            model_full = _safe_fit(model_full, Xa, ya, min_pos=2)

            mu_model_tuples[int(a)] = (model_full, thr)
        else:
            model = _safe_fit(model, Xa, ya, min_pos=2)
            mu_model_tuples[int(a)] = (model, None)

    return mu_model_tuples


def predict_mu(model_tuple, X, return_label=False):
    """
    model_tuple = (model, threshold)

    return_label=False: 返回 μ(x)=E[Y|X]
        - reg: predict
        - clf: predict_proba[:,1]

    return_label=True: 返回 0/1 label（只对 clf 有效）
    """
    model, thr = model_tuple

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        if return_label:
            if thr is None:
                raise ValueError("No threshold stored for this classifier.")
            return (prob >= thr).astype(int)
        return prob

    # regressor
    if return_label:
        raise ValueError("return_label=True is only valid for classifier models.")
    
    return model.predict(X)
