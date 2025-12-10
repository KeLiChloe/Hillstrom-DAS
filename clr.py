import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted


# ============================================================
#  Core CLR logic: CLRpRegressor, clr, best_clr, bic_score
# ============================================================

class CLRpRegressor(BaseEstimator):
    """
    Piecewise linear regression with clustering on (X, D).

    用法：
        X_D = np.column_stack([X, D])
        clr = CLRpRegressor(num_planes=K, kmeans_coef=..., ...)
        clr.fit(X_D, y)
        labels = clr.cluster_labels
        models = clr.models
        # clf 用来在新 X 上预测 cluster
    """

    def __init__(
        self,
        num_planes,
        kmeans_coef,
        clr_lr=None,
        max_iter=5,
        num_tries=8,
        clf=None,
        random_state=None,
    ):
        self.num_planes = num_planes
        self.kmeans_coef = kmeans_coef
        self.num_tries = num_tries
        self.clr_lr = clr_lr
        self.max_iter = max_iter
        self.random_state = random_state

        if clf is None:
            self.clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=None,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            self.clf = clf

    def fit(self, X_D, y):
        """
        X_D: (N, d+1) matrix, 最后一列是 D
        y:   (N,) vector
        """

        # 核心：用 best_clr 找到最优 cluster_labels 和 models
        (
            self.cluster_labels_,
            self.models_,
            self.weights_,
            self.obj_,
        ) = best_clr(
            X_D,
            y,
            k=self.num_planes,
            kmeans_coef=self.kmeans_coef,
            max_iter=self.max_iter,
            num_tries=self.num_tries,
            lr=self.clr_lr,
        )

        # 确保至少有 2 个 cluster，否则 RandomForestClassifier 报错
        if np.unique(self.cluster_labels_).shape[0] == 1:
            # 人为把第一个点的 label 改成另一个类
            if self.cluster_labels_[0] == 0:
                self.cluster_labels_[0] = 1
            else:
                self.cluster_labels_[0] = 0

        # 拟合一个 classifier: X -> cluster_label
        X_no_D = X_D[:, :-1]  # 去掉 D
        self.clf.fit(X_no_D, self.cluster_labels_)

        return self

    def predict(self, X_only):
        """
        X_only: (N, d) 只含 X，不含 D

        返回 cluster labels (segments)。
        """
        check_is_fitted(self, ["cluster_labels_", "models_", "clf"])
        return self.clf.predict(X_only)


def best_clr(X_D, y, k, num_tries=5, **kwargs):
    """
    多次随机初始化，取目标值最小的那次 CLR 解。
    """
    best_obj = np.inf
    best_cluster_labels = None
    best_models = None
    best_weights = None

    for _ in range(num_tries):
        cluster_labels, models, weights, obj = clr(X_D, y, k, **kwargs)
        if obj < best_obj:
            best_obj = obj
            best_cluster_labels = cluster_labels
            best_models = models
            best_weights = weights

    return best_cluster_labels, best_models, best_weights, best_obj


def clr(X_D, y, k, kmeans_coef, lr=None, max_iter=10, cluster_labels=None):
    """
    核心 CLR 算法：
      - 初始 cluster_labels（随机）
      - 交替更新：
        1) 每个 cluster 拟合一个线性回归模型
        2) 重新分配每个样本到损失 (误差^2 + kmeans_coef*距离^2) 最小的 cluster
      - 直到收敛或达到 max_iter
    """
    N, d_plus_1 = X_D.shape

    if cluster_labels is None:
        cluster_labels = np.random.choice(k, size=N)

    if lr is None:
        # set fit_intercept=True if X_D 没有自己加截距
        lr = Ridge(alpha=1e-5, fit_intercept=True)

    models = [clone(lr) for _ in range(k)]
    scores = np.empty((N, k))
    preds = np.empty((N, k))

    for _ in range(max_iter):
        # 1) rebuild models
        for cl_idx in range(k):
            mask = (cluster_labels == cl_idx)
            if mask.sum() == 0:
                continue
            models[cl_idx].fit(X_D[mask], y[mask])

        # 2) reassign points
        for cl_idx in range(k):
            if models[cl_idx] is None:
                scores[:, cl_idx] = np.inf
                preds[:, cl_idx] = 0.0
                continue

            preds[:, cl_idx] = models[cl_idx].predict(X_D)
            # 回归误差
            scores[:, cl_idx] = (y - preds[:, cl_idx]) ** 2

            # k-means regularization
            if kmeans_coef > 0 and (cluster_labels == cl_idx).sum() > 0:
                center = np.mean(X_D[cluster_labels == cl_idx], axis=0)
                dist2 = np.sum((X_D - center) ** 2, axis=1)
                scores[:, cl_idx] += kmeans_coef * dist2

        cluster_labels_prev = cluster_labels.copy()
        cluster_labels = np.argmin(scores, axis=1)

        if np.allclose(cluster_labels, cluster_labels_prev):
            break

    # final objective
    obj = np.mean(scores[np.arange(N), cluster_labels])

    # cluster weights
    weights = (cluster_labels == np.arange(k)[:, np.newaxis]).sum(axis=1).astype(float)
    weights /= weights.sum()

    return cluster_labels, models, weights, obj


def bic_score(X_D, y, cluster_labels, models):
    """
    BIC for piecewise linear regression with Gaussian errors.

    - X_D: (N, d+1)
    - y: (N,)
    - cluster_labels: (N,)
    - models: list of linear models, length k
    """
    n, d = X_D.shape
    k = len(models)

    y_hat = np.zeros_like(y, dtype=float)
    for cl_idx in range(k):
        mask = (cluster_labels == cl_idx)
        if mask.sum() == 0:
            continue
        y_hat[mask] = models[cl_idx].predict(X_D[mask])

    residuals = y - y_hat
    sigma2 = np.mean(residuals ** 2)
    # log-likelihood under Gaussian errors
    logL = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)

    # number of parameters: each linear model has (d+1) params (including intercept)
    p = k * (d + 1)

    BIC = -2.0 * logL + p * np.log(n)
    return BIC


# ============================================================
#  Wrapper class for your current pipeline
#  - fit(X, D, y)
#  - assign(X)
#  - has attributes: n_segments, kmeans_coef, cluster_labels, models
# ============================================================

class CLRSeg:
    """
    Pipeline-friendly CLR segmentation:

    使用方法（和 KMeansSeg / GMM 一致风格）：

        seg = CLRSeg(n_segments=K, kmeans_coef=0.1)
        seg.fit(X_pilot, D_pilot, y_pilot)
        labels_pilot = seg.assign(X_pilot)
        labels_impl  = seg.assign(X_impl)

    你可以在外面写 run_clr_segmentation，用 BIC 做 K 选择。
    """

    def __init__(
        self,
        n_segments,
        kmeans_coef=0.3,
        num_tries=8,
        clr_lr=None,
        max_iter=10,
        clf=None,
        random_state=0,
    ):
        self.k = n_segments
        self.kmeans_coef = kmeans_coef
        self.num_tries = num_tries
        self.clr_lr = clr_lr
        self.max_iter = max_iter
        self.random_state = random_state
        self.clf = clf

        # 这些会在 fit 之后被填充
        self._core = None          # CLRpRegressor 实例
        self.cluster_labels = None
        self.models = None

    def fit(self, X, D, y):
        """
        X: (N, d) covariates
        D: (N,) treatment indicator
        y: (N,) outcome
        """
        X = np.asarray(X)
        D = np.asarray(D).reshape(-1, 1)
        y = np.asarray(y).ravel()

        X_D = np.column_stack([X, D])

        core = CLRpRegressor(
            num_planes=self.k,
            kmeans_coef=self.kmeans_coef,
            clr_lr=self.clr_lr,
            max_iter=self.max_iter,
            num_tries=self.num_tries,
            clf=self.clf,
            random_state=self.random_state,
        )
        core.fit(X_D, y)

        self._core = core
        self.cluster_labels = core.cluster_labels_
        self.models = core.models_
        return self

    def assign(self, X):
        """
        X: (N, d) covariates
        返回 segment label (cluster labels).
        """
        if self._core is None:
            raise RuntimeError("CLRSeg: call fit(X, D, y) before assign().")
        X = np.asarray(X)
        return self._core.predict(X)


def clr_bic_score(seg_model: CLRSeg, X, D, y):
    """
    Convenience 函数：给 run_clr_segmentation 用。

    seg_model: 已经 fit 过的 CLRSeg
    X, D, y: 和 fit 时同一个 pilot 数据

    返回一个 scalar BIC（越小越好）。
    """
    X = np.asarray(X)
    D = np.asarray(D).reshape(-1, 1)
    y = np.asarray(y).ravel()

    X_D = np.column_stack([X, D])
    return bic_score(X_D, y, seg_model.cluster_labels, seg_model.models)
