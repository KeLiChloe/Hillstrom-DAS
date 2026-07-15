import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from segmentation_fit import (
    fit_plane_model,
    plane_predict_loss,
    segmentation_bic,
)


# ============================================================
#  Core CLR logic: CLRpRegressor, clr, best_clr, bic_score
# ============================================================



def make_clr_plane_model(clr_lr=None):
    """Default plane model: logistic regression."""
    if clr_lr is not None:
        return clone(clr_lr)
    return LogisticRegression(max_iter=500, solver="lbfgs")

class CLRpRegressor(BaseEstimator):
    """
    Piecewise logistic regression with clustering on (X, D).

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
                n_jobs=1,
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
    Core CLR: alternate plane fitting and cluster reassignment.
    Binary y: logistic planes + deviance.
    """
    N, _ = X_D.shape
    y = np.asarray(y, dtype=float)

    if cluster_labels is None:
        cluster_labels = np.random.choice(k, size=N)

    models = [make_clr_plane_model(clr_lr=lr) for _ in range(k)]
    scores = np.empty((N, k))

    for _ in range(max_iter):
        # 1) rebuild models
        for cl_idx in range(k):
            mask = cluster_labels == cl_idx
            if mask.sum() == 0:
                continue
            if not fit_plane_model(models[cl_idx], X_D[mask], y[mask]):
                models[cl_idx] = make_clr_plane_model(clr_lr=lr)

        # 2) reassign points
        for cl_idx in range(k):
            mask = cluster_labels == cl_idx
            if mask.sum() == 0 or not hasattr(models[cl_idx], "coef_"):
                scores[:, cl_idx] = np.inf
                continue

            scores[:, cl_idx] = plane_predict_loss(models[cl_idx], X_D, y)

            if kmeans_coef > 0:
                center = np.mean(X_D[mask], axis=0)
                dist2 = np.sum((X_D - center) ** 2, axis=1)
                scores[:, cl_idx] += kmeans_coef * dist2

        cluster_labels_prev = cluster_labels.copy()
        cluster_labels = np.argmin(scores, axis=1)

        if np.allclose(cluster_labels, cluster_labels_prev):
            break

    obj = np.mean(scores[np.arange(N), cluster_labels])

    # cluster weights
    weights = (cluster_labels == np.arange(k)[:, np.newaxis]).sum(axis=1).astype(float)
    weights /= weights.sum()

    return cluster_labels, models, weights, obj


def bic_score(X_D, y, cluster_labels, models):
    """BIC for piecewise planes (logistic if models are classifiers, else Gaussian)."""
    return segmentation_bic(X_D, y, cluster_labels, models)


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
