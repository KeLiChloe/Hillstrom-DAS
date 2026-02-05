import numpy as np
from sklearn.metrics import silhouette_score


def kmeans_silhouette_score(seg_model, X_pilot):
    """
    KMeans segmentation scoring using Silhouette Score.

    Parameters
    ----------
    seg_model : KMeansSeg
        已经调用 seg_model.fit(X_pilot) 的 KMeans segmentation 模型
    X_pilot : np.ndarray
        Feature matrix for pilot dataset  

    Returns
    -------
    float
        Silhouette Score (越大越好)
    """
    labels = seg_model.assign(X_pilot)
    return silhouette_score(X_pilot, labels)


def dams_score(seg_model, X_val, D_val, y_val, Gamma_val, action, value_type_dams):
    """
    Decision-Aware Model Selection 的 scoring 函数（Algorithm 3），
    K-action 通用版本。

    参数
    ----
    seg_model : segmentation 模型（在 train_seg 上已经 fit 好）
                需要有 .assign(X) 方法
    X_val : np.ndarray, shape (N_val, d)
        validation 特征
    D_val : np.ndarray, shape (N_val,)
        validation 实际动作（已经编码为 0..K-1 的整数）
    y_val : np.ndarray, shape (N_val,)
        validation 实际 outcome
    Gamma_val : np.ndarray, shape (N_val, K)
        对每个样本 i 和每个动作 a 的 DR 估计 Γ_{i,a}
        —— 例如由 prepare_pilot_impl 产生的 Gamma_pilot 再划分出的 val 部分
    action : np.ndarray, shape (M,)
        每个 segment m 的动作 a_m（整数 0..K-1），在 train_seg 阶段学好

    返回
    ----
    float
        DAMS 的 validation score（平均 policy value）
    """
    X_val = np.asarray(X_val)
    D_val = np.asarray(D_val).astype(int)
    y_val = np.asarray(y_val, dtype=float)
    Gamma_val = np.asarray(Gamma_val, dtype=float)
    action = np.asarray(action, dtype=int)

    # 1) segmentation: assign each i to segment m
    labels = seg_model.assign(X_val)        # shape (N_val,)

    a_i = action[labels].astype(int)        # π^{C_M}(i)，shape (N_val,)

    v_hat = np.empty_like(y_val, dtype=float)

    mask_match = (D_val == a_i)
    mask_mismatch = ~mask_match

    # factual 部分
    if value_type_dams == "hybrid":
        v_hat[mask_match] = y_val[mask_match]
        idx_mismatch = np.where(mask_mismatch)[0]
        v_hat[idx_mismatch] = Gamma_val[idx_mismatch, a_i[idx_mismatch]]
    elif value_type_dams == "dr":
        N = X_val.shape[0]
        v_hat = Gamma_val[np.arange(N), a_i]
    return float(v_hat.mean())
