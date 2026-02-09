# estimation.py

import numpy as np
from sklearn.linear_model import LinearRegression  # 当前未使用，可按需删掉


def estimate_segment_policy(X, y, D, seg_labels):
    """
    Estimate segment-level policy by segment-wise argmax of mean(y) over actions.

    对每个 segment m 和每个 action a，计算：
        μ̂_{m,a} = mean(y_i : seg_labels_i = m, D_i = a)
    然后选择：
        action[m] = argmax_a μ̂_{m,a}

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Feature matrix (当前函数并未使用，但保留接口以便将来扩展)
    y : np.ndarray, shape (N,)
        Outcome vector
    D : np.ndarray, shape (N,)
        Discrete action / treatment labels (e.g., 0/1 or 0,1,2,...)
    seg_labels : np.ndarray, shape (N,)
        Segment assignments for each sample

    Returns
    -------
    action_M : np.ndarray, shape (M,)
        Recommended action for each segment m, where M = 1 + max(seg_labels).
        action[m] ∈ unique(D)
    """
    y = np.asarray(y)
    D = np.asarray(D)
    seg_labels = np.asarray(seg_labels)

    M = int(seg_labels.max() + 1)
    action_M = np.zeros(M, dtype=int)

    actions = np.unique(D)

    for m in range(M):
        idx_m = (seg_labels == m)

        D_seg = D[idx_m]
        y_seg = y[idx_m]

        est_means = []
        for a in actions:
            mask_a = (D_seg == a)
            if mask_a.sum() == 0:
                raise ValueError(
                    f"No samples for segment {m} and action {a}. "
                    "Cannot estimate mean outcome."
                )
            else:
                est_means.append(y_seg[mask_a].mean())

        est_means = np.array(est_means)

        best_a = actions[np.argmax(est_means)]
        action_M[m] = int(best_a)

    return action_M
