# estimation.py

import numpy as np
from sklearn.linear_model import LogisticRegression


# =========================================================
# Primitive: best action for an arbitrary subset of samples
# =========================================================

def _action_for_subset(
    X: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    Gamma,          # np.ndarray (N, K) or None
    indices: np.ndarray,
    method: str,
    actions: np.ndarray,
) -> int:
    """
    Return the best action for the subset of samples selected by `indices`.

    This is the single shared primitive used by both:
      - estimate_segment_policy  (post-hoc segment → batch of segments)
      - DASTree._get_node_action (grow phase  → single node by raw indices)

    Parameters
    ----------
    X       : (N, d) features — used only by logistic
    y       : (N,)  outcomes  — used by diff_in_means and logistic
    D       : (N,)  integer actions — used by all methods
    Gamma   : (N, K) DR pseudo-outcome matrix — used by gamma; may be None
    indices : 1-D int array, sample indices to consider
    method  : 'diff_in_means', 'gamma', or 'logistic'
    actions : 1-D int array, full action space (from the whole dataset)

    Returns
    -------
    int : the best action
    """
    if len(indices) == 0:
        return int(actions[0])

    y_sub = y[indices]
    D_sub = D[indices]

    if method == "diff_in_means":
        means = []
        for a in actions:
            mask = D_sub == a
            means.append(float(y_sub[mask].mean()) if mask.any() else 0.0)
        return int(actions[np.argmax(means)])

    elif method == "gamma":
        Gamma_sub = Gamma[indices, :]
        means = [float(Gamma_sub[:, a].mean()) for a in actions]
        return int(actions[np.argmax(means)])

    elif method == "logistic":
        K = int(actions.max()) + 1
        X_sub = X[indices]
        y_int = y_sub.astype(int)
        n_sub = len(indices)

        # fallback: too little data or constant outcome
        if n_sub < 2 or len(np.unique(y_int)) < 2:
            means = []
            for a in actions:
                mask = D_sub == a
                means.append(float(y_sub[mask].mean()) if mask.any() else 0.0)
            return int(actions[np.argmax(means)])

        # Build treatment dummy matrix (reference action = 0, θ_0 ≡ 0)
        if K == 2:
            D_dummy = (D_sub == 1).astype(float).reshape(-1, 1)
        else:
            D_dummy = np.zeros((n_sub, K - 1), dtype=float)
            for j, a in enumerate(range(1, K)):
                D_dummy[:, j] = (D_sub == a).astype(float)

        XD = np.hstack([X_sub, D_dummy])
        clf = LogisticRegression(max_iter=500, solver="lbfgs", C=1.0)
        try:
            clf.fit(XD, y_int)
            d_feat = X_sub.shape[1]
            theta = clf.coef_[0][d_feat:]               # (K-1,): θ_1 ... θ_{K-1}
            theta_full = np.concatenate([[0.0], theta])  # (K,):  θ_0=0, θ_1, ...
            return int(np.argmax(theta_full))
        except Exception:
            means = []
            for a in actions:
                mask = D_sub == a
                means.append(float(y_sub[mask].mean()) if mask.any() else 0.0)
            return int(actions[np.argmax(means)])

    else:
        raise ValueError(
            f"Unknown method='{method}'. "
            "Choose from: 'diff_in_means', 'gamma', 'logistic'."
        )


# =========================================================
# Public API: segment-level policy estimation
# =========================================================

def estimate_segment_policy(
    X,
    y,
    D,
    seg_labels,
    method,
    Gamma=None,
):
    """
    Estimate segment-level policy action assignments.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Feature matrix.
    y : np.ndarray, shape (N,)
        Outcome vector.  Used by diff_in_means and logistic.
    D : np.ndarray, shape (N,)
        Discrete action labels.  Used by diff_in_means and logistic.
    seg_labels : np.ndarray, shape (N,)
        Segment id per sample (0-indexed, contiguous).
    method : {"diff_in_means", "gamma", "logistic"}
        How to estimate the best action per segment.
    Gamma : np.ndarray, shape (N, K) or None
        DR pseudo-outcome matrix.  Required when method="gamma".

    Returns
    -------
    action_M : np.ndarray, shape (M,)
        Recommended action for each segment m.
    """
    y = np.asarray(y, dtype=float)
    D = np.asarray(D, dtype=int)
    seg_labels = np.asarray(seg_labels, dtype=int)

    if method == "gamma" and Gamma is None:
        raise ValueError("method='gamma' requires Gamma (DR pseudo-outcome matrix).")
    if Gamma is not None:
        Gamma = np.asarray(Gamma, dtype=float)
    if method == "logistic":
        X = np.asarray(X, dtype=float)

    M = int(seg_labels.max()) + 1
    actions = np.unique(D)

    action_M = np.zeros(M, dtype=int)
    for m in range(M):
        indices = np.where(seg_labels == m)[0]
        action_M[m] = _action_for_subset(X, y, D, Gamma, indices, method, actions)

    return action_M
