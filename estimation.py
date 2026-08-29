# estimation.py

import numpy as np
from sklearn.linear_model import LogisticRegression


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def _reference_action(actions: np.ndarray) -> int:
    return int(np.min(actions))


def _cost_adjusted_argmax(
    mean_by_action: dict[int, float],
    actions: np.ndarray,
    treatment_cost: float,
) -> int:
    """
    Scheme A (ref = min action):
      - K=2, cost>0: treat iff mean_1 - mean_0 > cost (strict)
      - K>2, cost>0: argmax_a mean_a - cost * 1[a != ref]
      - cost=0: argmax mean_a (ties → smaller action index)
    """
    actions = np.asarray(actions, dtype=int)
    if treatment_cost <= 0:
        scores = [mean_by_action[int(a)] for a in actions]
        return int(actions[np.argmax(scores)])

    if len(actions) == 2:
        ref = _reference_action(actions)
        treat = int(actions[actions != ref][0])
        if mean_by_action[treat] - mean_by_action[ref] > treatment_cost:
            return treat
        return ref

    ref = _reference_action(actions)
    scores = [
        mean_by_action[int(a)] - (treatment_cost if int(a) != ref else 0.0)
        for a in actions
    ]
    return int(actions[np.argmax(scores)])


def cost_adjusted_argmax_rows(
    mu_mat: np.ndarray,
    treatment_cost: float = 0.0,
    reference_action: int = 0,
) -> np.ndarray:
    """
  Per-user argmax of net outcome mu_a(x) - c·𝟙[a≠ref].

    Matches segment-level _cost_adjusted_argmax for K>2; for K=2 uses the
    same rule as argmax on [mu_ref, mu_treat - c] (tie → ref).
    """
    mu_mat = np.asarray(mu_mat, dtype=float)
    c = float(treatment_cost)
    if c <= 0:
        return np.argmax(mu_mat, axis=1).astype(int)
    ref = int(reference_action)
    scores = mu_mat.copy()
    for a in range(scores.shape[1]):
        if a != ref:
            scores[:, a] -= c
    return np.argmax(scores, axis=1).astype(int)


def gamma_with_action_cost(
    gamma: np.ndarray,
    action: int | np.ndarray,
    treatment_cost: float,
    reference_action: int = 0,
) -> np.ndarray | float:
    """
    Return Γ_{i,a} - c when a ≠ reference_action (flat implementation cost).

    For binary {0,1} with ref=0 this matches the paper's Γ_{i,a} - c·a.
    For K>2, every non-reference arm subtracts c (not c·a).
    """
    if treatment_cost <= 0:
        return gamma
    a = np.asarray(action, dtype=int)
    ref = int(reference_action)
    deduct = np.where(a != ref, float(treatment_cost), 0.0)
    return gamma - deduct


def _fit_logistic_or_fallback(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    D_sub: np.ndarray,
    actions: np.ndarray,
    K: int,
):
    """Fit logistic on [X, D_dummy]; return (intercept, beta, theta_full) or None."""
    n_sub = len(y_sub)
    y_int = y_sub.astype(int)
    if n_sub < 2 or len(np.unique(y_int)) < 2:
        return None

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
    except Exception:
        return None

    d_feat = X_sub.shape[1]
    beta = clf.coef_[0][:d_feat]
    theta = clf.coef_[0][d_feat:]
    theta_full = np.concatenate([[0.0], theta])
    return float(clf.intercept_[0]), beta, theta_full


def _logistic_arm_tau_hat(
    X_sub: np.ndarray,
    intercept: float,
    beta: np.ndarray,
    theta_a: float,
) -> float:
    """Σᵢ [σ(α+β'xᵢ+θ_a) − σ(α+β'xᵢ)] for one arm (θ_ref ≡ 0)."""
    eta0 = intercept + X_sub @ beta
    return float(np.sum(_sigmoid(eta0 + theta_a) - _sigmoid(eta0)))


def _logistic_action_from_fit(
    X_sub: np.ndarray,
    intercept: float,
    beta: np.ndarray,
    theta_full: np.ndarray,
    actions: np.ndarray,
    treatment_cost: float,
    n_sub: int,
) -> int:
    """
    Logistic node/segment action from fitted coefficients.

    τ̂_a = Σᵢ[σ(ηᵢ+θ_a)−σ(ηᵢ)] with θ_ref ≡ 0.
    cost > 0: score_a = τ̂_a − c·|node| for a ≠ ref; score_ref = 0.
    cost = 0: score_a = τ̂_a.
    """
    actions = np.asarray(actions, dtype=int)
    ref = _reference_action(actions)
    treatment_cost = float(treatment_cost)

    if treatment_cost > 0 and len(actions) == 2:
        treat = int(actions[actions != ref][0])
        tau_treat = _logistic_arm_tau_hat(
            X_sub, intercept, beta, float(theta_full[treat])
        )
        return treat if tau_treat >= treatment_cost * n_sub else ref

    best_a = ref
    best_score = -np.inf

    for a in actions:
        a = int(a)
        if a == ref:
            tau_a = 0.0
        else:
            tau_a = _logistic_arm_tau_hat(X_sub, intercept, beta, float(theta_full[a]))

        if treatment_cost > 0 and a != ref:
            score = tau_a - treatment_cost * n_sub
        else:
            score = tau_a

        if score > best_score + 1e-12:
            best_score = score
            best_a = a
        elif abs(score - best_score) <= 1e-12 and a < best_a:
            best_a = a

    return int(best_a)


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
    treatment_cost: float = 0.0,
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
    treatment_cost : float
        Per-unit implementation cost c (default 0). Action 0 is reference (no cost).

    Returns
    -------
    int : the best action
    """
    treatment_cost = float(treatment_cost)
    if len(indices) == 0:
        return int(actions[0])

    y_sub = y[indices]
    D_sub = D[indices]
    n_sub = len(indices)
    actions = np.asarray(actions, dtype=int)

    if method == "diff_in_means":
        mean_by_action = {}
        for a in actions:
            mask = D_sub == a
            mean_by_action[int(a)] = float(y_sub[mask].mean()) if mask.any() else 0.0
        return _cost_adjusted_argmax(mean_by_action, actions, treatment_cost)

    elif method == "gamma":
        Gamma_sub = Gamma[indices, :]
        mean_by_action = {
            int(a): float(Gamma_sub[:, int(a)].mean()) for a in actions
        }
        return _cost_adjusted_argmax(mean_by_action, actions, treatment_cost)

    elif method == "logistic":
        K = int(actions.max()) + 1
        X_sub = X[indices]
        fit = _fit_logistic_or_fallback(X_sub, y_sub, D_sub, actions, K)
        if fit is None:
            mean_by_action = {}
            for a in actions:
                mask = D_sub == a
                mean_by_action[int(a)] = float(y_sub[mask].mean()) if mask.any() else 0.0
            return _cost_adjusted_argmax(mean_by_action, actions, treatment_cost)

        intercept, beta, theta_full = fit
        return _logistic_action_from_fit(
            X_sub, intercept, beta, theta_full, actions, treatment_cost, n_sub
        )

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
    treatment_cost: float = 0.0,
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
    treatment_cost : float
        Per-unit implementation cost c (default 0).

    Returns
    -------
    action_M : np.ndarray, shape (M,)
        Recommended action for each segment m.
    """
    treatment_cost = float(treatment_cost)
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
        action_M[m] = _action_for_subset(
            X, y, D, Gamma, indices, method, actions, treatment_cost
        )

    return action_M
