# policytree.py
"""
Policy Tree individual-level policy recommendations via R's policytree package.

Workflow:
1. Fit GRF multi_arm_causal_forest on pilot data (W always as factor)
2. Compute doubly-robust scores (Gamma)
3. Fit a shallow policy tree on (X, Gamma)
4. Predict recommended action per implementation customer (type="action.id")

NOTE: we do NOT use a numpy2ri localconverter block.  The numpy2ri converter
round-trips rpy2 R objects back through numpy, stripping class attributes and
turning R numeric vectors into R 'array' objects.  Instead every R object is
built explicitly from rpy2 primitive constructors so GRF sees the correct types:
  X → R matrix   (class: 'matrix' 'array')
  Y → R numeric  (class: 'numeric')
  W → R factor   (class: 'factor')
"""

import numpy as np

_r_initialized = False
_ro = None
_grf = None
_policytree_r = None


def _init_r():
    global _r_initialized, _ro, _grf, _policytree_r
    if _r_initialized:
        return
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    _ro = ro
    _grf = importr("grf")
    _policytree_r = importr("policytree")
    _r_initialized = True
    print("[policytree] R packages loaded (grf, policytree)", flush=True)


# =====================================================================
#  Explicit R object constructors (no localconverter / numpy2ri)
# =====================================================================

def _as_r_matrix(arr2d: np.ndarray):
    """(n, p) float64 numpy array → R matrix (class 'matrix' 'array')."""
    n, p = arr2d.shape
    flat = arr2d.reshape(-1).tolist()
    rvec = _ro.FloatVector(flat)
    return _ro.r.matrix(rvec, nrow=n, ncol=p, byrow=True)


def _as_r_numeric(arr1d: np.ndarray):
    """1-D float64 numpy array → R numeric vector (class 'numeric')."""
    return _ro.FloatVector(arr1d.tolist())


def _as_r_factor(arr1d_int: np.ndarray):
    """1-D int numpy array → R factor (class 'factor')."""
    ivec = _ro.IntVector(arr1d_int.tolist())
    return _ro.r["as.factor"](ivec)


def _gamma_column_actions(gamma_r) -> np.ndarray:
    """
    Return the actual action label for each column of the Gamma matrix,
    in R column order.

    policytree predict(type='action.id') returns 1-based column indices;
    this mapping converts those back to the original integer treatment codes.
    """
    n_cols = int(list(_ro.r("ncol")(gamma_r))[0])
    if n_cols < 1:
        raise ValueError("Gamma matrix has no columns")

    colnames = _ro.r("colnames")(gamma_r)
    if list(_ro.r["is.null"](colnames))[0]:
        # fallback: assume columns are 0, 1, ..., n_cols-1
        return np.arange(n_cols, dtype=int)

    return np.array([int(float(str(s))) for s in list(colnames)], dtype=int)


# =====================================================================
#  Public API: individual policy (classic policytree usage)
# =====================================================================

def run_policytree_individual(
    X_pilot: np.ndarray,
    y_pilot: np.ndarray,
    D_pilot: np.ndarray,
    X_impl: np.ndarray,
    depth: int,
) -> np.ndarray:
    """
    Classic policytree usage: fit on pilot, predict individual actions for
    implementation customers.

    Steps:
      1. Fit GRF multi_arm_causal_forest on pilot (W as R factor).
      2. Compute double_robust_scores → Gamma (N_pilot × K).
      3. Fit policy_tree(X_pilot, Gamma, depth=depth) in R.
      4. Predict per-customer action for each row of X_impl.

    Parameters
    ----------
    X_pilot : (N_pilot, d)
    y_pilot : (N_pilot,)
    D_pilot : (N_pilot,)  integer-coded treatments
    X_impl  : (N_impl, d)
    depth   : int  

    Returns
    -------
    action_impl : np.ndarray, shape (N_impl,)
        Per-customer recommended action, using the original treatment codes.
    """
    _init_r()

    # ── normalise inputs ──────────────────────────────────────────────────────
    X_pilot = np.asarray(X_pilot, dtype=np.float64)
    y_pilot = np.asarray(y_pilot, dtype=np.float64).ravel()
    D_pilot = np.asarray(D_pilot).ravel().astype(int)
    X_impl  = np.asarray(X_impl,  dtype=np.float64)
    if X_pilot.ndim == 1:
        X_pilot = X_pilot.reshape(-1, 1)
    if X_impl.ndim == 1:
        X_impl = X_impl.reshape(-1, 1)

    unique_actions = np.unique(D_pilot)
    if len(unique_actions) < 2:
        raise ValueError(
            f"policytree needs >= 2 actions in pilot data, got {unique_actions}"
        )

    n_train, n_impl = X_pilot.shape[0], X_impl.shape[0]
    print(
        f"[policytree] start: n_train={n_train}, n_impl={n_impl}, "
        f"d={X_pilot.shape[1]}, depth={depth}, actions={unique_actions.tolist()}",
        flush=True,
    )

    # ── build R objects (no localconverter) ───────────────────────────────────
    X_r      = _as_r_matrix(X_pilot)
    y_r      = _as_r_numeric(y_pilot)
    D_r      = _as_r_factor(D_pilot)
    X_impl_r = _as_r_matrix(X_impl)

    # ── GRF forest → Gamma ────────────────────────────────────────────────────
    print("[policytree] fitting multi_arm_causal_forest ...", flush=True)
    forest = _grf.multi_arm_causal_forest(X_r, y_r, D_r)

    print("[policytree] computing double_robust_scores (Gamma) ...", flush=True)
    gamma_r = _policytree_r.double_robust_scores(forest)
    action_identity = _gamma_column_actions(gamma_r)   # actual action labels per column
    print(
        f"[policytree] Gamma: {n_train} x {len(action_identity)}, "
        f"column actions = {action_identity.tolist()}",
        flush=True,
    )

    # ── policy tree ───────────────────────────────────────────────────────────
    print(f"[policytree] fitting policy_tree(depth={depth}) ...", flush=True)
    tree = _policytree_r.policy_tree(X_r, gamma_r, depth=depth)

    # ── predict on implementation set ─────────────────────────────────────────
    print("[policytree] predicting on implementation set ...", flush=True)
    action_r = _policytree_r.predict_policy_tree(tree, X_impl_r, type="action.id")
    action_ids_raw = np.array(list(action_r), dtype=int)  # 1-based column index

    # Convert 1-based column index → 0-based → actual action label
    recommended_idx = action_ids_raw - 1
    n_cols = len(action_identity)
    if recommended_idx.min() < 0 or recommended_idx.max() >= n_cols:
        raise ValueError(
            f"R action.id out of range [1, {n_cols}]; "
            f"got [{action_ids_raw.min()}, {action_ids_raw.max()}]"
        )

    action_impl = action_identity[recommended_idx]

    unique_rec, counts = np.unique(action_impl, return_counts=True)
    dist = ", ".join(f"a={a}:{c}" for a, c in zip(unique_rec, counts))
    print(f"[policytree] done: recommended actions {{{dist}}}", flush=True)

    return action_impl.astype(int)
