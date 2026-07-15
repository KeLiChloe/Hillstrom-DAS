# segmentation.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from scoring import dams_score, kmeans_silhouette_score
from dast import DASTree
from estimation import estimate_segment_policy  
from clr import CLRSeg, clr_bic_score
from mst import MSTree



class BaseSegmentation:
    """Base class for segmentation methods."""
    def fit(self, X, *args, **kwargs):
        raise NotImplementedError

    def assign(self, X):
        raise NotImplementedError


class KMeansSeg(BaseSegmentation):
    """K-Means based segmentation."""
    def __init__(self, n_segments, random_state=0):
        self.k = n_segments
        self.random_state = random_state
        self.model = None

    def fit(self, X, *args, **kwargs):
        self.model = KMeans(
            n_clusters=self.k,
            n_init=5,
            random_state=self.random_state
        ).fit(X)
        return self

    def assign(self, X):
        if self.model is None:
            raise RuntimeError("KMeansSeg: call fit() first")
        return self.model.predict(X)


class GMMSeg(BaseSegmentation):
    """Gaussian Mixture Model based segmentation."""
    def __init__(self, n_segments, covariance_type="full", random_state=0):
        self.k = n_segments
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None

    def fit(self, X, *args, **kwargs):
        self.model = GaussianMixture(
            n_components=self.k,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=5
        )
        self.model.fit(X)
        return self

    def assign(self, X):
        if self.model is None:
            raise RuntimeError("GMMSeg: call fit() first")
        return self.model.predict(X)

    def bic(self, X):
        """Convenience wrapper for BIC on given data."""
        if self.model is None:
            raise RuntimeError("GMMSeg: call fit() before bic()")
        return self.model.bic(X)





# =========================================================
# 2. KMeans segmentation + K 选择
# =========================================================
def run_kmeans_segmentation(X_pilot, M_candidates, random_state):
    print("\n" + "=" * 60)
    print("KMeans - selecting optimal K")
    print("=" * 60)

    best_M = None
    best_score = -np.inf
    best_seg = None

    for M in M_candidates:
        seg = KMeansSeg(M, random_state=random_state)
        seg.fit(X_pilot)

        score = kmeans_silhouette_score(seg_model=seg, X_pilot=X_pilot)

        if score > best_score:
            best_score = score
            best_M = M
            best_seg = seg

    seg_labels_pilot = best_seg.assign(X_pilot)
    return best_seg, seg_labels_pilot, best_M

def _prepare_dams_kfold(
    X_pilot,
    D_pilot,
    y_pilot,
    Gamma_pilot,
    *,
    n_folds: int,
    cv_random_state: int,
):
    """Validate pilot arrays and return (X, D, y, Gamma, fold_splits)."""
    X_pilot = np.asarray(X_pilot)
    D_pilot = np.asarray(D_pilot).astype(int)
    y_pilot = np.asarray(y_pilot, dtype=float)
    Gamma_pilot = np.asarray(Gamma_pilot, dtype=float)

    n_pilot = X_pilot.shape[0]
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
    if n_pilot < n_folds:
        raise ValueError(
            f"Cannot run {n_folds}-fold DAMS with only {n_pilot} pilot samples."
        )
    if D_pilot.shape[0] != n_pilot or y_pilot.shape[0] != n_pilot:
        raise ValueError("X_pilot, D_pilot, y_pilot must have the same length.")
    if Gamma_pilot.shape[0] != n_pilot:
        raise ValueError(
            f"Gamma_pilot length {Gamma_pilot.shape[0]} != n_pilot {n_pilot}."
        )

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cv_random_state)
    return X_pilot, D_pilot, y_pilot, Gamma_pilot, list(kf.split(X_pilot))


def run_kmeans_dams_segmentation(
    X_pilot,
    D_pilot,
    y_pilot,
    Gamma_pilot,
    M_candidates,
    random_state,
    value_type_dams,
    action_method,
    n_folds: int = 5,
    cv_random_state: int = 0,
):
    print("\n" + "=" * 60)
    print("KMeans_DAMS - selecting optimal K via K-fold DAMS")
    print("=" * 60)

    X_pilot, D_pilot, y_pilot, Gamma_pilot, fold_splits = _prepare_dams_kfold(
        X_pilot,
        D_pilot,
        y_pilot,
        Gamma_pilot,
        n_folds=n_folds,
        cv_random_state=cv_random_state,
    )

    best_M = None
    best_score = -np.inf

    print(f"\nTesting M candidates with {n_folds}-fold DAMS: {list(M_candidates)}")
    for M in M_candidates:
        fold_scores = []
        for tr_idx, va_idx in fold_splits:
            X_tr, D_tr, y_tr, Gamma_tr = (
                X_pilot[tr_idx],
                D_pilot[tr_idx],
                y_pilot[tr_idx],
                Gamma_pilot[tr_idx],
            )
            seg = KMeansSeg(M, random_state=random_state)
            seg.fit(X_tr)
            action = estimate_segment_policy(
                X_tr,
                y_tr,
                D_tr,
                seg.assign(X_tr),
                method=action_method,
                Gamma=Gamma_tr,
            )
            fold_scores.append(
                dams_score(
                    seg_model=seg,
                    X_val=X_pilot[va_idx],
                    D_val=D_pilot[va_idx],
                    y_val=y_pilot[va_idx],
                    Gamma_val=Gamma_pilot[va_idx],
                    action=action,
                    value_type_dams=value_type_dams,
                )
            )

        score_M = float(np.mean(fold_scores))
        print(
            f"  M={M}: mean DAMS={score_M:.6f} "
            f"(folds={[f'{s:.6f}' for s in fold_scores]})"
        )
        if score_M >= best_score:  # tie break by larger M
            best_score = score_M
            best_M = M

    print(
        f"\n✓ KMeans_DAMS: selected M = {best_M} with "
        f"{n_folds}-fold mean DAMS-score = {best_score:.6f}\n"
    )

    final_seg = KMeansSeg(best_M, random_state=random_state)
    final_seg.fit(X_pilot)
    seg_labels_pilot = final_seg.assign(X_pilot)

    return final_seg, seg_labels_pilot, best_M


# =========================================================
# 3. GMM segmentation + BIC 选 K
# =========================================================
def run_gmm_segmentation(X_pilot, M_candidates, random_state):
    print("\n" + "=" * 60)
    print("GMM - selecting optimal M via BIC")
    print("=" * 60)

    best_M = None
    best_bic = np.inf
    best_seg = None

    for M in M_candidates:
        seg = GMMSeg(M, random_state=random_state)
        seg.fit(X_pilot)

        bic = seg.model.bic(X_pilot)

        if bic < best_bic:
            best_bic = bic
            best_M = M
            best_seg = seg

    seg_labels_pilot = best_seg.assign(X_pilot)
    return best_seg, seg_labels_pilot, best_M


def run_gmm_dams_segmentation(
    X_pilot,
    D_pilot,
    y_pilot,
    Gamma_pilot,
    M_candidates,
    random_state,
    value_type_dams,
    action_method,
    n_folds: int = 5,
    cv_random_state: int = 0,
):
    print("\n" + "=" * 60)
    print("GMM_DAMS - selecting optimal K via K-fold DAMS")
    print("=" * 60)

    X_pilot, D_pilot, y_pilot, Gamma_pilot, fold_splits = _prepare_dams_kfold(
        X_pilot,
        D_pilot,
        y_pilot,
        Gamma_pilot,
        n_folds=n_folds,
        cv_random_state=cv_random_state,
    )

    best_M = None
    best_score = -np.inf

    print(f"\nTesting M candidates with {n_folds}-fold DAMS: {list(M_candidates)}")
    for M in M_candidates:
        fold_scores = []
        for tr_idx, va_idx in fold_splits:
            X_tr, D_tr, y_tr, Gamma_tr = (
                X_pilot[tr_idx],
                D_pilot[tr_idx],
                y_pilot[tr_idx],
                Gamma_pilot[tr_idx],
            )
            seg = GMMSeg(M, random_state=random_state)
            seg.fit(X_tr)
            action = estimate_segment_policy(
                X_tr,
                y_tr,
                D_tr,
                seg.assign(X_tr),
                method=action_method,
                Gamma=Gamma_tr,
            )
            fold_scores.append(
                dams_score(
                    seg_model=seg,
                    X_val=X_pilot[va_idx],
                    D_val=D_pilot[va_idx],
                    y_val=y_pilot[va_idx],
                    Gamma_val=Gamma_pilot[va_idx],
                    action=action,
                    value_type_dams=value_type_dams,
                )
            )

        score_M = float(np.mean(fold_scores))
        print(
            f"  M={M}: mean DAMS={score_M:.6f} "
            f"(folds={[f'{s:.6f}' for s in fold_scores]})"
        )
        if score_M >= best_score:  # tie break by larger M
            best_score = score_M
            best_M = M

    print(
        f"\n✓ GMM_DAMS: selected M = {best_M} with "
        f"{n_folds}-fold mean DAMS-score = {best_score:.6f}\n"
    )

    final_seg = GMMSeg(best_M, random_state=random_state)
    final_seg.fit(X_pilot)
    seg_labels_pilot = final_seg.assign(X_pilot)

    return final_seg, seg_labels_pilot, best_M


# =========================================================
# 4. DAST + DAMS（M selection）
# =========================================================
def dast_candidate_thresholds(X_pilot, bins: int = 200) -> dict:
    """Midpoint thresholds per feature for DAST splitting."""
    d_full = X_pilot.shape[1]
    H_full = {}
    for j in range(d_full):
        col = X_pilot[:, j]
        unique_values = np.unique(col)
        if len(unique_values) > bins + 1:
            qs = np.linspace(0, 1, num=bins + 1)
            grid = np.unique(np.quantile(col, qs))
        else:
            grid = unique_values
        if len(grid) > 1:
            H_full[j] = (grid[:-1] + grid[1:]) / 2.0
        else:
            H_full[j] = grid
    print(f"Candidate thresholds computed for {d_full} features.")
    return H_full


def select_dast_M_via_dams(
    X_pilot,
    D_pilot,
    y_pilot,
    Gamma_pilot,
    M_candidates,
    min_leaf_size,
    value_type_dast,
    value_type_dams,
    action_method,
    n_folds: int = 5,
    cv_random_state: int = 0,
):
    """
    Pick M by K-fold DAMS on the full pilot set.

    For each candidate M:
      - fit DAST + segment policy on each training fold
      - score with dams_score on the held-out fold
      - average fold scores
    Tie-break: larger M wins (same as the previous single hold-out DAMS).
    Threshold grid H_full is computed once on the full pilot (unchanged).
    """
    X_pilot, D_pilot, y_pilot, Gamma_pilot, fold_splits = _prepare_dams_kfold(
        X_pilot,
        D_pilot,
        y_pilot,
        Gamma_pilot,
        n_folds=n_folds,
        cv_random_state=cv_random_state,
    )

    H_full = dast_candidate_thresholds(X_pilot)

    best_M = None
    best_score = -np.inf

    print(
        f"\nTesting M candidates with {n_folds}-fold DAMS: {list(M_candidates)}"
    )
    for M in M_candidates:
        fold_scores = []
        fold_leaves = []
        for fold_id, (tr_idx, va_idx) in enumerate(fold_splits):
            X_tr = X_pilot[tr_idx]
            D_tr = D_pilot[tr_idx]
            y_tr = y_pilot[tr_idx]
            Gamma_tr = Gamma_pilot[tr_idx]

            X_va = X_pilot[va_idx]
            D_va = D_pilot[va_idx]
            y_va = y_pilot[va_idx]
            Gamma_va = Gamma_pilot[va_idx]

            tree = DASTree(
                x=X_tr,
                y=y_tr,
                D=D_tr,
                gamma=Gamma_tr,
                candidate_thresholds=H_full,
                min_leaf_size=min_leaf_size,
                value_type_dast=value_type_dast,
                action_method=action_method,
            )
            tree.build(M)
            fold_leaves.append(len(tree._get_leaf_nodes()))

            labels_train = tree.assign(X_tr)
            action_M = estimate_segment_policy(
                X_tr,
                y_tr,
                D_tr,
                labels_train,
                method=action_method,
                Gamma=Gamma_tr,
            )
            score_fold = dams_score(
                seg_model=tree,
                X_val=X_va,
                D_val=D_va,
                y_val=y_va,
                Gamma_val=Gamma_va,
                action=action_M,
                value_type_dams=value_type_dams,
            )
            fold_scores.append(float(score_fold))

        score_M = float(np.mean(fold_scores))
        print(
            f"  M={M}: mean DAMS={score_M:.6f} "
            f"(folds={[f'{s:.6f}' for s in fold_scores]}, "
            f"leaves={fold_leaves})"
        )
        if score_M >= best_score:  # tie break by larger M
            best_score = score_M
            best_M = M

    print(
        f"\n✓ DAST: selected M = {best_M} with "
        f"{n_folds}-fold mean DAMS-score = {best_score:.6f}\n"
    )
    return best_M, best_score, H_full


def fit_dast_pilot_tree_at_M(
    X_pilot,
    D_pilot,
    y_pilot,
    Gamma_pilot,
    M,
    H_full,
    min_leaf_size,
    value_type_dast,
    action_method,
):
    tree = DASTree(
        x=X_pilot,
        y=y_pilot,
        D=D_pilot,
        gamma=Gamma_pilot,
        candidate_thresholds=H_full,
        min_leaf_size=min_leaf_size,
        value_type_dast=value_type_dast,
        action_method=action_method,
    )
    tree.build(M)
    seg_labels_pilot = tree.assign(X_pilot)
    action_pilot = estimate_segment_policy(
        X_pilot,
        y_pilot,
        D_pilot,
        seg_labels_pilot,
        method=action_method,
        Gamma=Gamma_pilot,
    )
    return tree, seg_labels_pilot, action_pilot


def run_dast_all_M_curves(
    X_pilot,
    D_pilot,
    y_pilot,
    X_impl,
    Gamma_pilot,
    M_candidates,
    min_leaf_size,
    value_type_dast,
    value_type_dams,
    action_method,
    n_folds: int = 5,
    cv_random_state: int = 0,
):
    """
    K-fold DAMS-select best_M, then fit one pilot tree per candidate M for OPE curves.

    Returns best_M, dict[M] -> (seg_labels_impl, action_per_segment).
    """
    best_M, best_score, H_full = select_dast_M_via_dams(
        X_pilot,
        D_pilot,
        y_pilot,
        Gamma_pilot,
        M_candidates,
        min_leaf_size,
        value_type_dast,
        value_type_dams,
        action_method,
        n_folds=n_folds,
        cv_random_state=cv_random_state,
    )
    pilot_by_M = {}
    for M in M_candidates:
        tree, _, action_pilot = fit_dast_pilot_tree_at_M(
            X_pilot,
            D_pilot,
            y_pilot,
            Gamma_pilot,
            M,
            H_full,
            min_leaf_size,
            value_type_dast,
            action_method,
        )
        seg_labels_impl = tree.assign(X_impl)
        pilot_by_M[M] = (seg_labels_impl, action_pilot)
        print(
            f"  Built pilot tree for M={M}: "
            f"actual leaves = {len(tree._get_leaf_nodes())}"
        )
    return best_M, pilot_by_M


def run_dast_dams(
    X_pilot,
    D_pilot,
    y_pilot,
    Gamma_pilot,
    M_candidates,
    min_leaf_size,
    value_type_dast,
    value_type_dams,
    action_method,
    n_folds: int = 5,
    cv_random_state: int = 0,
):
    best_M, _, H_full = select_dast_M_via_dams(
        X_pilot,
        D_pilot,
        y_pilot,
        Gamma_pilot,
        M_candidates,
        min_leaf_size,
        value_type_dast,
        value_type_dams,
        action_method,
        n_folds=n_folds,
        cv_random_state=cv_random_state,
    )
    tree_final, seg_labels_pilot, action_full_pilot = fit_dast_pilot_tree_at_M(
        X_pilot,
        D_pilot,
        y_pilot,
        Gamma_pilot,
        best_M,
        H_full,
        min_leaf_size,
        value_type_dast,
        action_method,
    )
    return tree_final, seg_labels_pilot, best_M, action_full_pilot




def run_clr_segmentation(
    X_pilot,
    D_pilot,
    y_pilot,
    M_candidates,
    random_state,
):
    best_M = None
    best_score = np.inf
    best_seg = None
    best_labels = None

    for M in M_candidates:
        seg = CLRSeg(
            n_segments=M,
            random_state=random_state,
        )
        seg.fit(X_pilot, D_pilot, y_pilot)
        bic = clr_bic_score(seg, X_pilot, D_pilot, y_pilot)
        print(f"CLR M={M} BIC={bic:.3f}")

        if bic < best_score and bic > -np.inf:
            best_score = bic
            best_M = M
            best_seg = seg
            best_labels = seg.assign(X_pilot)

    return best_seg, best_labels, best_M

def run_clr_dams_segmentation(
    X_pilot,
    D_pilot,
    y_pilot,
    Gamma_pilot,
    M_candidates,
    random_state,
    value_type_dams,
    action_method,
    n_folds: int = 5,
    cv_random_state: int = 0,
):
    print("\n" + "=" * 60)
    print("CLR_DAMS - selecting optimal K via K-fold DAMS")
    print("=" * 60)

    X_pilot, D_pilot, y_pilot, Gamma_pilot, fold_splits = _prepare_dams_kfold(
        X_pilot,
        D_pilot,
        y_pilot,
        Gamma_pilot,
        n_folds=n_folds,
        cv_random_state=cv_random_state,
    )

    best_M = None
    best_score = -np.inf

    print(f"\nTesting M candidates with {n_folds}-fold DAMS: {list(M_candidates)}")
    for M in M_candidates:
        fold_scores = []
        for tr_idx, va_idx in fold_splits:
            X_tr, D_tr, y_tr, Gamma_tr = (
                X_pilot[tr_idx],
                D_pilot[tr_idx],
                y_pilot[tr_idx],
                Gamma_pilot[tr_idx],
            )
            seg = CLRSeg(
                n_segments=M,
                random_state=random_state,
            )
            seg.fit(X_tr, D_tr, y_tr)
            action = estimate_segment_policy(
                X_tr,
                y_tr,
                D_tr,
                seg.assign(X_tr),
                method=action_method,
                Gamma=Gamma_tr,
            )
            fold_scores.append(
                dams_score(
                    seg_model=seg,
                    X_val=X_pilot[va_idx],
                    D_val=D_pilot[va_idx],
                    y_val=y_pilot[va_idx],
                    Gamma_val=Gamma_pilot[va_idx],
                    action=action,
                    value_type_dams=value_type_dams,
                )
            )

        score_M = float(np.mean(fold_scores))
        print(
            f"  M={M}: mean DAMS={score_M:.6f} "
            f"(folds={[f'{s:.6f}' for s in fold_scores]})"
        )
        if score_M >= best_score:  # tie break by larger M
            best_score = score_M
            best_M = M

    print(
        f"\n✓ CLR_DAMS: selected M = {best_M} with "
        f"{n_folds}-fold mean DAMS-score = {best_score:.6f}\n"
    )

    final_seg = CLRSeg(
        n_segments=best_M,
        random_state=random_state,
    )
    final_seg.fit(X_pilot, D_pilot, y_pilot)
    seg_labels_pilot = final_seg.assign(X_pilot)

    return final_seg, seg_labels_pilot, best_M


def run_mst_dams(
    X_pilot,
    D_pilot,
    y_pilot,
    Gamma_pilot,
    M_candidates,
    min_leaf_size,
    value_type_dams,
    action_method,
    n_folds: int = 5,
    cv_random_state: int = 0,
):
    X_pilot, D_pilot, y_pilot, Gamma_pilot, fold_splits = _prepare_dams_kfold(
        X_pilot,
        D_pilot,
        y_pilot,
        Gamma_pilot,
        n_folds=n_folds,
        cv_random_state=cv_random_state,
    )

    d = X_pilot.shape[1]

    # candidate thresholds: same construction as before (full pilot)
    d_full = X_pilot.shape[1]
    bins = 200
    H_full = {}

    for j in range(d_full):
        col = X_pilot[:, j]
        unique_values = np.unique(col)

        if len(unique_values) <= 1:
            H_full[j] = unique_values
        else:
            if len(unique_values) > bins + 1:
                qs = np.linspace(0, 1, num=bins + 1)
                grid = np.quantile(col, qs)
                grid = np.unique(grid)
            else:
                grid = unique_values

            H_full[j] = grid

    print(f"Candidate thresholds computed for {d} features.")

    # Per-fold cache: depth -> unpruned MSTree built on that fold's train set
    tree_cache_by_fold = {fold_id: {} for fold_id in range(len(fold_splits))}

    best_M = None
    best_score = -np.inf

    print(f"\nTesting M candidates for MST with {n_folds}-fold DAMS: {list(M_candidates)}")
    for M in M_candidates:
        if M == 1:
            depth = 0
        else:
            depth = int(np.ceil(np.log2(M)))

        fold_scores = []
        for fold_id, (tr_idx, va_idx) in enumerate(fold_splits):
            X_tr = X_pilot[tr_idx]
            D_tr = D_pilot[tr_idx]
            y_tr = y_pilot[tr_idx]
            Gamma_tr = Gamma_pilot[tr_idx]

            fold_cache = tree_cache_by_fold[fold_id]
            if depth not in fold_cache:
                tree_original = MSTree(
                    x=X_tr,
                    y=y_tr,
                    D=D_tr,
                    candidate_thresholds=H_full,
                    min_leaf_size=min_leaf_size,
                    max_depth=depth,
                    epsilon=0.0,
                )
                tree_original.build()
                fold_cache[depth] = tree_original

            tree = fold_cache[depth].copy()
            tree.prune_to_M(M)

            labels_train = tree.assign(X_tr)
            action_M = estimate_segment_policy(
                X_tr,
                y_tr,
                D_tr,
                labels_train,
                method=action_method,
                Gamma=Gamma_tr,
            )
            fold_scores.append(
                float(
                    dams_score(
                        seg_model=tree,
                        X_val=X_pilot[va_idx],
                        D_val=D_pilot[va_idx],
                        y_val=y_pilot[va_idx],
                        Gamma_val=Gamma_pilot[va_idx],
                        action=action_M,
                        value_type_dams=value_type_dams,
                    )
                )
            )

        score_M = float(np.mean(fold_scores))
        print(
            f"  M={M}: mean DAMS={score_M:.6f} "
            f"(folds={[f'{s:.6f}' for s in fold_scores]})"
        )
        if score_M >= best_score:  # tie break by larger M
            best_score = score_M
            best_M = M

    print(
        f"\n✓ MST: selected M = {best_M} with "
        f"{n_folds}-fold mean DAMS-score = {best_score:.6f}\n"
    )

    # Refit on full pilot
    print("Reusing candidate thresholds H_full on full pilot...")

    tree_final = MSTree(
        x=X_pilot,
        y=y_pilot,
        D=D_pilot,
        candidate_thresholds=H_full,
        min_leaf_size=min_leaf_size,
        max_depth=0 if best_M == 1 else int(np.ceil(np.log2(best_M))),
        epsilon=0.0,
    )
    tree_final.build()
    tree_final.prune_to_M(best_M)
    seg_labels_pilot = tree_final.assign(X_pilot)

    return tree_final, seg_labels_pilot, best_M
