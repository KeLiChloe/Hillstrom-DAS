# segmentation.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from outcome_model import predict_mu
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

def run_kmeans_dams_segmentation(X_pilot,
                                 X_train, D_train, y_train,
                                 X_val, D_val, y_val,
                                 Gamma_val,
                                 M_candidates,
                                 random_state,
                                 value_type_dams,
                                 action_method,
                                 Gamma_train=None):
    print("\n" + "=" * 60)
    print("KMeans_DAMS - selecting optimal K")
    print("=" * 60)

    best_M = None
    best_score = -np.inf

    for M in M_candidates:
        seg = KMeansSeg(M, random_state=random_state)
        seg.fit(X_train)
        action = estimate_segment_policy(
            X_train, y_train, D_train, seg.assign(X_train),
            method=action_method, Gamma=Gamma_train,
        )

        score = dams_score(seg_model=seg,
                           X_val=X_val, D_val=D_val, y_val=y_val,
                           Gamma_val=Gamma_val,
                           action=action,
                           value_type_dams=value_type_dams)
        

        if score > best_score:
            best_score = score
            best_M = M


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


def run_gmm_dams_segmentation(X_pilot, 
                             X_train, D_train, y_train,
                             X_val, D_val, y_val,
                                Gamma_val,
                              M_candidates,
                              random_state,
                              value_type_dams,
                              action_method,
                              Gamma_train=None):
    print("\n" + "=" * 60)
    print("GMM_DAMS - selecting optimal K")
    print("=" * 60)
    
    
    best_M = None
    best_score = -np.inf

    for M in M_candidates:
        seg = GMMSeg(M, random_state=random_state)
        seg.fit(X_train)
        action = estimate_segment_policy(
            X_train, y_train, D_train, seg.assign(X_train),
            method=action_method, Gamma=Gamma_train,
        )

        score = dams_score(seg_model=seg, 
                           X_val=X_val,
                           D_val=D_val, 
                           y_val=y_val,
                           Gamma_val=Gamma_val,
                           action=action,
                           value_type_dams=value_type_dams)
        

        if score > best_score:
            best_score = score
            best_M = M


    final_seg = GMMSeg(best_M, random_state=random_state)
    final_seg.fit(X_pilot)
    seg_labels_pilot = final_seg.assign(X_pilot)

    return final_seg, seg_labels_pilot, best_M

# =========================================================
# 4. DAST + DAMS（M selection）
# =========================================================
def run_dast_dams(
    X_pilot, D_pilot,y_pilot,
    X_train, D_train, y_train,
    X_val, D_val, y_val,
    Gamma_pilot,
    Gamma_train,
    Gamma_val,
    M_candidates,
    min_leaf_size,
    value_type_dast,
    value_type_dams,
    action_method,
):

    d_full = X_pilot.shape[1]

    # Generate candidate thresholds (midpoints between unique values)
    bins = 200
    H_full = {}

    for j in range(d_full):
        col = X_pilot[:, j]
        unique_values = np.unique(col)

        if len(unique_values) > bins + 1:
            qs = np.linspace(0, 1, num=bins + 1)
            grid = np.quantile(col, qs)
            grid = np.unique(grid)
        else:
            grid = unique_values

        if len(grid) > 1:
            H_full[j] = (grid[:-1] + grid[1:]) / 2.0
        else:
            H_full[j] = grid

    print(f"Candidate thresholds computed for {d_full} features.")

    best_M = None
    best_score = -np.inf

    print(f"\nTesting M candidates: {list(M_candidates)}")
    for M in M_candidates:
        # Best-first growth: build a fresh tree directly to M leaves
        tree = DASTree(
            x=X_train,
            y=y_train,
            D=D_train,
            gamma=Gamma_train,
            candidate_thresholds=H_full,
            min_leaf_size=min_leaf_size,
            value_type_dast=value_type_dast,
            action_method=action_method,
        )
        tree.build(M)
        actual_leaves = len(tree._get_leaf_nodes())
        print(f"  Built tree for M={M}: actual leaves = {actual_leaves}")

        # segment labels on train + segment-level policy
        labels_train = tree.assign(X_train)
        action_M = estimate_segment_policy(
            X_train, y_train, D_train, labels_train,
            method=action_method, Gamma=Gamma_train,
        )

        # DAMS scoring on validation
        score_M = dams_score(
            seg_model=tree,
            X_val=X_val,
            D_val=D_val,
            y_val=y_val,
            Gamma_val=Gamma_val,
            action=action_M,
            value_type_dams=value_type_dams,
        )

        if score_M >= best_score: # tie break by larger M (more segments)
            best_score = score_M
            best_M = M

    if best_M <= 4:
        best_M = 8  # Avoid excessive pruning causing excessive variance

    print(f"\n✓ DAST: selected M = {best_M} with DAMS-score = {best_score:.6f}\n")

    # Re-fit final tree on full pilot data

    tree_final = DASTree(
        x=X_pilot,
        y=y_pilot,
        D=D_pilot,
        gamma=Gamma_pilot,
        candidate_thresholds=H_full,
        min_leaf_size=min_leaf_size,
        value_type_dast=value_type_dast,
        action_method=action_method,
    )
    tree_final.build(best_M)
    seg_labels_pilot = tree_final.assign(X_pilot)
    action_full_pilot = estimate_segment_policy(
        X_pilot, y_pilot, D_pilot, seg_labels_pilot,
        method=action_method, Gamma=Gamma_pilot,
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

def run_clr_dams_segmentation(X_pilot, D_pilot,y_pilot,
                                X_train, D_train, y_train,
                                X_val, D_val, y_val,
                                Gamma_val,
                              M_candidates,
                              random_state,
                              value_type_dams,
                              action_method,
                              Gamma_train=None):
    print("\n" + "=" * 60)
    print("CLR_DAMS - selecting optimal K")
    print("=" * 60)
    

    
    best_M = None
    best_score = -np.inf
    
    for M in M_candidates:
        seg = CLRSeg(
            n_segments=M,
            random_state=random_state,
        )
        seg.fit(X_train, D_train, y_train)
        action = estimate_segment_policy(
            X_train, y_train, D_train, seg.assign(X_train),
            method=action_method, Gamma=Gamma_train,
        )

        score = dams_score(seg_model=seg, X_val=X_val,
                            D_val=D_val, y_val=y_val,
                            Gamma_val=Gamma_val,
                            action=action,
                            value_type_dams=value_type_dams)

        if score > best_score:
            best_score = score
            best_M = M


    final_seg = CLRSeg(
        n_segments=best_M,
        random_state=random_state,
    )
    final_seg.fit(X_pilot, D_pilot, y_pilot)
    seg_labels_pilot = final_seg.assign(X_pilot)

    return final_seg, seg_labels_pilot, best_M



def run_mst_dams(
    X_pilot, D_pilot,y_pilot,
    X_train, D_train, y_train,
    X_val, D_val, y_val,
    Gamma_val,
    M_candidates,
    min_leaf_size,
    value_type_dams,
    action_method,
    Gamma_train=None,
):

    d = X_pilot.shape[1]


    # --------------------------------------------------
    # candidate thresholds: 跟 run_dast_dams 一样
    # --------------------------------------------------
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

    # --------------------------------------------------
    # DAMS: loop over M
    # --------------------------------------------------
    best_M = None
    best_score = -np.inf
    tree_cache = {}   # depth -> built MSTree (unpruned)

    print(f"\nTesting M candidates for MST: {list(M_candidates)}")
    for M in M_candidates:
        # ===== 修复：正确计算 max_depth，并添加 buffer =====
        if M == 1:
            depth = 0
        else:
            depth = int(np.ceil(np.log2(M))) 

        # --------------------------------------------------
        # 1) 复用相同 depth 的树 —— 只 build 一次
        # --------------------------------------------------
        if depth not in tree_cache:
            tree_original = MSTree(
                x=X_train,
                y=y_train,
                D=D_train,
                candidate_thresholds=H_full,
                min_leaf_size=min_leaf_size,
                max_depth=depth,
                epsilon=0.0,
            )
            tree_original.build()
            actual_leaves_mst = len(tree_original._get_leaf_nodes())
            tree_cache[depth] = tree_original  # 保存原始树到 cache
        
        # ⚠️ 关键修复：每次都从 cache 中 copy，避免 prune 操作修改 cache 中的原始树
        tree = tree_cache[depth].copy()
        tree.prune_to_M(M)

        # segment labels on train + segment-level policy
        labels_train = tree.assign(X_train)
        action_M = estimate_segment_policy(
            X_train, y_train, D_train, labels_train,
            method=action_method, Gamma=Gamma_train,
        )

        # DAMS scoring on validation (dual) —— 跟 DAST 完全一样
        score_M = dams_score(
            seg_model=tree,
            X_val=X_val,
            D_val=D_val,
            y_val=y_val,
            Gamma_val=Gamma_val,
            action=action_M,
            value_type_dams=value_type_dams,
        )

        if score_M > best_score:
            best_score = score_M
            best_M = M


    # --------------------------------------------------
    # 用 full pilot 重新 fit
    # --------------------------------------------------ß
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
