# segmentation.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from outcome_model import predict_mu
from scoring import dams_score, kmeans_silhouette_score
from dast import DASTree
from estimation import estimate_segment_policy  
from clr import CLRSeg, clr_bic_score
from policytree import PolicyTreeSeg, _fit_policytree_with_grf, policytree_post_prune_tree
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
        print(f"  KMeans K={M} score={score:.4f}")

        if score > best_score:
            best_score = score
            best_M = M
            best_seg = seg

    print(f"\n✓ KMeans: selected K = {best_M} with score = {best_score:.4f}\n")
    seg_labels_pilot = best_seg.assign(X_pilot)
    return best_seg, seg_labels_pilot, best_M

def run_kmeans_dams_segmentation(X_pilot,
                                 X_train, D_train, y_train,
                                 X_val, D_val, y_val,
                                 Gamma_val,
                                 M_candidates,
                                 random_state):
    print("\n" + "=" * 60)
    print("KMeans_DAMS - selecting optimal K")
    print("=" * 60)

    best_M = None
    best_score = -np.inf

    for M in M_candidates:
        seg = KMeansSeg(M, random_state=random_state)
        seg.fit(X_train)
        action = estimate_segment_policy(
            X_train, y_train, D_train, seg.assign(X_train)
        )

        score = dams_score(seg_model = seg, 
                           X_val = X_val, D_val = D_val, y_val = y_val, 
                           Gamma_val = Gamma_val, 
                           action = action)
        
        print(f"  KMeans_DAMS M={M} score={score:.4f}")

        if score > best_score:
            best_score = score
            best_M = M

    print(f"\n✓ KMeans_DAMS: selected M = {best_M} with score = {best_score:.4f}\n")

    final_seg = KMeansSeg(best_M, random_state=random_state)
    final_seg.fit(X_pilot)
    seg_labels_pilot = final_seg.assign(X_pilot)

    return final_seg, seg_labels_pilot, best_M



# =========================================================
# 3. GMM segmentation + BIC 选 K
# =========================================================
def run_gmm_segmentation(X_pilot, M_candidates, random_state):
    print("\n" + "=" * 60)
    print("STEP 4b: GMM - selecting optimal M via BIC")
    print("=" * 60)

    best_M = None
    best_bic = np.inf
    best_seg = None

    for M in M_candidates:
        seg = GMMSeg(M, random_state=random_state)
        seg.fit(X_pilot)

        bic = seg.model.bic(X_pilot)
        print(f"  GMM M={M} BIC={bic:.1f}")

        if bic < best_bic:
            best_bic = bic
            best_M = M
            best_seg = seg

    print(f"\n✓ GMM: selected M = {best_M} with BIC = {best_bic:.1f}\n")
    seg_labels_pilot = best_seg.assign(X_pilot)
    return best_seg, seg_labels_pilot, best_M


def run_gmm_dams_segmentation(X_pilot, 
                             X_train, D_train, y_train,
                             X_val, D_val, y_val,
                                Gamma_val,
                              M_candidates,
                              random_state):
    print("\n" + "=" * 60)
    print("GMM_DAMS - selecting optimal K")
    print("=" * 60)
    
    
    best_M = None
    best_score = -np.inf

    for M in M_candidates:
        seg = GMMSeg(M, random_state=random_state)
        seg.fit(X_train)
        action = estimate_segment_policy(
            X_train, y_train, D_train, seg.assign(X_train)
        )

        score = dams_score(seg_model=seg, 
                           X_val=X_val,
                           D_val=D_val, 
                           y_val=y_val,
                           Gamma_val=Gamma_val,
                           action=action)
        
        print(f"  GMM_DAMS M={M} score={score:.4f}")

        if score > best_score:
            best_score = score
            best_M = M

    print(f"\n✓ GMM_DAMS: selected M = {best_M} with score = {best_score:.4f}\n")

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
    value_type,
):
    print("\n" + "=" * 60)
    print("STEP 5: DAST - selecting optimal M via DAMS")
    print("=" * 60)

    d = X_pilot.shape[1]

    # candidate thresholds
    d_full = X_pilot.shape[1]
    
    # Generate candidate thresholds (midpoints between unique values)

    bins = 200
    H_full = {}

    for j in range(d_full):
        col = X_pilot[:, j]
        # 去掉nan的话可以先 col = col[~np.isnan(col)]
        unique_values = np.unique(col)


        # 如果 unique 太多，只取 K+1 个“代表点”，再在中间算 midpoints
        if len(unique_values) > bins + 1:
            # 取 K+1 个分位数，比如 [0, 1/K, 2/K, ..., 1]
            qs = np.linspace(0, 1, num=bins+1)
            # 用 quantile 近似 unique-values 的分布
            grid = np.quantile(col, qs)
            grid = np.unique(grid)  # 可能有重复
        else:
            grid = unique_values

        if len(grid) > 1:
            H_full[j] = (grid[:-1] + grid[1:]) / 2.0
        else:
            H_full[j] = grid


    print(f"Candidate thresholds computed for {d} features.")

    best_M = None
    best_score = -np.inf

    tree_cache = {}   # key = depth, value = built tree object


    print(f"\nTesting M candidates: {list(M_candidates)}")
    for M in M_candidates:
        # ===== 修复：正确计算 max_depth，并添加 buffer =====
        if M == 1:
            depth = 0
        else:
            depth = int(np.ceil(np.log2(M)))
        
        # ====== 1) 复用相同 depth 的树 ======
        if depth not in tree_cache:
            tree_original = DASTree(
                x=X_train,
                y=y_train,
                D=D_train,
                gamma=Gamma_train,
                candidate_thresholds=H_full,
                min_leaf_size=min_leaf_size,
                max_depth=depth,
                value_type=value_type,
            )
            tree_original.build()
            actual_leaves = len(tree_original._get_leaf_nodes())
            print(f"  Built tree for M={M} (max_depth={depth}): actual leaves = {actual_leaves}")
            tree_cache[depth] = tree_original  # 保存原始树到 cache
        
        # ⚠️ 关键修复：每次都从 cache 中 copy，避免 prune 操作修改 cache 中的原始树
        tree = tree_cache[depth].copy()
        tree.prune_to_M(M)

        # segment labels on train + segment-level policy (diff-in-means on y)
        labels_train = tree.assign(X_train)
        action_M = estimate_segment_policy(
            X_train, y_train, D_train, labels_train
        )

        # DAMS scoring on validation (dual)
        score_M = dams_score(
            seg_model=tree,
            X_val=X_val,
            D_val=D_val,
            y_val=y_val,
            Gamma_val=Gamma_val,
            action=action_M,
        )
        print(f"  DAST M={M} DAMS-score={score_M:.6f}")

        if score_M >= best_score:
            best_score = score_M
            best_M = M

    print(f"\n✓ DAST: selected M = {best_M} with DAMS-score = {best_score:.6f}\n")

    # 用 full pilot 重新 fit
    print("\n" + "=" * 60)
    print("STEP 6: Fitting final DAST on full pilot")
    print("=" * 60)

    
    tree_final = DASTree(
        x=X_pilot,
        y=y_pilot,
        D=D_pilot,
        gamma=Gamma_pilot,
        candidate_thresholds=H_full,
        min_leaf_size=min_leaf_size,
        max_depth=0 if best_M == 1 else int(np.ceil(np.log2(best_M))),
    )
    tree_final.build()
    tree_final.prune_to_M(best_M)
    seg_labels_pilot = tree_final.assign(X_pilot)
    action_full_pilot = estimate_segment_policy(
        X_pilot, y_pilot, D_pilot, seg_labels_pilot
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

    print(f"\n✓ CLR selected M={best_M} with BIC={best_score:.3f}\n")
    return best_seg, best_labels, best_M

def run_clr_dams_segmentation(X_pilot, D_pilot,y_pilot,
                                X_train, D_train, y_train,
                                X_val, D_val, y_val,
                                Gamma_val,
                              M_candidates,
                              random_state):
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
            X_train, y_train, D_train, seg.assign(X_train)
        )

        score = dams_score(seg_model=seg, X_val=X_val,
                            D_val=D_val, y_val=y_val,
                            Gamma_val=Gamma_val,
                            action=action)
        print(f"  CLR_DAMS M={M} score={score:.4f}")

        if score > best_score:
            best_score = score
            best_M = M

    print(f"\n✓ CLR_DAMS: selected M = {best_M} with score = {best_score:.4f}\n")

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
    min_leaf_size
    
):
    print("\n" + "=" * 60)
    print("STEP 5 (MST): selecting optimal M via DAMS (residual-based splits)")
    print("=" * 60)

    d = X_pilot.shape[1]


    # --------------------------------------------------
    # candidate thresholds: 跟 run_dast_dams 一样
    # --------------------------------------------------
    d_full = X_pilot.shape[1]
    bins = 100
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

            if len(grid) > 1:
                H_full[j] = (grid[:-1] + grid[1:]) / 2.0
            else:
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
            print(f"  Built MST for M={M} (max_depth={depth}): actual leaves = {actual_leaves_mst}")
            tree_cache[depth] = tree_original  # 保存原始树到 cache
        
        # ⚠️ 关键修复：每次都从 cache 中 copy，避免 prune 操作修改 cache 中的原始树
        tree = tree_cache[depth].copy()
        tree.prune_to_M(M)

        # segment labels on train + segment-level policy (diff-in-means on y)
        labels_train = tree.assign(X_train)
        action_M = estimate_segment_policy(
            X_train, y_train, D_train, labels_train
        )

        # DAMS scoring on validation (dual) —— 跟 DAST 完全一样
        score_M = dams_score(
            seg_model=tree,
            X_val=X_val,
            D_val=D_val,
            y_val=y_val,
            Gamma_val=Gamma_val,
            action=action_M,
        )
        print(f"  MST  M={M} DAMS-score={score_M:.6f}")

        if score_M > best_score:
            best_score = score_M
            best_M = M

    print(f"\n✓ MST: selected M = {best_M} with DAMS-score = {best_score:.6f}\n")

    # --------------------------------------------------
    # 用 full pilot 重新 fit
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 (MST): Fitting final MST on full pilot")
    print("=" * 60)

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
    action_full_pilot = estimate_segment_policy(
        X_pilot, y_pilot, D_pilot, seg_labels_pilot
    )

    return tree_final, seg_labels_pilot, best_M, action_full_pilot




# =====================================================================
#  外部调用：run_policytree_segmentation（带 DAMS 选 M）
# =====================================================================

def run_policytree_segmentation(
    X_pilot: np.ndarray,
    D_pilot: np.ndarray,
    y_pilot: np.ndarray,
    X_train: np.ndarray,
    D_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    D_val: np.ndarray,
    y_val: np.ndarray,
    Gamma_val,
    M_candidates,
):
    """
    POLICYTREE + DAMS 版本（Gamma 由 R 端 GRF 计算）。

    区别于之前版本：
      - 对于每一个候选 M，都在 train_seg 上重新 fit 一次
        GRF + policy_tree，然后在这棵树上做 post-pruning 到 M。

    输入
    ----
    X_pilot, D_pilot, y_pilot : pilot 数据
    mu1_pilot_model, mu0_pilot_model, e_pilot : 用于 DAMS 的 DR
    depth : policy_tree 的最大深度（R 里的 depth 参数）
    train_frac : pilot 中用于 segmentation 训练的比例
    M_candidates : 候选的 segment 数

    输出
    ----
    seg_model_final : PolicyTreeSeg，在 full pilot 上重训 + prune 后的最终模型
    seg_labels_pilot : full pilot 上的 segment_id
    best_M : 选出来的最佳 segment 数
    action_final : full pilot 上用 diff-in-means(y) 学出的 segment-level action
    """

    print("\n" + "=" * 60)
    print("POLICYTREE: selecting M via DAMS (Gamma from R)")
    print("=" * 60)


    M_candidates = list(M_candidates)
    print(f"Candidate M's: {M_candidates}")
    # 先算好每个 M 对应的 depth
    # 修复：正确计算 depth，M=1 时 depth=0，其他情况加 buffer
    depth_for_M = {
        M: (0 if M == 1 else int(np.ceil(np.log2(M))) ) for M in M_candidates
    }

    # depth -> (tree_r_train, Gamma_train, leaf_parent_map, leaf_ids_train, action_ids_train)
    depth_cache = {}

    best_M = None
    best_score = -np.inf

    # 2) 对每个 M：重新 fit 一棵 policy tree，再 prune 到 M + DAMS
    for M in M_candidates:
        depth = depth_for_M[M]
        print("\n" + "-" * 60)
        print(f"  >> POLICYTREE: M = {M}, depth = {depth}")
        print("-" * 60)

        # 2.1 在 train_seg 上 fit GRF + policytree
        if depth not in depth_cache:
            tree_r_train, Gamma_train, leaf_parent_map, leaf_ids_train, action_ids_train = \
                _fit_policytree_with_grf(
                    X_train, y_train, D_train, depth=depth
                )
            depth_cache[depth] = (
                tree_r_train,
                Gamma_train,
                leaf_parent_map,
                leaf_ids_train,
                action_ids_train,
            )
        else:
            # 复用同一个 depth 对应的树和相关量
            (
                tree_r_train,
                Gamma_train,
                leaf_parent_map,
                leaf_ids_train,
                action_ids_train,
            ) = depth_cache[depth]



        # 2.2 在这棵树上 prune 到 M 个叶子
        seg_labels_train, action_ids_seg, leaf_to_pruned = policytree_post_prune_tree(
            leaf_ids_train,
            action_ids_train,
            Gamma_train,
            target_leaf_num=M,
            leaf_to_parent_map=leaf_parent_map,
        )

        # 2.3 在 train_seg 上，用 diff-in-means(y) 学各 segment 的 action
        action_M = estimate_segment_policy(
            X_train, y_train, D_train, seg_labels_train
        )

        # 2.4 构造一个临时 segmentation model（用来在 val_seg 上做 DAMS）
        seg_model_tmp = PolicyTreeSeg(tree_r_train, leaf_to_pruned)

        score_M = dams_score(
            seg_model=seg_model_tmp,
            X_val=X_val,
            D_val=D_val,
            y_val=y_val,
            Gamma_val=Gamma_val,
            action=action_M,
        )
        print(f"    DAMS-score(M={M}) = {score_M:.6f}")

        if score_M > best_score:
            best_score = score_M
            best_M = M

    if best_M is None:
        raise RuntimeError(
            "PolicyTree: no valid M found (all M_cand > leaf_count on train)."
        )

    print(f"\n✓ POLICYTREE: selected M = {best_M} with DAMS-score = {best_score:.6f}")

    # 3) 在 full pilot 上重训一棵 policy tree，并 prune 到 best_M
    print("\nRe-fitting GRF + PolicyTree on FULL pilot ...")
    # 修复：正确计算 depth
    depth_best = 0 if best_M == 1 else int(np.ceil(np.log2(best_M)))
    tree_r_full, Gamma_full, leaf_parent_full, leaf_ids_full, action_ids_full = \
        _fit_policytree_with_grf(
            X_pilot, y_pilot, D_pilot, depth=depth_best
        )

    n_leaves_full = len(np.unique(leaf_ids_full))
    target_M_full = min(best_M, n_leaves_full)
    if target_M_full < best_M:
        print(
            f"  [WARN] full-pilot leaf_count={n_leaves_full} < best_M={best_M}, "
            f"so we prune to {target_M_full} instead."
        )

    seg_labels_full, action_ids_full_seg, leaf_to_pruned_full = policytree_post_prune_tree(
        leaf_ids_full,
        action_ids_full,
        Gamma_full,
        target_leaf_num=target_M_full,
        leaf_to_parent_map=leaf_parent_full,
    )

    # === 用 R / policytree 的 action 来定义每个 segment 的动作 ===
    M_full = int(seg_labels_full.max() + 1)
    action_final = np.zeros(M_full, dtype=int)

    for m in range(M_full):
        idx = (seg_labels_full == m)

        # 理论上每个 segment 内的 action_ids_full_seg 都一致
        unique_actions = np.unique(action_ids_full_seg[idx])
        if len(unique_actions) != 1:
            raise ValueError(f"Segment {m} has multiple actions: {unique_actions}")
        action_final[m] = int(unique_actions[0])


    seg_model_final = PolicyTreeSeg(tree_r_full, leaf_to_pruned_full)

    return seg_model_final, seg_labels_full, best_M, action_final
