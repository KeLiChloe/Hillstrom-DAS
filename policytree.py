# policytree.py  （或 policytree_segmentation.py，看你项目里的命名）
import numpy as np

# rpy2 / R
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

# =====================================================================
# R packages
# =====================================================================

grf = importr("grf")
policytree = importr("policytree")
DiagrammeRsvg = importr("DiagrammeRsvg")

# 定义一个 R 端的辅助函数：提取 leaf → parent 映射
ro.r(
    """
    extract_leaf_parent_map <- function(tree) {
      node_list <- as.list(tree)$nodes
      leaf_to_parent <- list()

      walk_tree <- function(node_id, parent_id = NA) {
        node <- node_list[[node_id]]
        if (is.null(node)) return(NULL)

        if (!is.null(node$is_leaf) && node$is_leaf) {
          leaf_to_parent[[as.character(node_id)]] <<- parent_id
        } else {
          walk_tree(node$left_child, node_id)
          walk_tree(node$right_child, node_id)
        }
      }

      walk_tree(1)  # root node id = 1
      return(leaf_to_parent)
    }
    """
)

# =====================================================================
#  Helper: 用 GRF 计算 Gamma，并在 R 里拟合 policy_tree （multi-action）
# =====================================================================

def _fit_policytree_with_grf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    D_train: np.ndarray,
    depth: int,
):
    """
    用 R 的 grf + policytree:
      - 若只有 2 个 action：用 causal_forest（与原版本一致）
      - 若 K >= 3：用 multi_arm_causal_forest / multi_causal_forest
      - 再用 policytree::double_robust_scores 得到 Gamma（N × K）
      - 再用 policytree::policy_tree(X, Gamma, depth)

    返回：
      - tree_r: R 的 policytree 对象
      - Gamma_train: numpy array, shape (N, K)
      - leaf_to_parent_map: dict[int -> int]
      - leaf_ids: numpy array, 每个样本所在叶子的 node.id
      - action_ids: numpy array, 每个样本的 action id（0,...,K-1）
    """

    # ---------- 1. 标准化形状 ----------
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    D_train = np.asarray(D_train).ravel()  # 可能是 int / float

    unique_actions = np.unique(D_train)
    K = len(unique_actions)
    # 保证是整数（0,1,2,...）
    D_int = D_train.astype(int)

    # ---------- 2. Python -> R ----------
    with localconverter(default_converter + numpy2ri.converter):
        X_r = ro.conversion.py2rpy(X_train)   # matrix

    y_r = FloatVector(y_train.tolist())

    # binary 和 multi-action 分开处理 W
    if K == 2:
        # 二元：直接用 numeric W
        D_r = FloatVector(D_int.tolist())
        # causal_forest
        cforest = grf.causal_forest(X_r, y_r, D_r)
    else:
        # 多 action：grf::multi_arm_causal_forest 期望 factor W
        with localconverter(default_converter + numpy2ri.converter):
            W_numeric_r = ro.conversion.py2rpy(D_int)
        # as.factor 变成 factor（level 顺序按数值来）
        W_factor_r = ro.r("as.factor")(W_numeric_r)
        # 这里可以直接调用 grf::multi_arm_causal_forest
        cforest = grf.multi_arm_causal_forest(X_r, y_r, W_factor_r)

    # ---------- 3. R 端：double robust scores + policy_tree ----------
    Gamma_r = policytree.double_robust_scores(cforest)

    ro.globalenv["X_r"] = X_r
    ro.globalenv["Gamma_r"] = Gamma_r
    ro.r(f"tree_obj <- policy_tree(X_r, Gamma_r, depth = {depth})")

    tree_r = ro.r("tree_obj")

    # ---------- 4. R -> Python ----------
    with localconverter(default_converter + numpy2ri.converter):
        # 4.0 把 Gamma_r 转回 numpy，注意现在是 N × K（K>=2）
        Gamma_train = np.asarray(ro.conversion.rpy2py(Gamma_r))

        # 4.1 leaf -> parent map
        leaf_to_parent_r = ro.r("extract_leaf_parent_map(tree_obj)")
        leaf_names = list(leaf_to_parent_r.names())
        leaf_values = list(leaf_to_parent_r)
        leaf_to_parent_map = {
            int(k): int(v)
            for k, v in zip(leaf_names, leaf_values)
        }

        # 4.2 training 上的 node.id / action.id
        segment_r = policytree.predict_policy_tree(tree_r, X_r, type="node.id")
        action_r = policytree.predict_policy_tree(tree_r, X_r, type="action.id")

        leaf_ids = np.asarray(ro.conversion.rpy2py(segment_r), dtype=int)
        action_ids_raw = np.asarray(ro.conversion.rpy2py(action_r), dtype=int)

    # R 里的 action.id 是 1..K，这里统一减 1 → 0..K-1
    action_ids = action_ids_raw - 1

    # ---------- 5. 清理 R 环境 ----------
    ro.r("rm(X_r, Gamma_r, tree_obj)")
    ro.r("gc()")

    return tree_r, Gamma_train, leaf_to_parent_map, leaf_ids, action_ids


# =====================================================================
#  Node 价值函数（用 K-action Γ）
# =====================================================================

def _node_best_action_and_value(Gamma: np.ndarray, indices: np.ndarray):
    """
    给定一个 node 中的样本集合 indices，以及其对应的 DR scores Gamma（N × K）：
      - 对每个 action a 计算 node 内 Γ_{i,a} 的平均（或总和）
      - 选出最优 action a* = argmax_a mean Γ_{i,a}
      - 返回 (a*, sum_i Γ_{i,a*})
    """
    if len(indices) == 0:
        return 0, 0.0

    G_node = Gamma[indices, :]   # (n_node, K)
    # 对每一列（每个 action）求平均，再 argmax
    avg_per_action = G_node.mean(axis=0)    # shape (K,)
    best_action = int(np.argmax(avg_per_action))
    value = float(G_node[:, best_action].sum())
    return best_action, value


def _compute_node_value_DR(Gamma: np.ndarray, indices: np.ndarray) -> float:
    """
    Node value（用于 pruning 中的 welfare 计算）：

    - K-action 通用版：
        对于 node 内样本 i ∈ L，给每个 action a 有 Γ_{i,a}
        在该 node 上强制用一个统一的 action a_L：
            a_L = argmax_a (1/|L|) ∑_{i∈L} Γ_{i,a}
        node 的 value 定义为：
            V̂_L = ∑_{i∈L} Γ_{i,a_L}
    """
    _, value = _node_best_action_and_value(Gamma, indices)
    return value


# =====================================================================
#  Post-pruning（只允许 sibling merge）
# =====================================================================

from itertools import combinations


def policytree_post_prune_tree(
    leaf_ids: np.ndarray,
    action_ids: np.ndarray,
    Gamma_train: np.ndarray,
    target_leaf_num: int,
    leaf_to_parent_map: dict,
):
    """
    基于 DR Γ（N × K）的 multi-action post-pruning：

    - 初始 segment = 每个叶子节点（leaf node.id）
    - 只允许合并 sibling 叶节点（共享同一个 parent）
    - 每次合并选择 welfare loss 最小的 pair
    - 合并后的 segment action 用 Gamma 决定：
          a* = argmax_a mean_{i in merged} Γ_{i,a}
    """

    # leaf segment -> sample indices
    segment_map = {
        s: np.where(leaf_ids == s)[0].astype(int)
        for s in sorted(set(leaf_ids))
    }
    segment_to_leaves = {s: {s} for s in segment_map}

    # 确保每个 leaf 内的 action_ids 一致（来自 policytree）
    for s, idxs in segment_map.items():
        assert np.all(action_ids[idxs] == action_ids[idxs[0]])

    action_map = {s: int(action_ids[idxs[0]]) for s, idxs in segment_map.items()}

    # 当前 segment 数 > target_leaf_num 时，不断 merge
    while len(segment_map) > target_leaf_num:
        best_pair = None
        best_action = None
        min_welfare_loss = float("inf")

        segments = list(segment_map.keys())

        for s1, s2 in combinations(segments, 2):
            # 只合并 “同一父节点的叶” 所构成的 segments
            combined_leaves = segment_to_leaves[s1] | segment_to_leaves[s2]
            combined_parents = {leaf_to_parent_map[leaf] for leaf in combined_leaves}
            if len(combined_parents) != 1:
                continue

            idx1, idx2 = segment_map[s1], segment_map[s2]
            merged_idx = np.concatenate([idx1, idx2])

            # 合并后的最佳 welfare（node 内选一个统一 action）
            _, merged_node_value = _node_best_action_and_value(Gamma_train, merged_idx)

            # merge 前各自 node 的 welfare（各自内部也选各自最优 action）
            w1 = _compute_node_value_DR(Gamma_train, idx1)
            w2 = _compute_node_value_DR(Gamma_train, idx2)
            original_total = w1 + w2

            loss = original_total - merged_node_value
            if loss <= min_welfare_loss:
                min_welfare_loss = loss
                best_pair = (s1, s2)

                # 合并后的 action：用 Gamma 的平均决定（K-action 通用）
                G_node = Gamma_train[merged_idx, :]
                avg_per_action = G_node.mean(axis=0)
                best_action = int(np.argmax(avg_per_action))

        if best_pair is None:
            raise ValueError("No valid sibling pairs to merge.")

        s1, s2 = best_pair
        new_seg_id = min(s1, s2)
        drop_id = s2 if new_seg_id == s1 else s1

        merged_indices = np.concatenate([segment_map[s1], segment_map[s2]])
        segment_map[new_seg_id] = merged_indices
        action_map[new_seg_id] = best_action
        segment_to_leaves[new_seg_id] = segment_to_leaves[s1] | segment_to_leaves[s2]

        # 删除被 merge 的 id
        for sid in (drop_id,):
            del segment_map[sid]
            del action_map[sid]
            del segment_to_leaves[sid]

    # 重新映射成 0,1,...,M-1
    final_segments = sorted(segment_map.keys())
    seg_id_map = {old: new for new, old in enumerate(final_segments)}

    pruned_segment_labels = np.zeros(len(leaf_ids), dtype=int)
    pruned_action_ids = np.zeros(len(leaf_ids), dtype=int)

    for old_seg, indices in segment_map.items():
        new_seg = seg_id_map[old_seg]
        pruned_segment_labels[indices] = new_seg
        pruned_action_ids[indices] = int(action_map[old_seg])

    # leaf node.id -> pruned segment id
    leaf_to_pruned_segment = {}
    for old_seg, leaf_set in segment_to_leaves.items():
        if old_seg not in seg_id_map:
            continue
        new_seg = seg_id_map[old_seg]
        for leaf in leaf_set:
            leaf_to_pruned_segment[leaf] = new_seg

    return pruned_segment_labels, pruned_action_ids, leaf_to_pruned_segment


# =====================================================================
#  Segmentation class: PolicyTreeSeg
# =====================================================================

class PolicyTreeSeg:
    """
    跟 KMeansSeg / GMMSeg / DASTree 一样，提供 assign(X) 接口。

    内部保存：
      - tree_r: R 的 policytree 对象
      - leaf_to_pruned_segment: dict[node.id -> seg_id]
    """

    def __init__(self, tree_r, leaf_to_pruned_segment):
        self.tree_r = tree_r
        self.leaf_to_pruned_segment = dict(leaf_to_pruned_segment)

    def assign(self, X: np.ndarray) -> np.ndarray:
        """
        对任意 X（numpy array）给出 segment_id。
        """
        X = np.asarray(X, dtype=float)

        with localconverter(default_converter + numpy2ri.converter):
            X_r = ro.conversion.py2rpy(X)

        segment_r = policytree.predict_policy_tree(self.tree_r, X_r, type="node.id")
        with localconverter(default_converter + numpy2ri.converter):
            leaf_ids = np.array(ro.conversion.rpy2py(segment_r), dtype=int)

        labels = np.zeros(len(leaf_ids), dtype=int)
        for i, leaf in enumerate(leaf_ids):
            labels[i] = self.leaf_to_pruned_segment[leaf]
        return labels
