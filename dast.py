# dast.py

"""
DAST (Decision-Aware Segmentation Tree)

Best-first tree growth: starting from a single root leaf, greedily expand
the leaf whose best admissible split gives the largest DR-value gain, until
exactly M leaves are reached or no positive-gain split exists.

- 输入：X, y, D, Gamma
- Gamma: doubly robust score matrix, shape (N, K), 第 a 列对应 action a ∈ {0,...,K-1}
- 候选阈值: H = {H_j}_j, H_j 是 feature j 上的候选 split points
"""

import copy
import warnings
import numpy as np
from estimation import _action_for_subset


class DASTNode:
    """Node in a DAST tree."""

    def __init__(self, indices, depth=0):
        self.indices = indices
        self.depth = depth
        self.value = None          # V̂(node), cached during build

        # split info (None for leaves)
        self.split_feature = None
        self.split_threshold = None
        self.left = None
        self.right = None
        self.is_leaf = True

        # segment id assigned after build
        self.segment_id = None


class DASTree:
    """
    Decision-aware segmentation tree: best-first growth to M leaves.

    只负责 segmentation：
    - build(M):  直接 best-first 生长到 M 个叶子（无需事后剪枝）
    - assign(X): 对任意 X 返回 segment_id
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        D: np.ndarray,
        gamma: np.ndarray,
        candidate_thresholds,
        min_leaf_size: int,
        value_type_dast: str,
        action_method: str,
    ):
        """
        Parameters
        ----------
        x : (N, d) features
        y : (N,) outcomes
        D : (N,) treatments, integer-coded 0, 1, ..., K-1
        gamma : (N, K) DR score matrix, column a corresponds to action a
        candidate_thresholds : dict of length d,
            H[j] is an iterable of candidate split thresholds for feature j
        min_leaf_size : int
            minimum samples per treatment in each leaf (StatisticallyAdmissible)
        value_type_dast : str
            'hybrid' or 'dr' — how to evaluate node value given the chosen action
        action_method : str
            'diff_in_means', 'gamma', or 'logistic' — how to choose the best
            action for each node (decoupled from value evaluation)
        """
        self.x = x
        self.y = y
        self.D = D.astype(int)
        self.gamma = gamma
        self.H = candidate_thresholds
        self.min_leaf_size = min_leaf_size
        self.value_type_dast = value_type_dast
        self.action_method = action_method

        self.actions = np.unique(self.D)
        self.K = int(self.gamma.shape[1])

        assert len(self.actions) == self.K, (
            f"D contains actions {self.actions.tolist()} but gamma has {self.K} columns; "
            "all K actions must appear in training data."
        )

        self.root: DASTNode | None = None
        self.leaf_nodes: list[DASTNode] = []

    # ======================================================================
    # Public API
    # ======================================================================

    def build(self, M: int, debug: bool = False):
        """
        Best-first grow tree to exactly M leaves (Algorithm 1: BuildTree).

        Starting from a single root leaf, greedily expand the leaf whose
        best admissible split gives the largest DR-value gain, until
        exactly M leaves are reached or no admissible split exists
        (min_leaf_size constraint). Non-positive-gain splits are allowed
        in order to reach the target M.

        Parameters
        ----------
        M     : target number of leaves
        debug : print split details when True
        """
        all_indices = np.arange(self.x.shape[0])
        self.root = DASTNode(all_indices, depth=0)
        self.leaf_nodes = [self.root]

        if debug:
            print(f"\n{'='*60}")
            print(f"Building DAST Tree  (N={len(all_indices)}, target M={M})")
            print(f"{'='*60}")

        while len(self.leaf_nodes) < M:
            best_global_gain = -np.inf
            best_leaf = None
            best_split = None  # (feature, threshold, left_idx, right_idx)

            # Snapshot to avoid iterating over a list modified after the loop
            for leaf in list(self.leaf_nodes):
                split, gain, V_node = self._find_best_split(leaf)
                leaf.value = V_node  # cache V̂_node; reused if leaf survives

                if gain > best_global_gain:
                    best_global_gain = gain
                    best_leaf = leaf
                    best_split = split
            
            # Hard stop: no admissible split exists anywhere in the tree
            if best_leaf is None or best_split is None:
                warnings.warn(
                    f"DAST terminates early with {len(self.leaf_nodes)} leaves "
                    f"(target M={M}): no admissible split satisfies min_leaf_size constraints.",
                    stacklevel=2,
                )
                break
            
            # Soft warning: best available gain is non-positive, but we keep splitting
            # if best_global_gain <= 0:
            #     warnings.warn(
            #         f"DAST: non-positive gain split (gain={best_global_gain:.4f}) at "
            #         f"{len(self.leaf_nodes)} leaves (target M={M}).",
            #         stacklevel=2,
            #     )

            # Apply the best split
            feat, thresh, left_idx, right_idx = best_split
            left_node = DASTNode(left_idx, depth=best_leaf.depth + 1)
            right_node = DASTNode(right_idx, depth=best_leaf.depth + 1)

            best_leaf.is_leaf = False
            best_leaf.split_feature = feat
            best_leaf.split_threshold = thresh
            best_leaf.left = left_node
            best_leaf.right = right_node

            self.leaf_nodes.remove(best_leaf)
            self.leaf_nodes.append(left_node)
            self.leaf_nodes.append(right_node)


        # The two newly-created leaves from the last split have value=None
        # (they were appended after the inner loop ran). Also handles M=1.
        for leaf in self.leaf_nodes:
            if leaf.value is None:
                leaf.value = self._compute_node_value(leaf.indices)

        # Assign segment IDs
        for seg_id, node in enumerate(self.leaf_nodes):
            node.segment_id = seg_id


    def copy(self):
        return copy.deepcopy(self)

    def assign(self, X: np.ndarray) -> np.ndarray:
        """
        Assign segment IDs to samples X by traversing the tree.
        Requires build() to have been called first.
        """
        if self.root is None:
            raise RuntimeError("Call build() before assign().")

        labels = np.empty(X.shape[0], dtype=int)
        for i, x_i in enumerate(X):
            node = self.root
            while not node.is_leaf:
                if x_i[node.split_feature] <= node.split_threshold:
                    node = node.left
                else:
                    node = node.right
            labels[i] = node.segment_id
        return labels

    # ======================================================================
    # Core: FindBestSplit
    # ======================================================================

    def _find_best_split(self, node: DASTNode):
        """
        Find the best admissible split for a leaf node.

        Returns
        -------
        split  : (feature, threshold, left_idx, right_idx) or None
        gain   : best gain achieved (-inf if no admissible split found)
        V_node : DR value of the node itself
        """
        indices = node.indices

        # Reuse cached value if available (avoids redundant computation)
        if node.value is not None:
            V_node = node.value
        else:
            V_node = self._compute_node_value(indices)

        best_gain = -np.inf
        best_j, best_t = None, None
        best_left = None
        best_right = None
        best_var_reduction = -np.inf   # tie-breaker

        for j in range(self.x.shape[1]):
            for t in self.H[j]:
                left_idx = indices[self.x[indices, j] <= t]
                right_idx = indices[self.x[indices, j] > t]

                if not (self._check_leaf_constraints(left_idx) and
                        self._check_leaf_constraints(right_idx)):
                    continue

                V_left = self._compute_node_value(left_idx)
                V_right = self._compute_node_value(right_idx)
                gain = V_left + V_right - V_node

                if gain > best_gain + 1e-9:
                    # Strictly better gain
                    best_gain = gain
                    best_j, best_t = j, t
                    best_left, best_right = left_idx, right_idx
                    best_var_reduction = self._compute_variance_reduction(
                        indices, left_idx, right_idx)

                elif abs(gain - best_gain) <= 1e-9:
                    # Tied gain: prefer larger covariate variance reduction
                    var_red = self._compute_variance_reduction(indices, left_idx, right_idx)
                    if var_red > best_var_reduction:
                        best_j, best_t = j, t
                        best_left, best_right = left_idx, right_idx
                        best_var_reduction = var_red

        if best_j is None:
            return None, -np.inf, V_node

        return (best_j, best_t, best_left, best_right), best_gain, V_node

    # ======================================================================
    # Node value computation
    # ======================================================================

    # ======================================================================
    # Node action selection  (step 1 — decoupled from value evaluation)
    # ======================================================================

    def _get_node_action(self, indices: np.ndarray) -> int:
        """
        Choose the best action for a node — delegates to _action_for_subset
        so grow-phase and post-grow segment policy share a single implementation.
        """
        return _action_for_subset(
            self.x, self.y, self.D, self.gamma,
            indices, self.action_method, self.actions,
        )

    # ======================================================================
    # Node value computation  (step 2 — given the chosen action)
    # ======================================================================

    def _compute_node_value(self, indices: np.ndarray) -> float:
        """
        ComputeNodeValue(L):

        Step 1 — choose best action via action_method:
            best_a = _get_node_action(L)

        Step 2 — evaluate V̂(L) using value_type_dast:
            hybrid : v̂_i = y_i        if D_i == best_a
                           Gamma_{i,best_a}  otherwise
            dr     : v̂_i = Gamma_{i,best_a}  (all customers)

        V̂(L) = sum_{i∈L} v̂_i
        """
        if len(indices) == 0:
            return 0.0

        best_a = self._get_node_action(indices)

        y_L = self.y[indices]
        D_L = self.D[indices]
        Gamma_L = self.gamma[indices, :]

        mask_a = (D_L == best_a)
        if self.value_type_dast == 'hybrid':
            v = np.where(mask_a, y_L, Gamma_L[:, best_a])
        else:  # 'dr'
            v = Gamma_L[:, best_a]

        return float(v.sum())

    # ======================================================================
    # Variance utilities (tie-breaking)
    # ======================================================================

    def _compute_variance_reduction(self, parent_idx, left_idx, right_idx) -> float:
        parent_var = self._compute_covariate_variance(parent_idx)
        left_var = self._compute_covariate_variance(left_idx)
        right_var = self._compute_covariate_variance(right_idx)

        n_parent = len(parent_idx)
        if n_parent == 0:
            return 0.0

        w_left = len(left_idx) / n_parent
        w_right = len(right_idx) / n_parent
        return parent_var - (w_left * left_var + w_right * right_var)

    def _compute_covariate_variance(self, indices) -> float:
        if len(indices) < 2:
            return 0.0
        return np.var(self.x[indices], axis=0, ddof=1).sum()

    # ======================================================================
    # Leaf constraints
    # ======================================================================

    def _check_leaf_constraints(self, indices) -> bool:
        """
        StatisticallyAdmissible: every treatment must appear >= min_leaf_size times.
        """
        if len(indices) == 0:
            return False
        D_sub = self.D[indices]
        for a in range(self.K):
            if np.sum(D_sub == a) < self.min_leaf_size:
                return False
        return True

    # ======================================================================
    # Tree traversal utilities
    # ======================================================================

    def _get_leaf_nodes(self):
        """Return current leaf list (maintained in sync during build)."""
        return list(self.leaf_nodes)

    def _gather_nodes(self, node, condition):
        if node is None:
            return []
        res = [node] if condition(node) else []
        res += self._gather_nodes(node.left, condition)
        res += self._gather_nodes(node.right, condition)
        return res
