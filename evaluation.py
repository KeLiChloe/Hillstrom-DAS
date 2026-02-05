import numpy as np


# =========================================================
# 工具函数：处理动作空间 & propensity
# =========================================================

def _infer_actions(D_impl, mu_models):
    """
    根据 D_impl 和 mu_models 推断动作空间 {0,1,...,K-1} 和 K。
    要求动作已经编码为非负整数。
    """
    D_impl = np.asarray(D_impl).astype(int)
    # 所有在数据里出现过的动作
    actions_data = np.unique(D_impl)
    # 所有有模型的动作
    actions_model = np.array(sorted(mu_models.keys()), dtype=int)

    actions = np.unique(np.concatenate([actions_data, actions_model]))
    if actions.min() < 0:
        raise ValueError("Actions must be encoded as non-negative integers (0,1,2,...)")

    K = int(actions.max()) + 1
    return actions, K


def _get_propensity_per_action(D_impl, actions, propensities):
    """
    返回每个动作 a 的行为策略概率 e_a.

    参数
    ----
    D_impl : array-like (n,)
        实际被行为策略选择的动作（日志中的 D）
    actions : array of int
        所有动作标签（例如 [0,1,2]）
    propensities : None, dict 或 1D array
        - None: 用 D_impl 的经验分布 P(D=a)
        - dict: {a: e_a}
        - array-like: e[a] = 对应动作 a 的概率

    返回
    ----
    e : np.ndarray, shape (K,)
        第 a 个元素为 e_a
    """
    D_impl = np.asarray(D_impl).astype(int)
    max_a = int(actions.max())
    K = max_a + 1
    e = np.zeros(K, dtype=float)

    if propensities is None:
        # 用经验分布 P(D=a)
        n = len(D_impl)
        for a in actions:
            e[a] = (D_impl == a).mean()
    elif isinstance(propensities, dict):
        for a in actions:
            e[a] = float(propensities.get(a, 0.0))
    else:
        # 当作 1D array 处理: e[a] = propensities[a]
        prop_arr = np.asarray(propensities, dtype=float)
        if prop_arr.ndim != 1 or prop_arr.shape[0] < K:
            raise ValueError("propensities as array must have length >= max(actions)+1")
        e[:] = prop_arr[:K]

    # 避免除0
    e = np.clip(e, 1e-6, 1.0)
    return e


def _build_mu_matrix(mu_models, X_impl, K, log_y):
    n = X_impl.shape[0]
    mu_mat = np.zeros((n, K), dtype=float)

    for a, model in mu_models.items():
        a_int = int(a)
        pred = model.predict(X_impl)
        if log_y:
            pred = np.expm1(pred)
        mu_mat[:, a_int] = pred

    return mu_mat

# =========================================================
# 1. 非 DR 版本：直接用 y 和 μ 做 counterfactual
# =========================================================

def evaluate_policy(
        X_impl, D_impl, y_impl,
        seg_labels_impl,
        mu_models,
        action,
        log_y,
    ):
    """
    K-action 非 DR policy evaluation：

        v_i = y_i                 if D_i == a_i
            = μ_{a_i}(x_i)        if D_i != a_i

    这里 μ_{a_i}(x_i) 由 outcome model 给出：
        a_i 是 segment-level policy 推荐的动作。
    """
    X_impl = np.asarray(X_impl)
    D_impl = np.asarray(D_impl).astype(int)
    y_impl = np.asarray(y_impl, dtype=float)
    seg_labels_impl = np.asarray(seg_labels_impl, dtype=int)
    action = np.asarray(action, dtype=int)

    actions, K = _infer_actions(D_impl, mu_models)
    mu_mat = _build_mu_matrix(mu_models, X_impl, K, log_y=log_y)  # shape (n, K)

    # 边界检查：确保 segment labels 在有效范围内
    if seg_labels_impl.max() >= len(action):
        raise ValueError(
            f"Segment label {seg_labels_impl.max()} exceeds action array length {len(action)}. "
            f"Expected seg_labels in [0, {len(action)-1}]."
        )
    
    # 根据 segment label 得到 policy 对每个样本的 action: a_i
    a_i = action[seg_labels_impl].astype(int)        # shape (n,)

    # 对应 policy action 的预测 μ_{a_i}(x_i)
    mu_a = mu_mat[np.arange(len(X_impl)), a_i]

    # factual: D_i = a_i → 用真实 y_i
    # counterfactual: D_i != a_i → 用 μ_{a_i}(x_i)
    v = np.where(D_impl == a_i, y_impl, mu_a)

    return {
        "value_mean": float(v.mean()),
        "value_sum": float(v.sum()),
    }


# =========================================================
# 2. Dual DR: factual 用 y，counterfactual 用 DR 的 Γ_{i,a_i}
# =========================================================

def evaluate_policy_dual_dr(
        X_impl, D_impl, y_impl,
        seg_labels_impl,
        mu_models,
        action,
        propensities,
        log_y,
    ):
    """
    K-action dual DR policy evaluation.

    思路：
        - 对每个动作 a 构造 DR pseudo-outcome:
              Gamma_{i,a} = μ_a(x_i) + 1{D_i = a} / e_a * (y_i - μ_a(x_i))
        - 然后：
              v_i = y_i                      if D_i == a_i
                  = Gamma_{i,a_i}           if D_i != a_i

        这样就是一个 “y + DR” 的评估器：
        - factual 用真实 y
        - counterfactual 用 DR 修正过的 Γ
    参数
    ----
    X_impl : (n, d)
    D_impl : (n,)  实际动作
    y_impl : (n,)  实际 outcome
    seg_labels_impl : (n,) segment id
    mu_models : dict[int, model]，fit_mu_models 的输出
    action : (M,) segment-level 决策，action[m] ∈ {0,1,...,K-1}
    propensities : None / dict / array
        行为策略的 per-action 概率 e_a。
        - None 时使用 D_impl 的经验分布 P(D=a)
    """
    X_impl = np.asarray(X_impl)
    D_impl = np.asarray(D_impl).astype(int)
    y_impl = np.asarray(y_impl, dtype=float)
    seg_labels_impl = np.asarray(seg_labels_impl, dtype=int)
    action = np.asarray(action, dtype=int)

    n = X_impl.shape[0]

    # 1. 动作空间 & propensity
    actions, K = _infer_actions(D_impl, mu_models)
    e = _get_propensity_per_action(D_impl, actions, propensities)  # shape (K,)

    # 2. μ_a(x_i) 矩阵
    mu_mat = _build_mu_matrix(mu_models, X_impl, K, log_y=log_y)  # (n, K)

    # 边界检查：确保 segment labels 在有效范围内
    if seg_labels_impl.max() >= len(action):
        raise ValueError(
            f"Segment label {seg_labels_impl.max()} exceeds action array length {len(action)}. "
            f"Expected seg_labels in [0, {len(action)-1}]."
        )

    # 3. 构造 DR Γ_{i,a}
    Gamma = np.zeros((n, K), dtype=float)
    for a in actions:
        a_int = int(a)
        mask_a = (D_impl == a_int).astype(float)
        mu_a = mu_mat[:, a_int]
        e_a = e[a_int]
        Gamma[:, a_int] = mu_a + (mask_a / e_a) * (y_impl - mu_a)

    # 4. segment-level policy → individual a_i
    a_i = action[seg_labels_impl].astype(int)  # (n,)

    # 5. dual DR 组合
    match = (D_impl == a_i)
    v = np.empty_like(y_impl, dtype=float)

    # factual 部分
    v[match] = y_impl[match]
    # counterfactual 部分用 DR Gamma
    v[~match] = Gamma[np.arange(n)[~match], a_i[~match]]

    return {
        "value_mean": float(v.mean()),
        "value_sum": float(v.sum()),
    }


# =========================================================
# 3. 传统 DR 版本（不区分 factual / counterfactual）
# =========================================================

def evaluate_policy_dr(
        X_impl, D_impl, y_impl,
        seg_labels_impl,
        mu_models,
        action,
        propensities,
        log_y,
    ):
    """
    K-action doubly-robust off-policy evaluation.

    多动作的自然推广：
        - μ_d(i) = μ_{D_i}(x_i)
        - μ_pi(i) = μ_{a_i}(x_i)
        - p_a(i) = e_{a_i}   (行为策略对 a_i 的概率，简单起见假设与 X 无关)
        - v_i = μ_pi(i) + 1{D_i = a_i} / p_a(i) * (y_i - μ_d(i))
    """
    X_impl = np.asarray(X_impl)
    D_impl = np.asarray(D_impl).astype(int)
    y_impl = np.asarray(y_impl, dtype=float)
    seg_labels_impl = np.asarray(seg_labels_impl, dtype=int)
    action = np.asarray(action, dtype=int)

    n = X_impl.shape[0]

    # 1. 动作空间 & propensity
    actions, K = _infer_actions(D_impl, mu_models)
    e = _get_propensity_per_action(D_impl, actions, propensities)  # (K,)

    # 2. μ_a(x_i) 矩阵
    mu_mat = _build_mu_matrix(mu_models, X_impl, K, log_y=log_y)

    # 边界检查：确保 segment labels 在有效范围内
    if seg_labels_impl.max() >= len(action):
        raise ValueError(
            f"Segment label {seg_labels_impl.max()} exceeds action array length {len(action)}. "
            f"Expected seg_labels in [0, {len(action)-1}]."
        )

    # 3. policy 的 action a_i = π(X_i)
    a_i = action[seg_labels_impl].astype(int)

    # 4. μ_d(i) 和 μ_pi(i)
    mu_d = mu_mat[np.arange(n), D_impl]    # μ_{D_i}(x_i)
    mu_pi = mu_mat[np.arange(n), a_i]      # μ_{π(X_i)}(x_i)

    p_a = e[a_i]                           # p_{π(X_i)}(X_i) ≈ e_{a_i}
    indicator = (D_impl == a_i).astype(float)

    v = mu_pi + indicator / p_a * (y_impl - mu_d)

    return {
        "value_mean": float(v.mean()),
        "value_sum": float(v.sum()),
    }


# =========================================================
# 4. 纯 IPW 版本
# =========================================================

def evaluate_policy_ipw(
        X_impl, D_impl, y_impl,
        seg_labels_impl,
        mu_models, # placeholder, no use
        action,
        propensities,
        log_y # placeholder, no use
    ):
    """
    纯 IPW policy evaluation，只用 y，不用 outcome model / Gamma。

    多动作版公式：
        V_hat = (1/n) * sum_i [ 1{D_i = a_i} / p_{a_i} * y_i ]

    其中 p_{a_i} = 行为策略对动作 a_i 的概率（propensities 提供）。
    """
    D_impl = np.asarray(D_impl).astype(int)
    y_impl = np.asarray(y_impl, dtype=float)
    seg_labels_impl = np.asarray(seg_labels_impl, dtype=int)
    action = np.asarray(action, dtype=int)

    n = len(y_impl)

    # 1. 动作空间 & propensity
    actions = np.unique(D_impl)
    max_a = int(actions.max())
    K = max_a + 1
    e = _get_propensity_per_action(D_impl, actions, propensities)  # (K,)

    # 边界检查：确保 segment labels 在有效范围内
    if seg_labels_impl.max() >= len(action):
        raise ValueError(
            f"Segment label {seg_labels_impl.max()} exceeds action array length {len(action)}. "
            f"Expected seg_labels in [0, {len(action)-1}]."
        )

    # 2. policy 决策 a_i
    a_i = action[seg_labels_impl].astype(int)

    # 3. 对应 policy action 的概率 p_{a_i}
    p_a = e[a_i]  # p_{π(X_i)}

    # 4. IPW 权重，只在 D==a 时非 0
    indicator = (D_impl == a_i).astype(float)
    w = indicator / p_a

    contrib = w * y_impl
    value_mean = float(contrib.mean())

    return {
        "value_mean": value_mean,
        "value_sum": float(value_mean * n),
    }


# =========================================================
# 5. 随机 baseline 的 dual DR 版本（individual-level a_random）
# =========================================================

def evaluate_policy_for_random_baseline(
        X_impl, D_impl, y_impl,
        a_random,
        mu_models,
        propensities,
        log_y,
    ):
    """
    Evaluate an individual-level random policy using multi-action dual DR estimator.

    Parameters
    ----------
    X_impl, D_impl, y_impl : implementation data
    a_random : np.ndarray, shape (N_impl,)
        Individual-level action recommended by the random policy (0..K-1).
    mu_models : dict[int, model]
        Outcome models fitted on pilot, one per action.
    propensities : None / dict / array
        行为策略的 per-action 概率 e_a（日志里 D 的生成机制）。
    """
    X_impl = np.asarray(X_impl)
    D_impl = np.asarray(D_impl).astype(int)
    y_impl = np.asarray(y_impl, dtype=float)
    a_random = np.asarray(a_random).astype(int)

    n = X_impl.shape[0]

    # 动作空间 & propensity
    actions, K = _infer_actions(D_impl, mu_models)
    e = _get_propensity_per_action(D_impl, actions, propensities)

    # μ 矩阵
    mu_mat = _build_mu_matrix(mu_models, X_impl, K, log_y=log_y)

    # 构造 DR Γ_{i,a}
    Gamma = np.zeros((n, K), dtype=float)
    for a in actions:
        a_int = int(a)
        mask_a = (D_impl == a_int).astype(float)
        mu_a = mu_mat[:, a_int]
        e_a = e[a_int]
        Gamma[:, a_int] = mu_a + (mask_a / e_a) * (y_impl - mu_a)

    # dual DR: factual 用 y, counterfactual 用 Γ_{i,a_random}
    match = (D_impl == a_random)
    v = np.empty_like(y_impl, dtype=float)
    v[match] = y_impl[match]
    v[~match] = Gamma[np.arange(n)[~match], a_random[~match]]

    return {
        "value_mean": float(v.mean()),
        "value_sum": float(v.sum()),
    }
