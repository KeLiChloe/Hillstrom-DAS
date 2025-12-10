from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_hillstrom
from outcome_model import fit_mu_models, predict_mu
import numpy as np
from sklearn.preprocessing import StandardScaler


def split_pilot_impl(X, D, y, pilot_frac, random_state=0):
    """
    Full data → pilot + implementation
    """
    X_pilot, X_impl, D_pilot, D_impl, y_pilot, y_impl = train_test_split(
        X,
        D,
        y,
        train_size=pilot_frac,
        random_state=random_state,
    )

    # Convert to numpy arrays if they are pandas objects
    # This ensures consistent integer-based indexing (not label-based)
    if hasattr(X_pilot, "reset_index"):
        X_pilot = X_pilot.reset_index(drop=True).values
        X_impl = X_impl.reset_index(drop=True).values
    elif hasattr(X_pilot, "values"):
        X_pilot = X_pilot.values
        X_impl = X_impl.values

    if hasattr(D_pilot, "reset_index"):
        D_pilot = D_pilot.reset_index(drop=True).values
        D_impl = D_impl.reset_index(drop=True).values
    elif hasattr(D_pilot, "values"):
        D_pilot = D_pilot.values
        D_impl = D_impl.values

    if hasattr(y_pilot, "reset_index"):
        y_pilot = y_pilot.reset_index(drop=True).values
        y_impl = y_impl.reset_index(drop=True).values
    elif hasattr(y_pilot, "values"):
        y_pilot = y_pilot.values
        y_impl = y_impl.values

    return X_pilot, X_impl, D_pilot, D_impl, y_pilot, y_impl


def split_seg_train_test(X_pilot, D_pilot, y_pilot, Gamma_pilot, test_frac):
    """
    Pilot → train_seg + val_seg

    不同 segmentation algorithm 会使用不同 test_frac。
    例如：
        KMeans → test_frac = 0
        DAST  → test_frac = 0.3
    """
    if test_frac <= 0:
        # No test split
        return (X_pilot, D_pilot, y_pilot, Gamma_pilot), (None, None, None, None)

    X_tr, X_te, D_tr, D_te, y_tr, y_te, Gamma_tr, Gamma_te = train_test_split(
        X_pilot,
        D_pilot,
        y_pilot,
        Gamma_pilot,
        test_size=test_frac,
        random_state=0,
    )

    if hasattr(X_tr, "reset_index"):
        X_tr = X_tr.reset_index(drop=True).values
        X_te = X_te.reset_index(drop=True).values
    elif hasattr(X_tr, "values"):
        X_tr = X_tr.values
        X_te = X_te.values

    if hasattr(D_tr, "reset_index"):
        D_tr = D_tr.reset_index(drop=True).values
        D_te = D_te.reset_index(drop=True).values
    elif hasattr(D_tr, "values"):
        D_tr = D_tr.values
        D_te = D_te.values

    if hasattr(y_tr, "reset_index"):
        y_tr = y_tr.reset_index(drop=True).values
        y_te = y_te.reset_index(drop=True).values
    elif hasattr(y_tr, "values"):
        y_tr = y_tr.values
        y_te = y_te.values

    return (X_tr, D_tr, y_tr, Gamma_tr), (X_te, D_te, y_te, Gamma_te)


import numpy as np
import pandas as pd
from sklift.datasets import fetch_hillstrom
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklift.datasets import fetch_hillstrom

def load_hillstrom(sample_frac, seed, target_col):
    np.random.seed(seed)
    print("Loading Hillstrom dataset ...")
    print("(Using random seed =", seed, ")")

    X, y, D = fetch_hillstrom(
        target_col=target_col,
        return_X_y_t=True,
    )

    # 子采样
    n_samples = int(len(X) * sample_frac)
    indices = np.random.choice(len(X), size=n_samples, replace=False)
    X, y, D = X.iloc[indices].copy(), y.iloc[indices].copy(), D.iloc[indices].copy()

    # ====== REMOVE history_segment (to avoid parsing strings) ======
    if "history_segment" in X.columns:
        X = X.drop(columns=["history_segment"])

    # ====== one-hot encoding for remaining categorical ======
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) > 0:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # ====== Standardize numerical features ======
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(float))

    # reset index
    y = y.reset_index(drop=True)
    D = D.reset_index(drop=True)

    # ====== treatment mapping to 0..K-1 ======
    unique_segments = sorted(D.unique())
    seg2id = {seg: i for i, seg in enumerate(unique_segments)}
    D_np = D.map(seg2id).astype(int).values

    # ====== debugging info ======
    print("\n" + "=" * 60)
    print("DATA EXPLORATION (Hillstrom)")
    print("=" * 60)
    print("\n Basic Information:")
    print(f"   X shape: {X_scaled.shape} (n={X_scaled.shape[0]}, d={X_scaled.shape[1]})")
    print(f"   Unique treatments: {unique_segments}")
    print(f"   Mapped as: {seg2id}")
    print(f"   Outcome mean (y): {y.mean():.6f}")

    # convert to numpy
    X_np = X_scaled.astype(float)
    y_np = y.values.astype(float)

    return X_np, y_np, D_np

# =========================================================
# 1. pilot / implementation 划分 + outcome model + Gamma (K-action)
# =========================================================
def prepare_pilot_impl(X, y, D, pilot_frac):
    """
    通用 K-action 版本：
      - 动作 D 可以是 0,1,...,K-1（Hillstrom: 3 actions）
      - 对每个 action a 拟合 μ_a(x)
      - 构造 DR 矩阵 Gamma_pilot, shape = (N_pilot, K)，第 a 列是 Γ_{i,a}
    """
    print("\n" + "=" * 60)
    print("Split & fit outcome models")
    print("=" * 60)

    X_pilot, X_impl, D_pilot, D_impl, y_pilot, y_impl = split_pilot_impl(
        X, D, y, pilot_frac=pilot_frac
    )
    print(f"Pilot size: {len(X_pilot)}, Implementation size: {len(X_impl)}")

    X_pilot = np.asarray(X_pilot)
    D_pilot = np.asarray(D_pilot)
    y_pilot = np.asarray(y_pilot)

    # ===== 1) 多 action 拟合 μ_a(x) =====
    mu_pilot_models = fit_mu_models(
        X_pilot,
        D_pilot,
        y_pilot,
        model_type="lightgbm_reg",
    )
    # 要求：D 已经被编码成非负整数（0,1,...）
    # Gamma 的列索引直接用 action 值：Gamma[:, a] 对应动作 a

    actions = np.unique(D_pilot).astype(int)
    max_a = actions.max()
    K = max_a + 1
    N_pilot = X_pilot.shape[0]

    print(f"Detected actions: {actions.tolist()} (K={K})")

    # ===== 2) 构造 K-action DR 矩阵 Gamma_pilot =====
    Gamma_pilot = np.zeros((N_pilot, K), dtype=float)

    for a in actions:
        mask_a = (D_pilot == a)
        e_a = mask_a.mean()
        # 避免除零，极端情况下如果某 action 特别少
        e_a = max(e_a, 1e-6)

        mu_a = predict_mu(mu_pilot_models[a], X_pilot)
        # 通用 DR 公式：
        #   Γ_{i,a} = μ_a(x_i) + 1{D_i = a} / e_a * (y_i - μ_a(x_i))
        Gamma_pilot[:, a] = mu_a + (mask_a.astype(float) / e_a) * (y_pilot - mu_a)

    # ===== 3) e_pilot 仍然返回一个标量（为兼容旧的 evaluator 接口）=====
    # 在多 action 设置下，这个值对 DR 没有实际意义，
    # 你可以在多 action evaluator 里完全忽略它。
    e_pilot = float(D_pilot.mean())

    return (
        X_pilot,
        X_impl,
        D_pilot,
        D_impl,
        y_pilot,
        y_impl,
        mu_pilot_models,  # dict[a] = model_a
        Gamma_pilot,      # (N_pilot, K) 的 DR 矩阵
    )
