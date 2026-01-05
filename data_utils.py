from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_hillstrom, fetch_criteo, fetch_lenta
from outcome_model import fit_mu_models, predict_mu
import numpy as np
import pandas as pd
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

    # ====== 移除包含空值的行 ======
    # 先重置索引，确保对齐
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    D = D.reset_index(drop=True)
    
    # 创建掩码并过滤
    mask_notnull = X.notnull().all(axis=1) & y.notnull() & D.notnull()
    n_removed = (~mask_notnull).sum()
    if n_removed > 0:
        print(f"Removing {n_removed} rows with null values ({n_removed/len(X)*100:.2f}%)")
        X = X[mask_notnull].reset_index(drop=True)
        y = y[mask_notnull].reset_index(drop=True)
        D = D[mask_notnull].reset_index(drop=True)
    
    print(f"Final sample size: {len(X)}")

    # Remove history_segment (to avoid parsing strings)
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

def load_criteo(sample_frac, seed, target_col):
    np.random.seed(seed)
    print("Loading Criteo uplift dataset ...")
    print("(Using random seed =", seed, ")")
    
    X, y, D = fetch_criteo(
        target_col=target_col,
        treatment_col="treatment",
        percent10=True,
        return_X_y_t=True,
    )
    

    n_samples = int(len(X) * sample_frac)
    indices = np.random.choice(len(X), size=n_samples, replace=False)
    X, y, D = X.iloc[indices].copy(), y.iloc[indices].copy(), D.iloc[indices].copy()
    
    # ====== 移除包含空值的行 ======
    # 先重置索引，确保对齐
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    D = D.reset_index(drop=True)
    
    # 创建掩码并过滤
    mask_notnull = X.notnull().all(axis=1) & y.notnull() & D.notnull()
    n_removed = (~mask_notnull).sum()
    if n_removed > 0:
        print(f"Removing {n_removed} rows with null values ({n_removed/len(X)*100:.2f}%)")
        X = X[mask_notnull].reset_index(drop=True)
        y = y[mask_notnull].reset_index(drop=True)
        D = D[mask_notnull].reset_index(drop=True)
    
    # 打印基本信息
    print(f"Final sample size: {len(X)}")
    print(f"Positive ratio of y: {y.mean():.6f}")

    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)

    print("\n Basic Information:")
    print(f"   X shape: {X.shape} (n={X.shape[0]}, d={X.shape[1]})")

    # print("\n Outcome by Treatment:")
    # y_control = y[D == 0]
    # y_treated = y[D == 1]
    # print(f"   Control (D=0) - mean: {y_control.mean():.6f}, std: {y_control.std():.6f}")
    # print(f"   Treated (D=1) - mean: {y_treated.mean():.6f}, std: {y_treated.std():.6f}")
    # print(f"   Naive ATE: {y_treated.mean() - y_control.mean():.6f}")
    
    # # print ratio of treatment assignment (D=1) and positive outcomes
    # print("\n Treatment Assignment:")
    # print(f"   Treatment (D=1) ratio: {D.mean():.6f}")
    # print(f"   Positive Outcome (y=1) ratio: {y.mean():.6f}")

    # 转成 numpy
    X_np = X.values
    y_np = y.values
    D_np = D.values

    # scale X features
    # scaler = StandardScaler()
    # X_np = scaler.fit_transform(X_np)
    
    return X_np, y_np, D_np


def load_lenta(sample_frac, seed, target_col=None):
    """
    Load Lenta.ru dataset from sklift.
    
    Dataset: Russian news website, binary treatment (email/no email)
    Target: Fixed binary outcome (conversion)
    Treatment: Binary (0=control, 1=treatment)
    Size: ~200k samples
    Features: Categorical + numerical
    
    Parameters
    ----------
    sample_frac : float
        Fraction of data to sample (0, 1]
    seed : int
        Random seed
    target_col : str
        Ignored for compatibility. Lenta has only one target.
    
    Returns
    -------
    X_np, y_np, D_np : numpy arrays
    """
    np.random.seed(seed)
    print("Loading Lenta.ru dataset ...")
    print("(Using random seed =", seed, ")")
    
    # fetch_lenta 正确用法
    X, y, D = fetch_lenta(return_X_y_t=True)
    
    # ====== 处理 treatment 列（可能是字符串类型）======
    if D.dtype == 'object' or D.dtype.name == 'category':
        print(f"Converting treatment from strings: {D.unique()}")
        # Lenta: "test" -> 1, "control" -> 0
        D = D.map({'test': 1, 'control': 0})
        if D.isnull().any():
            raise ValueError(f"Unknown treatment values found after mapping: {D[D.isnull()].unique()}")
    
    # 子采样
    n_samples = int(len(X) * sample_frac)
    indices = np.random.choice(len(X), size=n_samples, replace=False)
    X, y, D = X.iloc[indices].copy(), y.iloc[indices].copy(), D.iloc[indices].copy()
    
    # ====== 移除包含空值的行 ======
    # 先重置索引，确保对齐
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    D = D.reset_index(drop=True)
    
    # 创建掩码并过滤
    mask_notnull = X.notnull().all(axis=1) & y.notnull() & D.notnull()
    n_removed = (~mask_notnull).sum()
    if n_removed > 0:
        print(f"Removing {n_removed} rows with null values ({n_removed/len(X)*100:.2f}%)")
        X = X[mask_notnull].reset_index(drop=True)
        y = y[mask_notnull].reset_index(drop=True)
        D = D[mask_notnull].reset_index(drop=True)
    
    # 打印基本信息
    print(f"Final sample size: {len(X)}")
    print(f"Positive ratio of y: {y.mean():.6f}")
    print(f"Treatment ratio (D=1): {D.mean():.6f}")
    
    # ====== 处理 categorical features ======
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) > 0:
        print(f"One-hot encoding {len(cat_cols)} categorical features: {cat_cols}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    # ====== Standardize numerical features ======
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(float))
    
    # reset index
    y = y.reset_index(drop=True)
    D = D.reset_index(drop=True)
    
    print("\n" + "=" * 60)
    print("DATA EXPLORATION (Lenta)")
    print("=" * 60)
    print("\n Basic Information:")
    print(f"   X shape: {X_scaled.shape} (n={X_scaled.shape[0]}, d={X_scaled.shape[1]})")
    print(f"   Unique treatments: {sorted(D.unique())}")
    print(f"   Outcome mean (y): {y.mean():.6f}")
    print(f"   Treatment assignment (D=1): {D.mean():.6f}")
    
    # convert to numpy
    X_np = X_scaled.astype(float)
    y_np = y.values.astype(float)
    D_np = D.values.astype(int)
    
    return X_np, y_np, D_np

# =========================================================
# 1. pilot / implementation 划分 + outcome model + Gamma (K-action)
# =========================================================
def prepare_pilot_impl(X, y, D, pilot_frac, model_type, log_y):
    """
    K-action 版本
    """
    print("\n" + "=" * 60)
    print("Split & fit outcome models")
    print("=" * 60)

    X_pilot, X_impl, D_pilot, D_impl, y_pilot, y_impl = split_pilot_impl(
        X, D, y, pilot_frac=pilot_frac
    )
    print(f"Pilot size: {len(X_pilot)}, Implementation size: {len(X_impl)}")

    X_pilot = np.asarray(X_pilot)
    D_pilot = np.asarray(D_pilot).astype(int)
    y_pilot = np.asarray(y_pilot, dtype=float)

    # ---- 1) log1p 训练目标（只在训练时变换）----
    if log_y:
        if (y_pilot < 0).any():
            raise ValueError("log1p requires non-negative y, but found y<0 in pilot.")
        y_fit = np.log1p(y_pilot)
    else:
        y_fit = y_pilot

    # ---- 2) fit μ_a models----
    mu_pilot_models = fit_mu_models(
        X_pilot,
        D_pilot,
        y_fit,
        model_type=model_type,   
    )

    K = int(np.max(D)) + 1   # 用全数据 D，不用 D_pilot
    actions = np.arange(K, dtype=int)
    
    # ---- 检查所有 action 是否都有模型 ----
    missing_actions = set(actions.tolist()) - set(mu_pilot_models.keys())
    if missing_actions:
        raise ValueError(
            f"Pilot split resulted in missing actions: {sorted(missing_actions)}. "
            f"These actions have no samples in pilot data. "
            f"Consider increasing pilot_frac or sample_frac."
        )

    N_pilot = X_pilot.shape[0]
    print(f"Detected actions: {actions.tolist()} (K={K}), log_y={log_y}")

    # ---- 3) build Gamma_pilot: (N, K) ----
    Gamma_pilot = np.zeros((N_pilot, K), dtype=float)
    for a in actions:
        mask_a = (D_pilot == a)
        e_a = mask_a.mean()
        e_a = max(e_a, 1e-6)

        mu_a_hat = predict_mu(mu_pilot_models[a], X_pilot)  # 在 log 空间 or 原空间
        if log_y:
            mu_a_hat = np.expm1(mu_a_hat)  # 还原到 revenue 空间

        Gamma_pilot[:, a] = mu_a_hat + (mask_a.astype(float) / e_a) * (y_pilot - mu_a_hat)
        
    return (
        X_pilot,
        X_impl,
        D_pilot,
        D_impl,
        y_pilot,
        y_impl,
        mu_pilot_models,
        Gamma_pilot,
    )
