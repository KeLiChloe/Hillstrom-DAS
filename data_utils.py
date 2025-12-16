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
from sklift.datasets import fetch_hillstrom, fetch_criteo

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
    X, y, D = X.iloc[indices], y.iloc[indices], D.iloc[indices]
    
    # print posit=iive ratio of y
    print(f"Positive ratio of y: {y.mean():.6f}")

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    D = D.reset_index(drop=True)

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

# =========================================================
# 1. pilot / implementation 划分 + outcome model + Gamma (K-action)
# =========================================================
def prepare_pilot_impl(X, y, D, pilot_frac, model_type, log_y):
    """
    K-action 版本 + log1p 回归稳定 heavy-tail revenue（Hillstrom 推荐 log_y=True）
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

    # ---- 2) fit μ_a models（你 outcome_model.py 不动）----
    mu_pilot_models = fit_mu_models(
        X_pilot,
        D_pilot,
        y_fit,
        model_type=model_type,   # 你 benchmark 要 NN 就用这个
    )

    K = int(np.max(D)) + 1   # 用全数据 D，不用 D_pilot
    actions = np.arange(K, dtype=int)

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
