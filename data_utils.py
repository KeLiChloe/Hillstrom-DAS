from sklearn.model_selection import train_test_split
from sklift.datasets import fetch_hillstrom, fetch_criteo
from outcome_model import fit_mu_models, predict_mu_values
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def split_pilot_impl(X, D, y, pilot_frac, random_state=0, return_indices=False):
    """
    Full data → pilot + implementation.

    If return_indices=True, also return cohort row indices (customer_id) for
    pilot and implementation splits. Indices refer to rows in the input cohort
    (0 .. len(X)-1) before the split.
    """
    if return_indices:
        indices = np.arange(len(X))
        idx_pilot, idx_impl, X_pilot, X_impl, D_pilot, D_impl, y_pilot, y_impl = (
            train_test_split(
                indices,
                X,
                D,
                y,
                train_size=pilot_frac,
                random_state=random_state,
            )
        )
        idx_pilot = np.asarray(idx_pilot, dtype=int)
        idx_impl = np.asarray(idx_impl, dtype=int)
    else:
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

    if return_indices:
        return (
            X_pilot,
            X_impl,
            D_pilot,
            D_impl,
            y_pilot,
            y_impl,
            idx_pilot,
            idx_impl,
        )
    return X_pilot, X_impl, D_pilot, D_impl, y_pilot, y_impl


def verify_impl_customer_alignment(
    impl_customer_id,
    D_impl,
    y_impl,
    D_cohort,
    y_cohort,
    *,
    context="",
):
    """
    Ensure implementation rows match cohort rows indexed by customer_id.

    Row k must satisfy D_impl[k] == D_cohort[customer_id[k]] (same for y).
    customer_id values must be unique and lie in [0, len(D_cohort)).
    """
    prefix = f"{context}: " if context else ""
    cid = np.asarray(impl_customer_id, dtype=int)
    D_i = np.asarray(D_impl, dtype=int)
    y_i = np.asarray(y_impl, dtype=float)
    D_all = np.asarray(D_cohort, dtype=int)
    y_all = np.asarray(y_cohort, dtype=float)

    n = len(cid)
    if not (len(D_i) == len(y_i) == n):
        raise ValueError(
            f"{prefix}length mismatch: customer_id={n}, D_impl={len(D_i)}, "
            f"y_impl={len(y_i)}"
        )
    if n == 0:
        return
    if len(np.unique(cid)) != n:
        raise ValueError(f"{prefix}duplicate customer_id in implementation split")
    if cid.min() < 0 or cid.max() >= len(D_all):
        raise ValueError(
            f"{prefix}customer_id out of range [0, {len(D_all)}): "
            f"min={cid.min()}, max={cid.max()}"
        )
    if not np.array_equal(D_i, D_all[cid]):
        bad = np.where(D_i != D_all[cid])[0][:5]
        raise ValueError(
            f"{prefix}D_impl does not match D_cohort[customer_id] at rows {bad.tolist()}"
        )
    if not np.allclose(y_i, y_all[cid], rtol=0.0, atol=0.0, equal_nan=True):
        bad = np.where(~np.isclose(y_i, y_all[cid], equal_nan=True))[0][:5]
        raise ValueError(
            f"{prefix}y_impl does not match y_cohort[customer_id] at rows {bad.tolist()}"
        )


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

    X_tr, X_val, D_tr, D_val, y_tr, y_val, Gamma_tr, Gamma_val = train_test_split(
        X_pilot,
        D_pilot,
        y_pilot,
        Gamma_pilot,
        test_size=test_frac,
        random_state=0,
    )

    if hasattr(X_tr, "reset_index"):
        X_tr = X_tr.reset_index(drop=True).values
        X_val = X_val.reset_index(drop=True).values
    elif hasattr(X_tr, "values"):
        X_tr = X_tr.values
        X_val = X_val.values

    if hasattr(D_tr, "reset_index"):
        D_tr = D_tr.reset_index(drop=True).values
        D_val = D_val.reset_index(drop=True).values
    elif hasattr(D_tr, "values"):
        D_tr = D_tr.values
        D_val = D_val.values

    if hasattr(y_tr, "reset_index"):
        y_tr = y_tr.reset_index(drop=True).values
        y_val = y_val.reset_index(drop=True).values
    elif hasattr(y_tr, "values"):
        y_tr = y_tr.values
        y_val = y_val.values

    return (X_tr, D_tr, y_tr, Gamma_tr), (X_val, D_val, y_val, Gamma_val)



HILLSTROM_CONTROL = "No E-Mail"
HILLSTROM_ACTION_ORDER = (
    HILLSTROM_CONTROL,   # 0 = holdout (no cost / regression reference)
    "Mens E-Mail",       # 1
    "Womens E-Mail",     # 2
)


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

    # ====== treatment mapping: control (No E-Mail) = 0, treatments = 1, 2 ======
    unique_segments = sorted(D.unique())
    seg2id = {seg: i for i, seg in enumerate(HILLSTROM_ACTION_ORDER)}
    unknown = set(unique_segments) - set(seg2id)
    if unknown:
        raise ValueError(
            f"Unexpected Hillstrom treatment labels: {sorted(unknown)}. "
            f"Expected {list(HILLSTROM_ACTION_ORDER)}."
        )
    D_np = D.map(seg2id).astype(int).values

    # ====== debugging info ======
    print("\n" + "=" * 60)
    print("DATA EXPLORATION (Hillstrom)")
    print("=" * 60)
    print("\n Basic Information:")
    print(f"   X shape: {X_scaled.shape} (n={X_scaled.shape[0]}, d={X_scaled.shape[1]})")
    print(f"   Unique treatments: {unique_segments}")
    print(f"   Mapped as: {seg2id}  (0={HILLSTROM_CONTROL!r} is control)")
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


# =========================================================
# 1. pilot / implementation 划分 + outcome model + Gamma (K-action)
# =========================================================
def prepare_pilot_impl(
    X, y, D, pilot_frac, mu_model_type, return_impl_customer_id=False, mu_hparams=None
):
    """
    K-action 版本
    """
    print("\n" + "=" * 60)
    print("Split & fit outcome models")
    print("=" * 60)

    # Always split with cohort indices so pilot/impl membership does not depend
    # on whether offline storage is enabled later in run_sims.py.
    (
        X_pilot,
        X_impl,
        D_pilot,
        D_impl,
        y_pilot,
        y_impl,
        _idx_pilot,
        impl_customer_id,
    ) = split_pilot_impl(
        X, D, y, pilot_frac=pilot_frac, return_indices=True
    )
    impl_customer_id = np.asarray(impl_customer_id, dtype=int)
    print(f"Pilot size: {len(X_pilot)}, Implementation size: {len(X_impl)}")

    X_pilot = np.asarray(X_pilot)
    X_impl = np.asarray(X_impl)
    D_pilot = np.asarray(D_pilot).astype(int)
    D_impl = np.asarray(D_impl).astype(int)
    y_pilot = np.asarray(y_pilot, dtype=float)
    y_impl = np.asarray(y_impl, dtype=float)

    verify_impl_customer_alignment(
        impl_customer_id,
        D_impl,
        y_impl,
        D,
        y,
        context="prepare_pilot_impl",
    )

    y_fit = y_pilot

    # ---- 2) fit μ_a models----
    mu_pilot_models = fit_mu_models(
        X_pilot,
        D_pilot,
        y_fit,
        mu_model_type=mu_model_type,
        mu_hparams=mu_hparams,
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

    # ---- 3) build Gamma_pilot: (N, K) ----
    Gamma_pilot = np.zeros((N_pilot, K), dtype=float)

    for a in actions:
        mask_a = (D_pilot == a)
        e_a = max(mask_a.mean(), 1e-6)

        model_a = mu_pilot_models[a]
        is_clf = hasattr(model_a, "predict_proba")

        mu_a_hat = predict_mu_values(mu_pilot_models[a], X_pilot)

        # （可选）一致性检查：clf 时 y 必须二元
        if is_clf:
            uy = np.unique(y_pilot)
            if not set(uy.tolist()).issubset({0, 1}):
                raise ValueError("Classifier mu requires y_pilot in {0,1}.")

        Gamma_pilot[:, a] = mu_a_hat + (mask_a.astype(float) / e_a) * (y_pilot - mu_a_hat)

        
    out = (
        X_pilot,
        X_impl,
        D_pilot,
        D_impl,
        y_pilot,
        y_impl,
        mu_pilot_models,
        Gamma_pilot,
    )
    if return_impl_customer_id:
        return out + (impl_customer_id,)
    return out  # impl_customer_id verified but not returned
