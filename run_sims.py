"""
多次重复（ Hillstrom）实验，并保存每次各算法（包括 CLR）的 value_mean 到 pkl。

Pickle layout (optional implementation block per sim):

  data["params"]
  data["results"][i]          # i-th successful simulation (one run)
      ["seed"]
      ["dast"], ["t_learner"], ...   # scalar OPE + time (+ best_M)
      ["implementation"]            
          ["customer_id"], ["D"], ["y"], ["actions"][algo]

  There is NO top-level implementation blob; use data["results"][i]["implementation"].

⚠ 现在是 K-action 版本：
  - D 可以是 {0,1,...,K-1}，例如 Hillstrom 三个 action。
  - prepare_pilot_impl 返回 mu_pilot_models: dict[action] -> outcome model
  - Gamma_pilot shape = (N_pilot, K)
  - evaluator 使用多 action 版 evaluate_policy_dual_dr
"""

import numpy as np
import pickle
import gzip
import os
import time
import random
import concurrent.futures as cf
import contextlib
import io


# ---------------------------------------------------------------------------
# gzip-transparent pickle helpers
# ---------------------------------------------------------------------------
def _pkl_dump(path: str, data) -> None:
    """Write data as gzip-compressed pickle (protocol 4)."""
    with gzip.open(path, "wb", compresslevel=6) as f:
        pickle.dump(data, f, protocol=4)


def _pkl_load(path: str):
    """Load a pickle file regardless of whether it is gzip-compressed or not."""
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as f:
            return pickle.load(f)


def _set_thread_env(n: int):
    n = int(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    os.environ["HILLSTORM_SKLEARN_N_JOBS"] = str(n)


def _run_single_experiment_worker(payload: dict):
    """
    Top-level function so ProcessPoolExecutor can pickle it.
    payload must be pickleable.
    """
    inner_threads = int(payload.get("inner_threads", 1))
    _set_thread_env(inner_threads)

    # 并行时避免 worker 日志互相打架：把 stdout/stderr 静默掉
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run_single_experiment(
            sample_frac=payload["sample_frac"],
            pilot_frac=payload["pilot_frac"],
            train_frac=payload["train_frac"],
            dataset=payload["dataset"],
            target_col=payload["target_col"],
            mu_model_type=payload["mu_model_type"],
            value_type_dast=payload["value_type_dast"],
            value_type_dams=payload["value_type_dams"],
            seed=int(payload["seed"]),
            save_offline_data=bool(payload.get("save_offline_data", True)),
            action_method=payload.get("action_method", "diff_in_means"),
        )

from data_utils import (
    load_criteo,
    load_hillstrom,
    load_lenta,
    split_seg_train_test,
    prepare_pilot_impl,
    verify_impl_customer_alignment,
)

from estimation import estimate_segment_policy
from evaluation import evaluate_policy_dual_dr, evaluate_policy_dr, evaluate_policy_ipw, _get_propensity_per_action  # 已改成多 action 版
from t_learner import fit_t_learner, predict_mu_t_learner_matrix
from s_learner import fit_s_learner, predict_mu_s_learner_matrix
from dr_learner import ( dr_learner_policy_binary, fit_dr_learner_binary,
                           dr_learner_policy_k_armed,  fit_dr_learner_k_armed)
from x_learner import fit_x_learner, predict_best_action_x_learner
from causal_forest import (
            fit_multiarm_causal_forest,
            predict_best_action_multiarm,
        )


from segmentation import (
    run_kmeans_segmentation,
    run_kmeans_dams_segmentation,
    run_gmm_segmentation,
    run_gmm_dams_segmentation,
    run_dast_dams,
    run_clr_segmentation,
    run_clr_dams_segmentation,
    run_mst_dams,
)
from policytree import run_policytree_individual

POLICYTREE_DEPTH = 2

ALGO_LIST = ["dast", "causal_forest", "mst", "clr", "kmeans", "gmm", "t_learner", "s_learner", "x_learner", "dr_learner"] #
# ALGO_LIST = ["kmeans", "kmeans_dams", "gmm", "gmm_dams", "clr", "clr_dams", "dast"]

eval_methods = ["dr", "dual_dr", "ipw"]

eval_classes = {
    "dr":      evaluate_policy_dr,
    "dual_dr": evaluate_policy_dual_dr,
    "ipw":     evaluate_policy_ipw,
}

M_candidates = [2, 3, 4, 5, 6, 7, 8, 9, 10]


def _impl_actions_from_segments(seg_labels_impl, action_per_segment):
    """Map per-user segment ids to recommended actions (same as evaluation.py)."""
    seg = np.asarray(seg_labels_impl, dtype=int)
    action = np.asarray(action_per_segment, dtype=int)
    return action[seg].astype(int)


def _record_impl_action(impl_actions, algo, seg_labels_impl, action_per_segment=None):
    """Store per-implementation-customer assigned action for offline analysis."""
    if action_per_segment is None:
        impl_actions[algo] = np.asarray(seg_labels_impl, dtype=int).copy()
    else:
        impl_actions[algo] = _impl_actions_from_segments(
            seg_labels_impl, action_per_segment
        )


def resolve_impl_pkl_path(out_path: str, save_offline_data: bool) -> str:
    """Use *_imp.pkl suffix when storing implementation offline payloads."""
    if not save_offline_data:
        return out_path
    root, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".pkl"
        out_path = out_path + ext
        root, ext = os.path.splitext(out_path)
    if not root.endswith("_imp"):
        return root + "_imp" + ext
    return out_path


def _attach_sim_implementation(
    sim_result,
    impl_actions,
    *,
    impl_customer_id,
    D_impl,
    y_impl,
    D_cohort,
    y_cohort,
    cohort_size,
):
    """
  Attach implementation-phase data to one simulation dict (one entry in
  experiment_data["results"]).

    Arrays are sorted by customer_id ascending. Row k refers to customer_id[k]
    (cohort row index before pilot/implementation split).
    """
    verify_impl_customer_alignment(
        impl_customer_id,
        D_impl,
        y_impl,
        D_cohort,
        y_cohort,
        context="attach_sim_implementation",
    )
    customer_id = np.asarray(impl_customer_id, dtype=np.int32)
    D = np.asarray(D_impl, dtype=np.int8)
    y = np.asarray(y_impl, dtype=np.float32)
    n = len(customer_id)

    order = np.argsort(customer_id, kind="stable")
    customer_id = customer_id[order]
    D = D[order]
    y = y[order]

    actions = {}
    for algo, arr in impl_actions.items():
        a = np.asarray(arr, dtype=int)
        if len(a) != n:
            raise ValueError(
                f"implementation alignment error for {algo}: "
                f"len(actions)={len(a)} != n_impl={n}"
            )
        actions[algo] = a[order].astype(np.int8)

    verify_impl_customer_alignment(
        customer_id,
        D,
        y,
        D_cohort,
        y_cohort,
        context="attach_sim_implementation(sorted)",
    )

    sim_result["implementation"] = {
        "customer_id": customer_id,
        "D": D,
        "y": y,
        "actions": actions,
        "cohort_size": int(cohort_size),
        "n_impl": int(n),
    }


def run_single_experiment(
    sample_frac,
    pilot_frac,
    train_frac,
    dataset,
    target_col,
    mu_model_type,
    value_type_dast,
    value_type_dams,
    seed,
    action_method,
    save_offline_data=True,
):
    # --------------------------------------------------
    # Load dataset based on parameter
    # --------------------------------------------------

    # 根据 dataset 参数选择加载函数
    dataset_loaders = {
        "hillstrom": load_hillstrom,
        "criteo": load_criteo,
        "lenta": load_lenta,
    }
    
    if dataset not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(dataset_loaders.keys())}")
    
    loader = dataset_loaders[dataset]
    X, y, D = loader(sample_frac=sample_frac, seed=seed, target_col=target_col)

    # --------------------------------------------------
    # 1–3. pilot + outcome models + Gamma_pilot (K-action DR)
    # --------------------------------------------------

    prep_out = prepare_pilot_impl(
        X,
        y,
        D,
        pilot_frac=pilot_frac,
        mu_model_type=mu_model_type,
        return_impl_customer_id=save_offline_data,
    )
    if save_offline_data:
        (
            X_pilot,
            X_impl,
            D_pilot,
            D_impl,
            y_pilot,
            y_impl,
            mu_pilot_models,
            Gamma_pilot,
            impl_customer_id,
        ) = prep_out
        cohort_size = len(X)
    else:
        (
            X_pilot,
            X_impl,
            D_pilot,
            D_impl,
            y_pilot,
            y_impl,
            mu_pilot_models,
            Gamma_pilot,
        ) = prep_out

    # K 个动作（0..K-1）
    action_K = Gamma_pilot.shape[1]
    actions_all = np.arange(action_K, dtype=int)

    # segmentation 训练 / 验证划分（对 DAST / MST / *_DAMS 用）
    (
        X_train,
        D_train,
        y_train,
        Gamma_train,
    ), (
        X_val,
        D_val,
        y_val,
        Gamma_val,
    ) = split_seg_train_test(
        X_pilot, D_pilot, y_pilot, Gamma_pilot, test_frac=1 - train_frac
    )

    # One simulation record → appended as experiment_data["results"][i]
    sim_result = {
        "seed": int(seed),
    }
    impl_actions = {}

    for algo in ALGO_LIST:
        sim_result[algo] = {}

    
    
    # --------------------------------------------------
    # ---- Direct argmax benchmark (T-learner) ----
    # --------------------------------------------------
    
    if "t_learner" in ALGO_LIST:
        t0 = time.perf_counter()

        # ========== fit ==========
        t_models = fit_t_learner(
            X_pilot,
            D_pilot,
            y_pilot,
            K=action_K,
            model_type="mlp_reg",    # "ridge" / "mlp_reg" / "lightgbm_reg"
            random_state=seed,
        )

        mu_mat_impl_t = predict_mu_t_learner_matrix(
            t_models,
            X_impl,
        )

        a_hat_t = np.argmax(mu_mat_impl_t, axis=1).astype(int)
        seg_labels_impl_t = a_hat_t
        action_identity = np.arange(action_K, dtype=int)

        for eval in eval_methods:
            value_t = eval_classes[eval](
                X_impl, D_impl, y_impl,
                seg_labels_impl_t,
                mu_pilot_models,
                action_identity,
                propensities=None,
                 
            )
            sim_result["t_learner"][f"{eval}"] = float(value_t["value_mean"])

        t1 = time.perf_counter()
        sim_result["t_learner"]["time"] = float(t1 - t0)
        if save_offline_data:
            _record_impl_action(impl_actions, "t_learner", seg_labels_impl_t)

    # --------------------------------------------------
    # ---- S-learner benchmark (single model mu(x,a) + argmax_a) ----
    # --------------------------------------------------
    if "s_learner" in ALGO_LIST:
        t0 = time.perf_counter()
        s_model = fit_s_learner(
            X_pilot,
            D_pilot,
            y_pilot,
            K=action_K,
            model_type="mlp_reg",    # "ridge" / "mlp_reg" / "lightgbm_reg"
             
            random_state=seed,
        )

        mu_mat_impl_s = predict_mu_s_learner_matrix(
            s_model,
            X_impl,
            K=action_K,
             
        )
        a_hat_s = np.argmax(mu_mat_impl_s, axis=1).astype(int)
        seg_labels_impl_s = a_hat_s
        action_identity = np.arange(action_K, dtype=int)
        for eval in eval_methods:
            value_s = eval_classes[eval](
                X_impl, D_impl, y_impl,
                seg_labels_impl_s,
                mu_pilot_models,        
                action_identity,
                propensities=None,
                 
            )
            sim_result["s_learner"][f"{eval}"] = float(value_s["value_mean"])

        
        t1 = time.perf_counter()
        sim_result["s_learner"]["time"] = float(t1 - t0)
        if save_offline_data:
            _record_impl_action(impl_actions, "s_learner", seg_labels_impl_s)

    # --------------------------------------------------
    # ---- X-learner benchmark (one-vs-control) ----
    # --------------------------------------------------
    if "x_learner" in ALGO_LIST:
        t0 = time.perf_counter()
        x_models = fit_x_learner(
            X_pilot=X_pilot,
            D_pilot=D_pilot,
            y_pilot=y_pilot,
            mu_pilot_models=mu_pilot_models,
             
            control_action=0,        # Hillstrom: 通常 0 是 control
            random_state=seed,
        )

        # 2) predict individual best action on IMPLEMENTATION set
        a_hat_x, mu_hat_x = predict_best_action_x_learner(
            x_learner_models=x_models,
            X=X_impl,
            mu_pilot_models=mu_pilot_models,
             
        )

        # 3) evaluate with your existing multi-action dual DR evaluator
        # trick: treat each action as its own "segment id"
        seg_labels_impl_x = a_hat_x
        action_identity = np.arange(action_K, dtype=int)  # segment m -> action m

        for eval in eval_methods:
            value_x = eval_classes[eval](
                X_impl, D_impl, y_impl,
                seg_labels_impl_x,
                mu_pilot_models,
                action_identity,
                propensities=None,
                 
            )
            sim_result["x_learner"][f"{eval}"] = float(value_x["value_mean"])  
        
        t1 = time.perf_counter()
        sim_result["x_learner"]["time"] = float(t1 - t0)
        if save_offline_data:
            _record_impl_action(impl_actions, "x_learner", seg_labels_impl_x)

    # --------------------------------------------------
    # ---- DR-learner benchmark (learn policy from Gamma labels) ----
    # --------------------------------------------------
    if "dr_learner" in ALGO_LIST:
        t0 = time.perf_counter()
        pi_vec = _get_propensity_per_action(D_pilot, actions_all, propensities=None)
        # 1) fit true DR-learner (CATE-style) on PILOT
        if action_K > 2:
            dr_model = fit_dr_learner_k_armed(
                X=X_pilot,
                D=D_pilot,
                y=y_pilot,
                K=action_K,
                pi=pi_vec,  # length K
                baseline=0,          # Hillstrom: 0 is control
                n_folds=5,
                mu_model_type="mlp_reg",   # "ridge" / "mlp_reg" / "lightgbm_reg"
                tau_model_type="mlp_reg",
            )

            # 2) predict individual best action on IMPLEMENTATION
            a_hat_dr, _ = dr_learner_policy_k_armed(dr_model, X_impl)
        
        elif action_K == 2:
            e = float(pi_vec[1])
            dr_model = fit_dr_learner_binary(
                X=X_pilot,
                D=D_pilot,
                y=y_pilot,

                e=e,  # P(D=1)
                n_folds=3,
                mu_model_type="mlp_reg",   # "ridge" / "mlp_reg" / "lightgbm_reg"
                tau_model_type="mlp_reg",
            )

            # 2) predict individual best action on IMPLEMENTATION
            a_hat_dr, _ = dr_learner_policy_binary(dr_model, X_impl)
            
        # 3) evaluate with your unified OPE interface
        seg_labels_impl_dr = a_hat_dr.astype(int)
        action_identity = np.arange(action_K, dtype=int)

        for eval in eval_methods:
            value_dr = eval_classes[eval](
                X_impl, D_impl, y_impl,
                seg_labels_impl_dr,
                mu_pilot_models,
                action_identity,
                propensities=None,
                 
            )
            sim_result["dr_learner"][f"{eval}"] = float(value_dr["value_mean"])

        t1 = time.perf_counter()
        sim_result["dr_learner"]["time"] = float(t1 - t0)
        if save_offline_data:
            _record_impl_action(impl_actions, "dr_learner", seg_labels_impl_dr)

    if "causal_forest" in ALGO_LIST:
        print("causal forest started")
        t0 = time.perf_counter()
        cf_model = fit_multiarm_causal_forest(
            X_pilot,
            y_pilot,
            D_pilot,
            action_levels=np.arange(action_K),   # 确保列顺序与 0..K-1 对齐
            num_trees=10,
            seed=int(seed),
        )
        a_hat_cf, _ = predict_best_action_multiarm(cf_model, X_impl)
        seg_labels_impl_cf = a_hat_cf.astype(int)      # (n,)
        action_identity = np.arange(action_K, dtype=int)      # segment m -> action m
        for eval in eval_methods:
            value_cf = eval_classes[eval](
                X_impl, D_impl, y_impl,
                seg_labels_impl_cf,
                mu_pilot_models,
                action_identity,
                propensities=None,
                 
            )
            sim_result["causal_forest"][f"{eval}"] = float(value_cf["value_mean"])
           
            
        t1 = time.perf_counter()
        sim_result["causal_forest"]["time"] = float(t1 - t0)
        if save_offline_data:
            _record_impl_action(impl_actions, "causal_forest", seg_labels_impl_cf)
        print("causal forest finished")

    # --------------------------------------------------
    # 4a. KMeans
    # --------------------------------------------------
    if "kmeans" in ALGO_LIST:
        t0 = time.perf_counter()
        kmeans_seg, seg_labels_pilot_kmeans, best_M_kmeans = run_kmeans_segmentation(
            X_pilot, M_candidates=M_candidates, random_state=seed
        )
        sim_result["kmeans"]["best_M"] = best_M_kmeans
        action_kmeans = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_kmeans,
            method=action_method, Gamma=Gamma_pilot,
        )  # shape (M_k,), each in {0,...,K-1}
        seg_labels_impl_kmeans = kmeans_seg.assign(X_impl)
        for eval in eval_methods:
            value_kmeans = eval_classes[eval](
                X_impl,
                D_impl,
                y_impl,
                seg_labels_impl_kmeans,
                mu_pilot_models,
                action_kmeans,
                propensities=None,
                 
            )
            sim_result["kmeans"][f"{eval}"] = float(value_kmeans["value_mean"])
            
        t1 = time.perf_counter()
        sim_result["kmeans"]["time"] = float(t1 - t0)
        if save_offline_data:
            _record_impl_action(
                impl_actions, "kmeans", seg_labels_impl_kmeans, action_kmeans
            )
        print(
            f"KMeans - Segments: {len(np.unique(seg_labels_pilot_kmeans))}, "
            f"Actions: {action_kmeans}",
        )

    if "kmeans_dams" in ALGO_LIST:
        t0 = time.perf_counter()
        kmeans_dams_seg, seg_labels_pilot_kmeans_dams, best_M_kmeans_dams = (
            run_kmeans_dams_segmentation(
                X_pilot,
                X_train,
                D_train,
                y_train,
                X_val,
                D_val,
                y_val,
                Gamma_val,
                M_candidates=M_candidates,
                random_state=seed,
                value_type_dams=value_type_dams,
                action_method=action_method,
                Gamma_train=Gamma_train,
            )
        )
        sim_result["kmeans_dams"]["best_M"] = best_M_kmeans_dams
        
        action_kmeans_dams = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_kmeans_dams,
            method=action_method, Gamma=Gamma_pilot,
        )
        seg_labels_impl_kmeans_dams = kmeans_dams_seg.assign(X_impl)
        for eval in eval_methods:
            value_kmeans_dams = eval_classes[eval](
                X_impl,
                D_impl,
                y_impl,
                seg_labels_impl_kmeans_dams,
                mu_pilot_models,
                action_kmeans_dams,
                propensities=None,
                 
            )
            sim_result["kmeans_dams"][f"{eval}"] = float(value_kmeans_dams["value_mean"])
        
        t1 = time.perf_counter()
        sim_result["kmeans_dams"]["time"] = float(t1 - t0)
        if save_offline_data:
            _record_impl_action(
                impl_actions,
                "kmeans_dams",
                seg_labels_impl_kmeans_dams,
                action_kmeans_dams,
            )
        print(
            f"KMeans_DAMS - Segments: {len(np.unique(seg_labels_pilot_kmeans_dams))}, "
            f"Actions: {action_kmeans_dams}",
        )

    # --------------------------------------------------
    # 4b. GMM
    # --------------------------------------------------
    if "gmm" in ALGO_LIST:
        t0 = time.perf_counter()
        gmm_seg, seg_labels_pilot_gmm, best_M_gmm = run_gmm_segmentation(
            X_pilot,
            M_candidates=M_candidates,
            random_state=seed,
        )
        sim_result["gmm"]["best_M"] = best_M_gmm
        
        action_gmm = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_gmm,
            method=action_method, Gamma=Gamma_pilot,
        )
        seg_labels_impl_gmm = gmm_seg.assign(X_impl)
        for eval in eval_methods:
            value_gmm = eval_classes[eval](
                X_impl,
                D_impl,
                y_impl,
                seg_labels_impl_gmm,
                mu_pilot_models,
                action_gmm,
                propensities=None,
                 
            )
            sim_result["gmm"][f"{eval}"] = float(value_gmm["value_mean"])
            
        t1 = time.perf_counter()
        sim_result["gmm"]["time"] = float(t1 - t0)
        if save_offline_data:
            _record_impl_action(
                impl_actions, "gmm", seg_labels_impl_gmm, action_gmm
            )
        print(
            f"GMM - Segments: {len(np.unique(seg_labels_pilot_gmm))}, "
            f"Actions: {action_gmm}",
        )

    if "gmm_dams" in ALGO_LIST:
        t0 = time.perf_counter()
        gmm_dams_seg, seg_labels_pilot_gmm_dams, best_M_gmm_dams = (
            run_gmm_dams_segmentation(
                X_pilot,
                X_train,
                D_train,
                y_train,
                X_val,
                D_val,
                y_val,
                Gamma_val,
                M_candidates,
                random_state=seed,
                value_type_dams=value_type_dams,
                action_method=action_method,
                Gamma_train=Gamma_train,
            )
        )
        
        sim_result["gmm_dams"]["best_M"] = best_M_gmm_dams
        
        action_gmm_dams = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_gmm_dams,
            method=action_method, Gamma=Gamma_pilot,
        )
        seg_labels_impl_gmm_dams = gmm_dams_seg.assign(X_impl)
        for eval in eval_methods:
            value_gmm_dams = eval_classes[eval](
                X_impl,
                D_impl,
                y_impl,
                seg_labels_impl_gmm_dams,
                mu_pilot_models,
                action_gmm_dams,
                propensities=None,
                 
            )
            sim_result["gmm_dams"][f"{eval}"] = float(value_gmm_dams["value_mean"])
            
        t1 = time.perf_counter()
        sim_result["gmm_dams"]["time"] = float(t1 - t0)
        if save_offline_data:
            _record_impl_action(
                impl_actions,
                "gmm_dams",
                seg_labels_impl_gmm_dams,
                action_gmm_dams,
            )
        print(
            f"GMM_DAMS - Segments: {len(np.unique(seg_labels_pilot_gmm_dams))}, "
            f"Actions: {action_gmm_dams}",
        )

    # --------------------------------------------------
    # 4c. CLR
    # --------------------------------------------------
    if "clr" in ALGO_LIST:
        t0 = time.perf_counter()
        clr_seg, seg_labels_pilot_clr, best_M_clr = run_clr_segmentation(
            X_pilot,
            D_pilot,
            y_pilot,
            M_candidates,
            random_state=seed,
        )
        sim_result["clr"]["best_M"] = best_M_clr
        
        action_clr = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_clr,
            method=action_method, Gamma=Gamma_pilot,
        )
        seg_labels_impl_clr = clr_seg.assign(X_impl)
        for eval in eval_methods:
            value_clr = eval_classes[eval](
                X_impl,
                D_impl,
                y_impl,
                seg_labels_impl_clr,
                mu_pilot_models,
                action_clr,
                propensities=None,
                 
            )
            sim_result["clr"][f"{eval}"] = float(value_clr["value_mean"])
            
        t1 = time.perf_counter()
        sim_result["clr"]["time"] = float(t1 - t0)
        if save_offline_data:
            _record_impl_action(
                impl_actions, "clr", seg_labels_impl_clr, action_clr
            )
        print(
            f"CLR - Segments: {len(np.unique(seg_labels_pilot_clr))}, "
            f"Actions: {action_clr}",
        )

    if "clr_dams" in ALGO_LIST:
        t0 = time.perf_counter()
        clr_dams_seg, seg_labels_pilot_clr_dams, best_M_clr_dams = (
            run_clr_dams_segmentation(
                X_pilot,
                D_pilot,
                y_pilot,
                X_train,
                D_train,
                y_train,
                X_val,
                D_val,
                y_val,
                Gamma_val,
                M_candidates,
                random_state=seed,
                value_type_dams=value_type_dams,
                action_method=action_method,
                Gamma_train=Gamma_train,
            )
        )
        sim_result["clr_dams"]["best_M"] = best_M_clr_dams
        action_clr_dams = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_clr_dams,
            method=action_method, Gamma=Gamma_pilot,
        )
        seg_labels_impl_clr_dams = clr_dams_seg.assign(X_impl)
        for eval in eval_methods:
            value_clr_dams = eval_classes[eval](
                X_impl,
                D_impl,
                y_impl,
                seg_labels_impl_clr_dams,
                mu_pilot_models,
                action_clr_dams,
                propensities=None,
                 
            )
            sim_result["clr_dams"][f"{eval}"] = float(value_clr_dams["value_mean"])
        t1 = time.perf_counter()
        sim_result["clr_dams"]["time"] = float(t1 - t0)
        if save_offline_data:
            _record_impl_action(
                impl_actions,
                "clr_dams",
                seg_labels_impl_clr_dams,
                action_clr_dams,
            )
        print(
            f"CLR_DAMS - Segments: {len(np.unique(seg_labels_pilot_clr_dams))}, "
            f"Actions: {action_clr_dams}",
        )

    # --------------------------------------------------
    # 5–6. DAST
    # --------------------------------------------------
    if "dast" in ALGO_LIST:
        t0 = time.perf_counter()
        (
            tree_final,
            seg_labels_pilot_dast,
            best_M_dast,
            best_action_dast_pilot,
        ) = run_dast_dams(
            X_pilot,
            D_pilot,
            y_pilot,
            X_train,
            D_train,
            y_train,
            X_val,
            D_val,
            y_val,
            Gamma_pilot,
            Gamma_train,
            Gamma_val,
            M_candidates,
            min_leaf_size=5,
            value_type_dast=value_type_dast,
            value_type_dams=value_type_dams,
            action_method=action_method,
        )
        
        sim_result["dast"]["best_M"] = best_M_dast

        seg_labels_impl_dast = tree_final.assign(X_impl)
        for eval in eval_methods:
            value_dast = eval_classes[eval](
                X_impl,
                D_impl,
                y_impl,
                seg_labels_impl_dast,
                mu_pilot_models,
                best_action_dast_pilot,
                propensities=None,
                 
            )
            sim_result["dast"][f"{eval}"] = float(value_dast["value_mean"])
            
        t1 = time.perf_counter()
        sim_result["dast"]["time"] = float(t1 - t0)
        if save_offline_data:
            _record_impl_action(
                impl_actions,
                "dast",
                seg_labels_impl_dast,
                best_action_dast_pilot,
            )
        print(
            f"DAST - Segments: {len(np.unique(seg_labels_pilot_dast))}, "
            f"Actions: {best_action_dast_pilot}",
        )

    # MST
    if "mst" in ALGO_LIST:
        t0 = time.perf_counter()
        tree_mst, seg_labels_pilot_mst, best_M_mst = run_mst_dams(
            X_pilot,
            D_pilot,
            y_pilot,
            X_train,
            D_train,
            y_train,
            X_val,
            D_val,
            y_val,
            Gamma_val,
            M_candidates,
            min_leaf_size=5,
            value_type_dams=value_type_dams,
            action_method=action_method,
            Gamma_train=Gamma_train,
        )
        action_mst = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_mst,
            method=action_method, Gamma=Gamma_pilot,
        )
        sim_result["mst"]["best_M"] = best_M_mst

        print(
            f"MST - Segments: {len(np.unique(seg_labels_pilot_mst))}, "
            f"Actions: {action_mst}",
        )

        seg_labels_impl_mst = tree_mst.assign(X_impl)
        for eval in eval_methods:
            value_mst = eval_classes[eval](
                X_impl,
                D_impl,
                y_impl,
                seg_labels_impl_mst,
                mu_pilot_models,
                action_mst,
                propensities=None,
                 
            )
            sim_result["mst"][f"{eval}"] = float(value_mst["value_mean"])
        
        t1 = time.perf_counter()
        sim_result["mst"]["time"] = float(t1 - t0)
        if save_offline_data:
            _record_impl_action(
                impl_actions, "mst", seg_labels_impl_mst, action_mst
            )

        # ---- Causal Forest benchmark (grf multi_arm_causal_forest) ----
    

    # Policytree (R based) — individual policy: fit on pilot, predict per customer
    if "policytree" in ALGO_LIST:
        t0 = time.perf_counter()
        action_impl_policytree = run_policytree_individual(
            X_pilot, y_pilot, D_pilot,
            X_impl,
            depth=POLICYTREE_DEPTH,
        )

        sim_result["policytree"]["depth"] = POLICYTREE_DEPTH

        # Treat each impl customer as their own "segment" for OPE reuse
        seg_labels_individual = np.arange(len(X_impl))
        for eval in eval_methods:
            value_policy = eval_classes[eval](
                X_impl,
                D_impl,
                y_impl,
                seg_labels_individual,
                mu_pilot_models,
                action_impl_policytree,
                propensities=None,
            )
            sim_result["policytree"][f"{eval}"] = float(value_policy["value_mean"])

        t1 = time.perf_counter()
        sim_result["policytree"]["time"] = float(t1 - t0)

        if save_offline_data:
            _record_impl_action(impl_actions, "policytree", action_impl_policytree)

        n_treated = int((action_impl_policytree > 0).sum())
        print(
            f"PolicyTree (depth={POLICYTREE_DEPTH}) - "
            f"Treated: {n_treated}/{len(action_impl_policytree)}, "
            f"Time: {t1 - t0:.2f} seconds"
        )

    # --------------------------------------------------
    # 输出 summary
    # --------------------------------------------------
    if save_offline_data:
        _attach_sim_implementation(
            sim_result,
            impl_actions,
            impl_customer_id=impl_customer_id,
            D_impl=D_impl,
            y_impl=y_impl,
            D_cohort=np.asarray(D, dtype=int),
            y_cohort=np.asarray(y, dtype=float),
            cohort_size=cohort_size,
        )

    print("\nResult for this run:")
    for k, v in sim_result.items():
        if k == "implementation":
            impl = v
            print(
                f"{k:20s}: n_impl={impl['n_impl']}, cohort_size={impl['cohort_size']}, "
                f"algos={list(impl['actions'].keys())}"
            )
            continue
        if isinstance(v, dict):
            print(f"{k:20s}: {v}")
        else:
            print(f"{k:20s}: {v}")

    return sim_result


def _param_equal(a, b) -> bool:
    if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            pass
    return a == b


def _load_experiment_checkpoint(out_path: str, expected_params: dict) -> tuple[dict, bool]:
    """
    Load an existing checkpoint or return a fresh experiment_data dict.
    Returns (experiment_data, already_complete).
    """
    if not os.path.isfile(out_path):
        return {
            "params": dict(expected_params),
            "results": [],
        }, False

    experiment_data = _pkl_load(out_path)

    if not isinstance(experiment_data, dict):
        raise ValueError(f"Invalid checkpoint format in {out_path!r}.")
    if "params" not in experiment_data or "results" not in experiment_data:
        raise ValueError(f"Checkpoint missing params/results in {out_path!r}.")

    saved = experiment_data["params"]
    mismatches = []
    for key, expected in expected_params.items():
        if not _param_equal(saved.get(key), expected):
            mismatches.append(
                f"{key}: checkpoint={saved.get(key)!r}, expected={expected!r}"
            )
    if mismatches:
        raise ValueError(
            f"Checkpoint params do not match current run ({out_path!r}):\n  "
            + "\n  ".join(mismatches)
        )

    n_done = len(experiment_data["results"])
    n_sim = int(saved.get("N_sim", expected_params["N_sim"]))
    if n_done >= n_sim:
        print(f"Checkpoint already complete ({n_done}/{n_sim}): {out_path}")
        return experiment_data, True

    attempts = int(saved.get("attempts_used", 0))
    print(
        f"Resuming checkpoint ({n_done}/{n_sim} results, "
        f"{attempts} attempts used): {out_path}"
    )
    return experiment_data, False


def run_multiple_experiments(
    N_sim,
    sample_frac,
    pilot_frac,
    train_frac,
    out_path,
    dataset,
    target_col,
    mu_model_type,
    value_type_dast,
    value_type_dams,
    seed_sequence,
    n_jobs,
    action_method,
    save_offline_data=True,
):
    # 并行配置：
    # - inner_threads 写死为 1，避免每个进程内部再开多线程导致过度并行
    inner_threads = 1
    n_jobs = int(n_jobs)
    max_attempts = int(N_sim) * 5
    out_path = resolve_impl_pkl_path(out_path, save_offline_data)

    expected_params = {
        "seed_sequence": seed_sequence,
        "sample_frac": sample_frac,
        "pilot_frac": pilot_frac,
        "train_frac": train_frac,
        "N_sim": int(N_sim),
        "max_attempts": max_attempts,
        "dataset": dataset,
        "target_col": target_col,
        "value_type_dast": value_type_dast,
        "value_type_dams": value_type_dams,
        "mu_model_type": mu_model_type,
        "action_method": action_method,
        "out_path": out_path,
        "n_jobs": n_jobs,
        "inner_threads": inner_threads,
        "ALGO_LIST": list(ALGO_LIST),
        "eval_methods": list(eval_methods),
        "M_candidates": list(M_candidates),
        "seeds": [],
        "attempts_used": 0,
    }

    # 成功完成的 simulation 所用 seed 顺序写入 params["seeds"]（失败会重试新 seed，直到满 N_sim 条结果）
    experiment_data, already_complete = _load_experiment_checkpoint(
        out_path, expected_params
    )
    if already_complete:
        return

    if not experiment_data["results"]:
        experiment_data["params"].setdefault("seeds", [])
        experiment_data["params"].setdefault("attempts_used", 0)
    # 续跑时保留已有 seeds / attempts_used / results；新跑时 expected_params 已写入 params

    print("Experiment parameters:")
    for k, v in experiment_data["params"].items():
        # seeds 太长，打印时折叠
        if k == "seeds":
            print(f"  {k:15s}: <successful seeds, filled as runs complete>")
        else:
            print(f"  {k:15s}: {v}")

    def _payload(seed: int) -> dict:
        return {
            "sample_frac": sample_frac,
            "pilot_frac": pilot_frac,
            "train_frac": train_frac,
            "dataset": dataset,
            "target_col": target_col,
            "mu_model_type": mu_model_type,
            "value_type_dast": value_type_dast,
            "value_type_dams": value_type_dams,
            "seed": int(seed),
            "inner_threads": int(inner_threads),
            "save_offline_data": bool(save_offline_data),
            "action_method": action_method,
        }

    # 串行：直到凑满 N_sim 条成功结果（单次失败则换新 seed 重试）；最多 max_attempts 次单次运行
    if int(n_jobs) <= 1:
        _set_thread_env(inner_threads)
        attempts_used = int(experiment_data["params"].get("attempts_used", 0))
        while len(experiment_data["results"]) < N_sim:
            if attempts_used >= max_attempts:
                experiment_data["params"]["attempts_used"] = attempts_used
                _pkl_dump(out_path, experiment_data)
                raise RuntimeError(
                    f"Exceeded max_attempts={max_attempts} (completed runs); "
                    f"only {len(experiment_data['results'])}/{N_sim} successes. "
                    f"Partial results saved to {out_path!r}."
                )
            attempts_used += 1
            experiment_data["params"]["attempts_used"] = attempts_used
            seed = random.randint(0, 1_000_000)
            try:
                res = run_single_experiment(
                    sample_frac=sample_frac,
                    pilot_frac=pilot_frac,
                    train_frac=train_frac,
                    dataset=dataset,
                    target_col=target_col,
                    mu_model_type=mu_model_type,
                    value_type_dast=value_type_dast,
                    value_type_dams=value_type_dams,
                    seed=int(seed),
                    save_offline_data=save_offline_data,
                    action_method=action_method,
                )
                experiment_data["results"].append(res)
                experiment_data["params"]["seeds"].append(int(seed))
                _pkl_dump(out_path, experiment_data)
                print(f'[SIM {len(experiment_data["results"])}/{N_sim}] saved → {out_path}')
                print("-" * 60)
            except Exception:
                import traceback

                traceback.print_exc()
                _pkl_dump(out_path, experiment_data)
                continue
    else:
        # 并行：主进程负责按完成顺序收集结果并覆盖保存（同样抗中断）
        _set_thread_env(inner_threads)

        # 重要：并行 worker 里会同时调用 sklift 的 fetch_* 下载/解压数据，
        # 在首次运行/无缓存时容易发生并发下载冲突或长时间无输出。
        # 这里先在主进程预取一次数据，确保缓存就绪，再启动进程池。
        try:
            if dataset == "criteo":
                from sklift.datasets import fetch_criteo

                print("Prefetching Criteo dataset cache (main process)...", flush=True)
                fetch_criteo(
                    target_col=target_col,
                    treatment_col="treatment",
                    percent10=True,
                    return_X_y_t=True,
                )
            elif dataset == "hillstrom":
                from sklift.datasets import fetch_hillstrom

                print("Prefetching Hillstrom dataset cache (main process)...", flush=True)
                fetch_hillstrom(
                    target_col=target_col,
                    treatment_col="segment",
                    return_X_y_t=True,
                )
            elif dataset == "lenta":
                from sklift.datasets import fetch_lenta

                print("Prefetching Lenta dataset cache (main process)...", flush=True)
                fetch_lenta(
                    target_col=target_col,
                    treatment_col="treatment",
                    return_X_y_t=True,
                )
        except Exception as e:
            print(f"Prefetch failed (will continue anyway): {e}", flush=True)

        t_start = time.perf_counter()
        pending = {}
        # 与串行一致：上限统计「发起的单次实验次数」（submit），避免因并行在途任务导致远超 max_attempts 次实际运行
        attempts_used = int(experiment_data["params"].get("attempts_used", 0))

        def submit_one(pool_ex):
            nonlocal attempts_used
            if attempts_used >= max_attempts:
                return False
            s = random.randint(0, 1_000_000)
            fut = pool_ex.submit(_run_single_experiment_worker, _payload(s))
            pending[fut] = s
            attempts_used += 1
            experiment_data["params"]["attempts_used"] = attempts_used
            return True

        with cf.ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            for _ in range(min(int(n_jobs), N_sim)):
                if not submit_one(ex):
                    break

            # 已满 N_sim 成功则退出；否则在仍有预算时保持 pending 非空。
            while len(experiment_data["results"]) < N_sim:
                if not pending:
                    if attempts_used >= max_attempts:
                        break
                    if not submit_one(ex):
                        break
                done, _ = cf.wait(set(pending.keys()), return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    seed_used = pending.pop(fut)

                    # 同一轮可能多个 future 同时完成；已满 N_sim 后丢弃多余结果
                    if len(experiment_data["results"]) >= N_sim:
                        try:
                            fut.result()
                        except Exception:
                            pass
                        continue

                    try:
                        res = fut.result()
                        experiment_data["results"].append(res)
                        experiment_data["params"]["seeds"].append(int(seed_used))
                        _pkl_dump(out_path, experiment_data)
                        completed = len(experiment_data["results"])
                        elapsed = time.perf_counter() - t_start
                        pct = 100.0 * completed / float(N_sim)
                        print(
                            f"\r[SIM {completed}/{N_sim}] {pct:6.2f}% | elapsed {elapsed:8.1f}s | saved → {out_path}",
                            end="",
                            flush=True,
                        )

                        seed_done = int(res.get("seed", seed_used))
                        total_time = 0.0
                        for algo in ALGO_LIST:
                            if isinstance(res.get(algo), dict):
                                total_time += float(res[algo].get("time", 0.0) or 0.0)

                        parts = [f"seed={seed_done}", f"total_time={total_time:.1f}s"]
                        for algo in ALGO_LIST:
                            if not isinstance(res.get(algo), dict):
                                continue
                            if "dual_dr" in res[algo]:
                                parts.append(f"{algo}.dual_dr={res[algo]['dual_dr']:.4g}")
                            elif "dr" in res[algo]:
                                parts.append(f"{algo}.dr={res[algo]['dr']:.4g}")
                            elif "ipw" in res[algo]:
                                parts.append(f"{algo}.ipw={res[algo]['ipw']:.4g}")

                        print("\n  " + " | ".join(parts), flush=True)
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        _pkl_dump(out_path, experiment_data)

                    if len(experiment_data["results"]) < N_sim and attempts_used < max_attempts:
                        submit_one(ex)

            # 已有 N_sim 条成功结果，或已达尝试上限：吞掉仍在跑的任务
            while pending:
                done, _ = cf.wait(set(pending.keys()), return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    pending.pop(fut)
                    try:
                        fut.result()
                    except Exception:
                        pass

            if len(experiment_data["results"]) < N_sim:
                experiment_data["params"]["attempts_used"] = attempts_used
                _pkl_dump(out_path, experiment_data)
                raise RuntimeError(
                    f"Exceeded max_attempts={max_attempts} (submitted runs); "
                    f"only {len(experiment_data['results'])}/{N_sim} successes. "
                    f"Partial results saved to {out_path!r}."
                )

        print("")  # 换行收尾

    print("\nALL SIMULATIONS DONE.")
    print(f"Results saved in '{out_path}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run multiple multi-action segmentation experiments"
    )

    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
        help="Output pkl path",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hillstrom", "criteo", "lenta"],
        help="Dataset to use (default: criteo)",
    )
    
    parser.add_argument(
        "--target",
        type=str,
        help="Target column",
    )

    parser.add_argument(
        "--sample_frac",
        type=float,
    )

    parser.add_argument(
        "--pilot_frac",
        type=float,
        default=0.2,
        help="Fraction of data used as pilot set (default: 0.2)",
    )
    
    parser.add_argument(
        "--mu_model_type", 
        type=str,
        help="Model type for gamma estimation",
    )
    
    parser.add_argument(
        "--value_type_dast",
        type=str,
        help="Value type for DAST splitting ('dr' or 'hybrid')",
    )
    
    parser.add_argument(
        "--value_type_dams",
        type=str,
        help="Value type for DAMS criterion ('dr' or 'hybrid')",
    )
    
    # add seed_sequence if needed
    parser.add_argument(
        "--seed_sequence",
        type=int,
        help="Seed sequence for reproducibility",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel simulations to run (outer parallelism).",
    )

    parser.add_argument(
        "--action_method",
        type=str,
        choices=["diff_in_means", "gamma", "logistic"],
        required=True,
        help="Method to estimate segment-level action.",
    )

    args = parser.parse_args()

    pilot_frac = args.pilot_frac  # fraction of data for pilot
    train_frac = 0.7  # 70% pilot for training
    
    if args.seed_sequence is not None:
        random.seed(args.seed_sequence)
        print(f"Using fixed sequence seed: {args.seed_sequence}")

    run_multiple_experiments(
        N_sim=100,
        sample_frac=args.sample_frac,
        pilot_frac=pilot_frac,
        train_frac=train_frac,
        out_path=args.outpath,
        dataset=args.dataset,
        target_col=args.target,
        mu_model_type=args.mu_model_type,
        value_type_dast=args.value_type_dast,
        value_type_dams=args.value_type_dams,
        seed_sequence=args.seed_sequence if args.seed_sequence is not None else None,
        n_jobs=args.n_jobs,
        action_method=args.action_method,
    )
