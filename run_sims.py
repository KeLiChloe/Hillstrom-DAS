"""
多次重复（ Hillstrom）实验，并保存每次各算法（包括 CLR）的 value_mean 到 pkl。

⚠ 现在是 K-action 版本：
  - D 可以是 {0,1,...,K-1}，例如 Hillstrom 三个 action。
  - prepare_pilot_impl 返回 mu_pilot_models: dict[action] -> outcome model
  - Gamma_pilot shape = (N_pilot, K)
  - evaluator 使用多 action 版 evaluate_policy_dual_dr
"""

import numpy as np
import pickle
import os
import time
import random
import concurrent.futures as cf
import contextlib
import io


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
        )

from data_utils import (
    load_criteo, load_hillstrom, load_lenta,
    split_seg_train_test, prepare_pilot_impl
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
    run_policytree_segmentation,
)

# 你目前只用 dual_dr
ALGO_LIST = ["causal_forest", "dast", "mst", "clr", "kmeans", "gmm", "t_learner", "s_learner", "x_learner", "dr_learner"] #
# ALGO_LIST = ["kmeans", "kmeans_dams", "gmm", "gmm_dams", "clr", "clr_dams", "dast"]

eval_methods = ["dr", "dual_dr", "ipw"]

eval_classes = {
    "dr": evaluate_policy_dr,
    "dual_dr": evaluate_policy_dual_dr,  # 多 action 版
    "ipw": evaluate_policy_ipw
}

M_candidates = [2, 3, 4, 5, 6, 7, 8, 9, 10]


def run_single_experiment(sample_frac, pilot_frac, train_frac, dataset, target_col, mu_model_type, value_type_dast, value_type_dams, seed):
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
    
    (
        X_pilot,
        X_impl,
        D_pilot,
        D_impl,
        y_pilot,
        y_impl,
        mu_pilot_models,   # dict[a] = model_a
        Gamma_pilot,       # (N_pilot, K)
    ) = prepare_pilot_impl(X, y, D, pilot_frac=pilot_frac, mu_model_type=mu_model_type)

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

    # storage for output
    results = {
        "seed": int(seed),
    }
    

    for algo in ALGO_LIST:
        results[algo] = {}

    
    
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
            results["t_learner"][f"{eval}"] = float(value_t["value_mean"])

        t1 = time.perf_counter()
        results["t_learner"]["time"] = float(t1 - t0)
    
        
    
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
            results["s_learner"][f"{eval}"] = float(value_s["value_mean"])

        
        t1 = time.perf_counter()
        results["s_learner"]["time"] = float(t1 - t0)   
        
    
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
            results["x_learner"][f"{eval}"] = float(value_x["value_mean"])  
        
        t1 = time.perf_counter()
        results["x_learner"]["time"] = float(t1 - t0)    
    
    
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
            results["dr_learner"][f"{eval}"] = float(value_dr["value_mean"])

        t1 = time.perf_counter()
        results["dr_learner"]["time"] = float(t1 - t0)
    
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
            results["causal_forest"][f"{eval}"] = float(value_cf["value_mean"])
           
            
        t1 = time.perf_counter()
        results["causal_forest"]["time"] = float(t1 - t0)
        print("causal forest finished")

    


    # --------------------------------------------------
    # 4a. KMeans
    # --------------------------------------------------
    if "kmeans" in ALGO_LIST:
        t0 = time.perf_counter()
        kmeans_seg, seg_labels_pilot_kmeans, best_M_kmeans = run_kmeans_segmentation(
            X_pilot, M_candidates=M_candidates, random_state=seed
        )
        results["kmeans"]["best_M"] = best_M_kmeans
        action_kmeans = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_kmeans
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
            results["kmeans"][f"{eval}"] = float(value_kmeans["value_mean"])
            
        t1 = time.perf_counter()
        results["kmeans"]["time"] = float(t1 - t0)
        
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
            )
        )
        results["kmeans_dams"]["best_M"] = best_M_kmeans_dams
        
        action_kmeans_dams = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_kmeans_dams
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
            results["kmeans_dams"][f"{eval}"] = float(value_kmeans_dams["value_mean"])
        
        t1 = time.perf_counter()
        results["kmeans_dams"]["time"] = float(t1 - t0)
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
        results["gmm"]["best_M"] = best_M_gmm
        
        action_gmm = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_gmm
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
            results["gmm"][f"{eval}"] = float(value_gmm["value_mean"])
            
        t1 = time.perf_counter()
        results["gmm"]["time"] = float(t1 - t0)
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
            )
        )
        
        results["gmm_dams"]["best_M"] = best_M_gmm_dams
        
        action_gmm_dams = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_gmm_dams
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
            results["gmm_dams"][f"{eval}"] = float(value_gmm_dams["value_mean"])
            
        t1 = time.perf_counter()
        results["gmm_dams"]["time"] = float(t1 - t0)
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
        results["clr"]["best_M"] = best_M_clr
        
        action_clr = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_clr
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
            results["clr"][f"{eval}"] = float(value_clr["value_mean"])
            
        t1 = time.perf_counter()
        results["clr"]["time"] = float(t1 - t0)
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
            )
        )
        results["clr_dams"]["best_M"] = best_M_clr_dams
        action_clr_dams = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_clr_dams
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
            results["clr_dams"][f"{eval}"] = float(value_clr_dams["value_mean"])
        t1 = time.perf_counter()
        results["clr_dams"]["time"] = float(t1 - t0)
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
        )
        
        results["dast"]["best_M"] = best_M_dast

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
            results["dast"][f"{eval}"] = float(value_dast["value_mean"])
            
        t1 = time.perf_counter()
        results["dast"]["time"] = float(t1 - t0)
        
        print(
            f"DAST - Segments: {len(np.unique(seg_labels_pilot_dast))}, "
            f"Actions: {best_action_dast_pilot}",
        )

    # MST
    if "mst" in ALGO_LIST:
        t0 = time.perf_counter()
        tree_mst, seg_labels_pilot_mst, best_M_mst, action_mst = run_mst_dams(
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
        )
        results["mst"]["best_M"] = best_M_mst

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
            results["mst"][f"{eval}"] = float(value_mst["value_mean"])
        
        t1 = time.perf_counter()
        results["mst"]["time"] = float(t1 - t0)


        # ---- Causal Forest benchmark (grf multi_arm_causal_forest) ----
    

    # Policytree (R based) — 如果你已经升级成多 action 版 policytree_segmentation
    if "policytree" in ALGO_LIST:
        t0 = time.perf_counter()
        (
            policy_seg,
            seg_labels_pilot_policy,
            best_M_policytree,
            best_action_policytree,
        ) = run_policytree_segmentation(
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
            value_type_dams=value_type_dams,
        )
        
        results["policytree"]["best_M"] = best_M_policytree 
        
        action_policy = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_policy
        )
        seg_labels_impl_policy = policy_seg.assign(X_impl)
        for eval in eval_methods:
            value_policy = eval_classes[eval](
                X_impl,
                D_impl,
                y_impl,
                seg_labels_impl_policy,
                mu_pilot_models,
                action_policy,
                propensities=None,
                 
            )
            results["policytree"][f"{eval}"] = float(value_policy["value_mean"])
        
        
        t1 = time.perf_counter()
        results["policytree"]["time"] = float(t1 - t0)

        print(
            f"PolicyTree - Segments: {len(np.unique(seg_labels_pilot_policy))}, "
            f"Actions: {action_policy}, Time: {t1 - t0:.2f} seconds",
        )

    # --------------------------------------------------
    # 输出 summary
    # --------------------------------------------------
    print("\nResult for this run:")
    for k, v in results.items():
        if "time" not in k:
            print(f"{k:20s}: {v}")

    return results


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
):
    # 并行配置：
    # - inner_threads 写死为 1，避免每个进程内部再开多线程导致过度并行
    inner_threads = 1
    n_jobs = int(n_jobs)

    # 预先生成每一轮的 seed（并行时不要在 worker 里用全局 random）
    seeds = [random.randint(0, 1_000_000) for _ in range(N_sim)]

    experiment_data = {
        "params": {
            "seed_sequence": seed_sequence,
            "sample_frac": sample_frac,
            "pilot_frac": pilot_frac,
            "train_frac": train_frac,
            "N_sim": N_sim,
            "dataset": dataset,
            "target_col": target_col,
            "value_type_dast": value_type_dast,
            "value_type_dams": value_type_dams,
            "mu_model_type": mu_model_type,
            "out_path": out_path,
            "n_jobs": n_jobs,
            "inner_threads": inner_threads,
            "ALGO_LIST": list(ALGO_LIST),
            "eval_methods": list(eval_methods),
            "M_candidates": list(M_candidates),
            "seeds": list(seeds),
        },
        "results": [],
    }

    print("Experiment parameters:")
    for k, v in experiment_data["params"].items():
        # seeds 太长，打印时折叠
        if k == "seeds":
            print(f"  {k:15s}: <list len={len(v)}>")
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
        }

    # 串行：保持原来的“每轮覆盖保存”（更抗中断）
    if int(n_jobs) <= 1:
        _set_thread_env(inner_threads)
        for s, seed in enumerate(seeds):
            try:
                # 串行模式保留详细日志，便于调试
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
                )
                experiment_data["results"].append(res)
                with open(out_path, "wb") as f:
                    pickle.dump(experiment_data, f)
                print(f'[SIM {len(experiment_data["results"])}/{N_sim}] saved → {out_path}')
                print("-" * 60)
            except Exception:
                import traceback
                traceback.print_exc()
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
        completed = 0
        with cf.ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            future_to_seed = {ex.submit(_run_single_experiment_worker, _payload(seed)): seed for seed in seeds}
            for fut in cf.as_completed(future_to_seed):
                try:
                    res = fut.result()
                    experiment_data["results"].append(res)
                    with open(out_path, "wb") as f:
                        pickle.dump(experiment_data, f)
                    completed += 1
                    elapsed = time.perf_counter() - t_start
                    pct = 100.0 * completed / float(N_sim)
                    # 动态进度行（同一行刷新）
                    print(
                        f"\r[SIM {completed}/{N_sim}] {pct:6.2f}% | elapsed {elapsed:8.1f}s | saved → {out_path}",
                        end="",
                        flush=True,
                    )

                    # 每完成 1 个 sim，额外打印摘要（主进程输出，避免 worker 刷屏）
                    seed_done = int(res.get("seed", future_to_seed.get(fut, -1)))
                    total_time = 0.0
                    for algo in ALGO_LIST:
                        if isinstance(res.get(algo), dict):
                            total_time += float(res[algo].get("time", 0.0) or 0.0)

                    parts = [f"seed={seed_done}", f"total_time={total_time:.1f}s"]
                    for algo in ALGO_LIST:
                        if not isinstance(res.get(algo), dict):
                            continue
                        # 优先显示 dual_dr；没有就退回 dr/ipw
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
                    continue
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

    args = parser.parse_args()

    pilot_frac = 0.2  # 20% data for pilot
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
    )
