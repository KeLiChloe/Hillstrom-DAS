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

from data_utils import (
    load_criteo, load_hillstrom, load_lenta,
    split_seg_train_test, prepare_pilot_impl
)

from estimation import estimate_segment_policy
from evaluation import evaluate_policy_dual_dr, _build_mu_matrix, evaluate_policy_dr, evaluate_policy_ipw, _get_propensity_per_action  # 已改成多 action 版
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

M_candidates = [2, 3, 4, 5, 6, 7, 8]


def run_single_experiment(sample_frac, pilot_frac, train_frac, dataset, target_col):
    # --------------------------------------------------
    # Load dataset based on parameter
    # --------------------------------------------------
    seed = np.random.randint(0, 1_000_000)
    
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
    log_y = False
    
    (
        X_pilot,
        X_impl,
        D_pilot,
        D_impl,
        y_pilot,
        y_impl,
        mu_pilot_models,   # dict[a] = model_a
        Gamma_pilot,       # (N_pilot, K)
    ) = prepare_pilot_impl(X, y, D, pilot_frac=pilot_frac, model_type="logistic", log_y=log_y)

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
        "random": {},
    }
    
    # 动态添加 "all_a" baseline 占位符（针对每个 action a）
    for a in actions_all:
        results[f"all_{a}"] = {}

    for algo in ALGO_LIST:
        results[algo] = {}

    
    if "causal_forest" in ALGO_LIST:
        t0 = time.perf_counter()
        cf_model = fit_multiarm_causal_forest(
            X_pilot,
            y_pilot,
            D_pilot,
            action_levels=np.arange(action_K),   # 确保列顺序与 0..K-1 对齐
            num_trees=100,
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
                log_y=log_y,
            )
            results["causal_forest"][f"{eval}"] = float(value_cf["value_mean"])
           
            
        t1 = time.perf_counter()
        results["causal_forest"]["time"] = float(t1 - t0)
    
        print("causal forest finished")
    
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
            log_y=log_y,
            random_state=seed,
        )

        mu_mat_impl_t = predict_mu_t_learner_matrix(
            t_models,
            X_impl,
            log_y=log_y,
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
                log_y=log_y,
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
            log_y=log_y,
            random_state=seed,
        )

        mu_mat_impl_s = predict_mu_s_learner_matrix(
            s_model,
            X_impl,
            K=action_K,
            log_y=log_y,
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
                log_y=log_y,
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
            log_y=log_y,
            control_action=0,        # Hillstrom: 通常 0 是 control
            random_state=seed,
        )

        # 2) predict individual best action on IMPLEMENTATION set
        a_hat_x, mu_hat_x = predict_best_action_x_learner(
            x_learner_models=x_models,
            X=X_impl,
            mu_pilot_models=mu_pilot_models,
            log_y=log_y,
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
                log_y=log_y,
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
                n_folds=3,
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
                log_y=log_y,
            )
            results["dr_learner"][f"{eval}"] = float(value_dr["value_mean"])

        t1 = time.perf_counter()
        results["dr_learner"]["time"] = float(t1 - t0)


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
                log_y=log_y,
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
                log_y=log_y,
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
                log_y=log_y,
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
                log_y=log_y,
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
                log_y=log_y,
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
                log_y=log_y,
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
            best_action_dast_train,
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
        )
        
        results["dast"]["best_M"] = best_M_dast
        
        action_dast = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_dast
        )
        seg_labels_impl_dast = tree_final.assign(X_impl)
        for eval in eval_methods:
            value_dast = eval_classes[eval](
                X_impl,
                D_impl,
                y_impl,
                seg_labels_impl_dast,
                mu_pilot_models,
                action_dast,
                propensities=None,
                log_y=log_y,
            )
            results["dast"][f"{eval}"] = float(value_dast["value_mean"])
            
        t1 = time.perf_counter()
        results["dast"]["time"] = float(t1 - t0)
        
        print(
            f"DAST - Segments: {len(np.unique(seg_labels_pilot_dast))}, "
            f"Actions: {action_dast}",
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
                log_y=log_y,
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
                log_y=log_y,
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
):
    experiment_data = {
        "params": {
            "sample_frac": sample_frac,
            "pilot_frac": pilot_frac,
            "train_frac": train_frac,
            "N_sim": N_sim,
            "dataset": dataset,
            "target_col": target_col,
        },
        "results": [],
    }

    print("\n" + "=" * 60)
    print(f"STARTING SIMULATIONS: N_sim = {N_sim}")
    print(f"Dataset: {dataset}, Target: {target_col}")
    print("=" * 60)

    for s in range(N_sim):
        try:
            res = run_single_experiment(
                sample_frac=sample_frac,
                pilot_frac=pilot_frac,
                train_frac=train_frac,
                dataset=dataset,
                target_col=target_col,
            )

            experiment_data["results"].append(res)

            # 每轮覆盖保存
            with open(out_path, "wb") as f:
                pickle.dump(experiment_data, f)

            print(f'[SIM {len(experiment_data["results"])}/{N_sim}] saved → {out_path}')
            print("-" * 60)

        except Exception:
            import traceback

            traceback.print_exc()
            continue

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

    args = parser.parse_args()

    pilot_frac = 0.2  # 20% data for pilot
    train_frac = 0.7  # 70% pilot for training

    run_multiple_experiments(
        N_sim=100,
        sample_frac=0.05,
        pilot_frac=pilot_frac,
        train_frac=train_frac,
        out_path=args.outpath,
        dataset=args.dataset,
        target_col=args.target,
    )
