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

from data_utils import split_seg_train_test
from data_utils import load_hillstrom, prepare_pilot_impl 
from estimation import estimate_segment_policy
from evaluation import evaluate_policy_dual_dr, _build_mu_matrix, evaluate_policy_dr, evaluate_policy_ipw   # 已改成多 action 版
from s_learner import fit_s_learner, predict_mu_s_learner_matrix
from dr_learner import fit_dr_learner_models, dr_learner_policy
from x_learner import fit_x_learner, predict_best_action_x_learner
from causal_forest import (
            fit_multiarm_causal_forest,
            predict_best_action_multiarm,
        )
from ot_lloyd_discrete import run_discrete_ot_lloyd_model_select



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
ALGO_LIST = ["causal_forest", "dast", "mst", "clr", "kmeans", "gmm", "t_learner", "x_learner", "s_learner", "dr_learner"] 
# ALGO_LIST = ["kmeans", "kmeans_dams", "gmm", "gmm_dams", "clr", "clr_dams", "dast"]

eval_methods = ["dr", "dual_dr", "ipw"]

eval_classes = {
    "dr": evaluate_policy_dr,
    "dual_dr": evaluate_policy_dual_dr,  # 多 action 版
    "ipw": evaluate_policy_ipw
}

M_candidates = [2, 3, 4, 5, 6, 7, 8]


def run_single_experiment(sample_frac, pilot_frac, train_frac):
    # --------------------------------------------------）
    # --------------------------------------------------
    seed = np.random.randint(0, 1_000_000)
    X, y, D = load_hillstrom(sample_frac=sample_frac, seed=seed, target_col="spend")

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
    ) = prepare_pilot_impl(X, y, D, pilot_frac=pilot_frac, model_type="mlp_reg", log_y=log_y)

    # K 个动作（0..K-1）
    K = Gamma_pilot.shape[1]
    actions_all = np.arange(K, dtype=int)

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
    "all_0":{},
    "all_1":{},
    "all_2":{},
    "random":{},    
    }

    for algo in ALGO_LIST:
        results[algo] = {}


    # --------------------------------------------------
    # Baselines — multi (K)-action 版本
    # --------------------------------------------------
    # 所有 baseline 都用“1 个 segment”的形式：seg_labels_impl = 0
    
    # seg_labels_single_seg = np.zeros(len(X_impl), dtype=int)
    # for a in actions_all:
    #     action_all_a = np.array([a], dtype=int)  # segment 0 -> action a
    #     for eval in eval_methods:
    #         value_all_a = eval_classes[eval](
    #             X_impl,
    #             D_impl,
    #             y_impl,
    #             seg_labels_single_seg,
    #             mu_pilot_models,
    #             action_all_a,
    #             log_y=log_y,
    #         )
    #         results[f"all_{a}"][f"{eval}"] = float(value_all_a["value_mean"])


    # # random baseline
    # seg_labels_random = np.random.randint(0, K, size=len(X_impl))
    # action_random = np.arange(K, dtype=int)
    # for eval in eval_methods:
    #     value_random = eval_classes[eval](
    #         X_impl,
    #         D_impl,
    #         y_impl,
    #         seg_labels_random,
    #         mu_pilot_models,
    #         action_random,
    #         log_y=log_y,
    #     )
    #     results["random"][f"{eval}"] = float(value_random["value_mean"])
    
    
    if "ot_lloyd" in ALGO_LIST:
        t0 = time.perf_counter()

        # 用 dual_dr 在 pilot 的 train/val 上选 best_M（你也可以换成 eval_methods 里任何一个）
        seg_ot, _, best_M_ot, action_ot = run_discrete_ot_lloyd_model_select(
            X_train=X_train,
            X_val=X_val,
            D_val=D_val,
            y_val=y_val,
            mu_models=mu_pilot_models,
            K=K,
            M_candidates=M_candidates,
            eval_fn=evaluate_policy_dual_dr,   # 用它来选 M
            log_y=log_y,
            seed=int(seed),
            max_iter=50,
            use_balanced_ot=True,              # 这一步更贴近“OT”的容量约束
            q=None,                            # None = 等分
        )

        results["ot_lloyd"]["best_M"] = int(best_M_ot)

        # apply to implementation set
        seg_labels_impl_ot = seg_ot.assign(X_impl)

        for eval in eval_methods:
            val_ot = eval_classes[eval](
                X_impl, D_impl, y_impl,
                seg_labels_impl_ot,
                mu_pilot_models,
                action_ot,          # 注意：这里 action_ot 长度 = L
                log_y=log_y,
            )
            results["ot_lloyd"][eval] = float(val_ot["value_mean"])

        t1 = time.perf_counter()
        results["ot_lloyd"]["time"] = float(t1 - t0)

    
    
    if "causal_forest" in ALGO_LIST:
        t0 = time.perf_counter()
        cf_model = fit_multiarm_causal_forest(
            X_pilot,
            y_pilot,
            D_pilot,
            action_levels=np.arange(K),   # 确保列顺序与 0..K-1 对齐
            num_trees=100,
            seed=int(seed),
        )
        a_hat_cf, _ = predict_best_action_multiarm(cf_model, X_impl)
        seg_labels_impl_cf = a_hat_cf.astype(int)      # (n,)
        action_identity = np.arange(K, dtype=int)      # segment m -> action m
        for eval in eval_methods:
            value_cf = eval_classes[eval](
                X_impl, D_impl, y_impl,
                seg_labels_impl_cf,
                mu_pilot_models,
                action_identity,
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
        K = Gamma_pilot.shape[1]
        mu_mat_impl = _build_mu_matrix(mu_pilot_models, X_impl, K, log_y=log_y)  # (n, K)
        a_hat = np.argmax(mu_mat_impl, axis=1).astype(int)

        seg_labels_impl = a_hat
        action_identity = np.arange(K, dtype=int)
        for eval in eval_methods:
            value_tlearner = eval_classes[eval](
                X_impl, D_impl, y_impl,
                seg_labels_impl,
                mu_pilot_models,
                action_identity,
                log_y=log_y,
            )
            results["t_learner"][f"{eval}"] = float(value_tlearner["value_mean"])
        
        t1 = time.perf_counter()
        results["t_learner"]["time"]= float(t1 - t0)   
        
    
    # --------------------------------------------------
    # ---- S-learner benchmark (single model mu(x,a) + argmax_a) ----
    # --------------------------------------------------
    if "s_learner" in ALGO_LIST:
        t0 = time.perf_counter()
        s_model = fit_s_learner(
            X_pilot,
            D_pilot,
            y_pilot,
            K=K,
            model_type="mlp_reg",   
            log_y=log_y,
            random_state=seed,
        )

        mu_mat_impl_s = predict_mu_s_learner_matrix(
            s_model,
            X_impl,
            K=K,
            log_y=log_y,
        )
        a_hat_s = np.argmax(mu_mat_impl_s, axis=1).astype(int)
        seg_labels_impl_s = a_hat_s
        action_identity = np.arange(K, dtype=int)
        for eval in eval_methods:
            value_s = eval_classes[eval](
                X_impl, D_impl, y_impl,
                seg_labels_impl_s,
                mu_pilot_models,        
                action_identity,
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
            use_rct_gate=True,       # 你的实验是 RCT -> 常数 gating
            min_samples_per_group=5,
            random_state=0,
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
        action_identity = np.arange(K, dtype=int)  # segment m -> action m

        for eval in eval_methods:
            value_x = eval_classes[eval](
                X_impl, D_impl, y_impl,
                seg_labels_impl_x,
                mu_pilot_models,
                action_identity,
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

        # 1) fit true DR-learner (CATE-style) on PILOT
        dr_model = fit_dr_learner_models(
            X_pilot=X_pilot,
            D_pilot=D_pilot,
            y_pilot=y_pilot,
            mu_pilot_models=mu_pilot_models,
            K=K,
            baseline=0,          # Hillstrom: 0 is control
            model_type="mlp",   # "ridge" / "mlp" / "lgbm"
        )

        # 2) predict individual best action on IMPLEMENTATION
        a_hat_dr, _ = dr_learner_policy(dr_model, X_impl)

        # 3) evaluate with your unified OPE interface
        seg_labels_impl_dr = a_hat_dr.astype(int)
        action_identity = np.arange(K, dtype=int)

        for eval in eval_methods:
            value_dr = eval_classes[eval](
                X_impl, D_impl, y_impl,
                seg_labels_impl_dr,
                mu_pilot_models,
                action_identity,
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
            X_pilot, M_candidates=M_candidates
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
):
    experiment_data = {
        "params": {
            "sample_frac": sample_frac,
            "pilot_frac": pilot_frac,
            "train_frac": train_frac,
            "N_sim": N_sim,
        },
        "results": [],
    }

    print("\n" + "=" * 60)
    print(f"STARTING SIMULATIONS: N_sim = {N_sim}")
    print("=" * 60)

    for s in range(N_sim):
        try:
            res = run_single_experiment(
                sample_frac=sample_frac,
                pilot_frac=pilot_frac,
                train_frac=train_frac,
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

    args = parser.parse_args()

    pilot_frac = 0.5  # 20% data for pilot
    train_frac = 0.7  # 70% pilot for training

    run_multiple_experiments(
        N_sim=100,
        sample_frac=1,
        pilot_frac=pilot_frac,
        train_frac=train_frac,
        out_path=args.outpath,
    )
