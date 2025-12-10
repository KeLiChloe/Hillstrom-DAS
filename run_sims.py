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
from evaluation import evaluate_policy_dual_dr  # 已改成多 action 版

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
ALGO_LIST = [ "dast", "mst", "clr", "kmeans", "gmm"] 
# ALGO_LIST = ["kmeans", "kmeans_dams", "gmm", "gmm_dams", "clr", "clr_dams", "dast"]

eval_method = "dual_dr"

eval_classes = {
    "dual_dr": evaluate_policy_dual_dr,  # 多 action 版
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
    (
        X_pilot,
        X_impl,
        D_pilot,
        D_impl,
        y_pilot,
        y_impl,
        mu_pilot_models,   # dict[a] = model_a
        Gamma_pilot,       # (N_pilot, K)
    ) = prepare_pilot_impl(X, y, D, pilot_frac=pilot_frac)

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
        "K": int(K),
    }

    # --------------------------------------------------
    # Baselines — K-action 版本
    # --------------------------------------------------
    # 所有 baseline 都用“1 个 segment”的形式：seg_labels_impl = 0
    seg_labels_single_seg = np.zeros(len(X_impl), dtype=int)

    # 1) all_a: everyone gets action a
    for a in actions_all:
        action_all_a = np.array([a], dtype=int)  # segment 0 -> action a
        value_all_a = eval_classes[eval_method](
            X_impl,
            D_impl,
            y_impl,
            seg_labels_single_seg,
            mu_pilot_models,
            action_all_a,
        )
        results[f"all_{a}"] = float(value_all_a["value_mean"])

    # 为了兼容旧版二元记号：K=2 时额外写入 all_control / all_treat
    if K == 2:
        results["all_control"] = results["all_0"]
        results["all_treat"] = results["all_1"]

    # 2) random baseline: 每个个体随机一个 action ∈ {0,...,K-1}
    seg_labels_random = np.random.randint(0, K, size=len(X_impl))
    # 定义 segment->action 为 identity: segment m -> action m
    action_random = np.arange(K, dtype=int)
    value_random = eval_classes[eval_method](
        X_impl,
        D_impl,
        y_impl,
        seg_labels_random,
        mu_pilot_models,
        action_random,
    )
    results["random"] = float(value_random["value_mean"])

    # --------------------------------------------------
    # 4a. KMeans
    # --------------------------------------------------
    if "kmeans" in ALGO_LIST:
        t0 = time.perf_counter()
        kmeans_seg, seg_labels_pilot_kmeans, best_M_kmeans = run_kmeans_segmentation(
            X_pilot, M_candidates=M_candidates
        )
        action_kmeans = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_kmeans
        )  # shape (M_k,), each in {0,...,K-1}
        seg_labels_impl_kmeans = kmeans_seg.assign(X_impl)
        value_kmeans = eval_classes[eval_method](
            X_impl,
            D_impl,
            y_impl,
            seg_labels_impl_kmeans,
            mu_pilot_models,
            action_kmeans,
        )
        t1 = time.perf_counter()
        results["kmeans"] = float(value_kmeans["value_mean"])
        results["time_kmeans"] = float(t1 - t0)
        results["best_M_kmeans"] = best_M_kmeans
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
        action_kmeans_dams = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_kmeans_dams
        )
        seg_labels_impl_kmeans_dams = kmeans_dams_seg.assign(X_impl)
        value_kmeans_dams = eval_classes[eval_method](
            X_impl,
            D_impl,
            y_impl,
            seg_labels_impl_kmeans_dams,
            mu_pilot_models,
            action_kmeans_dams,
        )
        t1 = time.perf_counter()
        results["kmeans_dams"] = float(value_kmeans_dams["value_mean"])
        results["time_kmeans_dams"] = float(t1 - t0)
        results["best_M_kmeans_dams"] = best_M_kmeans_dams
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
        action_gmm = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_gmm
        )
        seg_labels_impl_gmm = gmm_seg.assign(X_impl)
        value_gmm = eval_classes[eval_method](
            X_impl,
            D_impl,
            y_impl,
            seg_labels_impl_gmm,
            mu_pilot_models,
            action_gmm,
        )
        t1 = time.perf_counter()
        results["gmm"] = float(value_gmm["value_mean"])
        results["time_gmm"] = float(t1 - t0)
        results["best_M_gmm"] = best_M_gmm
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
        action_gmm_dams = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_gmm_dams
        )
        seg_labels_impl_gmm_dams = gmm_dams_seg.assign(X_impl)
        value_gmm_dams = eval_classes[eval_method](
            X_impl,
            D_impl,
            y_impl,
            seg_labels_impl_gmm_dams,
            mu_pilot_models,
            action_gmm_dams,
        )
        t1 = time.perf_counter()
        results["gmm_dams"] = float(value_gmm_dams["value_mean"])
        results["time_gmm_dams"] = float(t1 - t0)
        results["best_M_gmm_dams"] = best_M_gmm_dams
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
        action_clr = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_clr
        )
        seg_labels_impl_clr = clr_seg.assign(X_impl)
        value_clr = eval_classes[eval_method](
            X_impl,
            D_impl,
            y_impl,
            seg_labels_impl_clr,
            mu_pilot_models,
            action_clr,
        )
        t1 = time.perf_counter()
        results["clr"] = float(value_clr["value_mean"])
        results["time_clr"] = float(t1 - t0)
        results["best_M_clr"] = best_M_clr
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
        action_clr_dams = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_clr_dams
        )
        seg_labels_impl_clr_dams = clr_dams_seg.assign(X_impl)
        value_clr_dams = eval_classes[eval_method](
            X_impl,
            D_impl,
            y_impl,
            seg_labels_impl_clr_dams,
            mu_pilot_models,
            action_clr_dams,
        )
        t1 = time.perf_counter()
        results["clr_dams"] = float(value_clr_dams["value_mean"])
        results["time_clr_dams"] = float(t1 - t0)
        results["best_M_clr_dams"] = best_M_clr_dams
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
        action_dast = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_dast
        )
        seg_labels_impl_dast = tree_final.assign(X_impl)
        value_dast = eval_classes[eval_method](
            X_impl,
            D_impl,
            y_impl,
            seg_labels_impl_dast,
            mu_pilot_models,
            action_dast,
        )
        t1 = time.perf_counter()
        results["dast"] = float(value_dast["value_mean"])
        results["time_dast"] = float(t1 - t0)
        results["best_M_dast"] = best_M_dast
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

        print(
            f"MST - Segments: {len(np.unique(seg_labels_pilot_mst))}, "
            f"Actions: {action_mst}",
        )

        seg_labels_impl_mst = tree_mst.assign(X_impl)

        value_mst = eval_classes[eval_method](
            X_impl,
            D_impl,
            y_impl,
            seg_labels_impl_mst,
            mu_pilot_models,
            action_mst,
        )
        t1 = time.perf_counter()
        results["mst"] = float(value_mst["value_mean"])
        results["time_mst"] = float(t1 - t0)
        results["best_M_mst"] = best_M_mst

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
        action_policy = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, seg_labels_pilot_policy
        )
        seg_labels_impl_policy = policy_seg.assign(X_impl)
        value_policy = eval_classes[eval_method](
            X_impl,
            D_impl,
            y_impl,
            seg_labels_impl_policy,
            mu_pilot_models,
            action_policy,
        )
        
        
        t1 = time.perf_counter()
        results["policytree"] = float(value_policy["value_mean"])
        results["time_policytree"] = float(t1 - t0)
        results["best_M_policytree"] = best_M_policytree
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
