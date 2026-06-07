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
import gzip
import os
import time


def _pkl_dump(path: str, data) -> None:
    with gzip.open(path, "wb", compresslevel=6) as f:
        pickle.dump(data, f, protocol=4)


def _pkl_load(path: str):
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as f:
            return pickle.load(f)

from outcome_model import tau_model_type_from_mu
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
from dast import DASTree
from scoring import dams_score


def run_dast_dams_all_M(
    X_train, D_train, y_train, 
    X_val, D_val, y_val, 
    X_pilot, D_pilot,y_pilot,
    X_impl, D_impl, y_impl,
    results,
    mu_pilot_models,
    Gamma_train,
    Gamma_val,
    Gamma_pilot,
    M_candidates,
    min_leaf_size,
    value_type_dast,
    value_type_dams,
    action_method,
):
    print("\n" + "=" * 60)
    print("STEP 5: DAST - selecting optimal M via DAMS")
    print("=" * 60)

    d_full = X_pilot.shape[1]

    # Generate candidate thresholds (midpoints between unique values)
    bins = 200
    H_full = {}

    for j in range(d_full):
        col = X_pilot[:, j]
        unique_values = np.unique(col)

        if len(unique_values) > bins + 1:
            qs = np.linspace(0, 1, num=bins + 1)
            grid = np.quantile(col, qs)
            grid = np.unique(grid)
        else:
            grid = unique_values

        if len(grid) > 1:
            H_full[j] = (grid[:-1] + grid[1:]) / 2.0
        else:
            H_full[j] = grid

    best_M = None
    best_score = -np.inf

    for M in M_candidates:
        # Best-first growth: build a fresh tree directly to M leaves
        tree = DASTree(
            x=X_train,
            y=y_train,
            D=D_train,
            gamma=Gamma_train,
            candidate_thresholds=H_full,
            min_leaf_size=min_leaf_size,
            value_type_dast=value_type_dast,
            action_method=action_method,
        )
        tree.build(M)
        actual_leaves = len(tree._get_leaf_nodes())
        print(f"  Built tree for M={M}: actual leaves = {actual_leaves}")

        # segment labels on train + segment-level policy
        labels_train = tree.assign(X_train)
        action_M = estimate_segment_policy(
            X_train, y_train, D_train, labels_train,
            method=action_method, Gamma=Gamma_train,
        )

        # DAMS scoring on validation
        score_M = dams_score(
            seg_model=tree,
            X_val=X_val,
            D_val=D_val,
            y_val=y_val,
            Gamma_val=Gamma_val,
            action=action_M,
            value_type_dams=value_type_dams,
        )
        print(f"  DAST M={M} DAMS-score={score_M:.6f}")

        if score_M >= best_score:
            best_score = score_M
            best_M = M

    results["dast"]["best_M"] = int(best_M)
    print(f"  Selected best M = {best_M} with DAMS score = {best_score:.6f}")

    for M in M_candidates:
        # Best-first growth on pilot data
        tree_pilot = DASTree(
            x=X_pilot,
            y=y_pilot,
            D=D_pilot,
            gamma=Gamma_pilot,
            candidate_thresholds=H_full,
            min_leaf_size=min_leaf_size,
            value_type_dast=value_type_dast,
            action_method=action_method,
        )
        tree_pilot.build(M)
        actual_leaves_pilot = len(tree_pilot._get_leaf_nodes())
        print(f"  Built pilot tree for M={M}: actual leaves = {actual_leaves_pilot}")

        # segment labels on pilot + segment-level policy
        labels_M = tree_pilot.assign(X_pilot)
        action_M = estimate_segment_policy(
            X_pilot, y_pilot, D_pilot, labels_M,
            method=action_method, Gamma=Gamma_pilot,
        )

        seg_labels_impl_dast = tree_pilot.assign(X_impl)
        for eval in eval_methods:
            if eval not in results["dast"]:
                results["dast"][f"{eval}"] = {}
            value_dast = eval_classes[eval](
                X_impl,
                D_impl,
                y_impl,
                seg_labels_impl_dast,
                mu_pilot_models,
                action_M,
                propensities=None,
                 
            )
            results["dast"][f"{eval}"][f"{M}"] = float(value_dast["value_mean"])





# 你目前只用 dual_dr
ALGO_LIST = ["causal_forest", "dast", "t_learner", "s_learner", "x_learner", "dr_learner"] #
# ALGO_LIST = ["kmeans", "kmeans_dams", "gmm", "gmm_dams", "clr", "clr_dams", "dast"]

eval_methods = ["dr", "dual_dr", "ipw"]

eval_classes = {
    "dr": evaluate_policy_dr,
    "dual_dr": evaluate_policy_dual_dr,  # 多 action 版
    "ipw": evaluate_policy_ipw
}

M_candidates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def run_single_simulation(
    sample_frac,
    pilot_frac,
    train_frac,
    dataset,
    target_col,
    mu_model_type,
    action_method,
    value_type_dast="hybrid",
    value_type_dams="hybrid",
):
    # --------------------------------------------------
    # Load dataset based on parameter
    # --------------------------------------------------
    # seed = np.random.randint(0, 1_000_000)
    seed = 646647
    
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
        "random": {},
    }
    
    # 动态添加 "all_a" baseline 占位符（针对每个 action a）
    for a in actions_all:
        results[f"all_{a}"] = {}

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
            model_type=mu_model_type,
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
            model_type=mu_model_type,
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
            mu_model_type=mu_model_type,
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
                mu_model_type=mu_model_type,
                tau_model_type=tau_model_type_from_mu(mu_model_type),
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
                mu_model_type=mu_model_type,
                tau_model_type=tau_model_type_from_mu(mu_model_type),
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
    # 5–6. DAST
    # --------------------------------------------------
    if "dast" in ALGO_LIST:
        run_dast_dams_all_M(
            X_train, D_train, y_train,
            X_val, D_val, y_val,
            X_pilot, D_pilot, y_pilot,
            X_impl, D_impl, y_impl,
            results,
            mu_pilot_models,
            Gamma_train,
            Gamma_val,
            Gamma_pilot,
            M_candidates,
            min_leaf_size=5,
            value_type_dast=value_type_dast,
            value_type_dams=value_type_dams,
            action_method=action_method,
        )



    # --------------------------------------------------
    # 输出 summary
    # --------------------------------------------------
    print("\nResult for this run:")
    for k, v in results.items():
        if "time" not in k:
            print(f"{k:20s}: {v}")

    return results


def run_multiple_simulations(
    N_sim,
    sample_frac,
    pilot_frac,
    train_frac,
    out_path,
    dataset,
    target_col,
    mu_model_type,
    action_method,
    value_type_dast="hybrid",
    value_type_dams="hybrid",
):
    experiment_data = {
        "params": {
            "sample_frac": sample_frac,
            "pilot_frac": pilot_frac,
            "train_frac": train_frac,
            "N_sim": N_sim,
            "dataset": dataset,
            "target_col": target_col,
            "mu_model_type": mu_model_type,
            "value_type_dast": value_type_dast,
            "value_type_dams": value_type_dams,
            "out_path": out_path,
        },
        "results": [],
    }

    print("\n" + "=" * 60)
    print(f"STARTING SIMULATIONS: N_sim = {N_sim}")
    print(f"Dataset: {dataset}, Target: {target_col}")
    print("=" * 60)

    for s in range(N_sim):
        try:
            res = run_single_simulation(
                sample_frac=sample_frac,
                pilot_frac=pilot_frac,
                train_frac=train_frac,
                dataset=dataset,
                target_col=target_col,
                mu_model_type=mu_model_type,
                action_method=action_method,
                value_type_dast=value_type_dast,
                value_type_dams=value_type_dams,
            )

            experiment_data["results"].append(res)

            # 每轮覆盖保存
            _pkl_dump(out_path, experiment_data)

            print(f'[SIM {len(experiment_data["results"])}/{N_sim}] saved → {out_path}')
            print("-" * 60)

        except Exception:
            import traceback

            traceback.print_exc()
            continue

    print("\nALL SIMULATIONS DONE.")
    print(f"Results saved in '{out_path}'")
    return experiment_data


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    _REPO_ROOT = Path(__file__).resolve().parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    parser = argparse.ArgumentParser(
        description="Run all-M DAST experiment and plot DAST value vs. M."
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
        default="lightgbm_reg",
    )
    
    
    parser.add_argument(
        "--value_type_dast",
        type=str,
        help="Value type for DAST splitting ('dr' or 'hybrid')",
        default="hybrid",
    )
    
    parser.add_argument(
        "--value_type_dams",
        type=str,
        help="Value type for DAMS criterion ('dr' or 'hybrid')",
        default="hybrid",
    )

    parser.add_argument(
        "--N-sim",
        "--N_sim",
        type=int,
        default=1,
        dest="N_sim",
        help="Number of simulation runs (default: 1).",
    )
    parser.add_argument(
        "--fig-dir",
        "--fig_dir",
        default=None,
        help="Figure output directory (default: <pkl_parent>/figures).",
    )
    parser.add_argument(
        "--eval-method",
        "--eval_method",
        default="all",
        choices=["all", "dr", "dual_dr", "ipw"],
        help="OPE method(s) to plot after run (default: all).",
    )
    parser.add_argument(
        "--run-index",
        "--run_index",
        type=int,
        default=0,
        help="Which results[] entry to plot (default: 0).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting after the experiment finishes.",
    )
    parser.add_argument(
        "--action_method",
        type=str,
        choices=["diff_in_means", "gamma", "logistic"],
        required=True,
        help="Method to estimate segment-level action.",
    )

    args = parser.parse_args()

    if not args.outpath:
        parser.error("--outpath is required.")

    pilot_frac = 0.2  # 20% data for pilot
    train_frac = 0.7  # 70% pilot for training

    experiment_data = run_multiple_simulations(
        N_sim=args.N_sim,
        sample_frac=args.sample_frac,
        pilot_frac=pilot_frac,
        train_frac=train_frac,
        out_path=args.outpath,
        dataset=args.dataset,
        target_col=args.target,
        mu_model_type=args.mu_model_type,
        action_method=args.action_method,
        value_type_dast=args.value_type_dast,
        value_type_dams=args.value_type_dams,
    )

    if args.no_plot:
        print("Skipping plots (--no-plot).")
    elif not experiment_data.get("results"):
        print("No successful runs; skipping plots.")
    else:
        from analysis.plot_all_M import plot_experiment

        out_pkl = Path(args.outpath).expanduser().resolve()
        fig_dir = args.fig_dir
        if fig_dir is None:
            fig_dir = out_pkl.parent / "figures"
        plot_experiment(
            experiment_data,
            fig_dir=fig_dir,
            run_index=args.run_index,
            eval_method=args.eval_method,
        )
