"""
Sweep DAST segment count M and compare OPE curves to meta-learner baselines.

Pickle layout (consumed by analysis/plot_all_M.py):
  results[i]["dast"]["dual_dr"]["5"]  -> policy value at M=5
  results[i]["dast"]["stz"]["5"]["t_learner"]  -> STZ advantage vs baseline at M=5
  results[i]["dast"]["best_M"]        -> DAMS-selected M (same rule as run_sims)
  results[i]["t_learner"]["dual_dr"]  -> scalar baseline (no M axis)

  STZ implementation actions live in {target}_stz.pkl sidecar (see conversion_stz.pkl).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import gzip
import io
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causal_forest import fit_multiarm_causal_forest, predict_best_action_multiarm
from data_utils import (
    load_criteo,
    load_hillstrom,
    load_lenta,
    prepare_pilot_impl,
    verify_impl_customer_alignment,
)
from dr_learner import (
    dr_learner_policy_binary,
    dr_learner_policy_k_armed,
    fit_dr_learner_binary,
    fit_dr_learner_k_armed,
)
from evaluation import (
    _get_propensity_per_action,
    evaluate_policy_dr,
    evaluate_policy_dual_dr,
    evaluate_policy_ipw,
)
from outcome_model import META_LEARNER_MU_MODEL_TYPE, tau_model_type_from_mu
from s_learner import fit_s_learner, predict_mu_s_learner_matrix
from segmentation import run_dast_all_M_curves
from t_learner import fit_t_learner, predict_mu_t_learner_matrix
from x_learner import fit_x_learner, predict_best_action_x_learner

# Meta-learner baselines only (DAST is evaluated per M separately).
ALGO_LIST = ["causal_forest", "t_learner", "s_learner", "x_learner", "dr_learner"]

eval_methods = ["dr", "dual_dr", "ipw"]
eval_classes = {
    "dr": evaluate_policy_dr,
    "dual_dr": evaluate_policy_dual_dr,
    "ipw": evaluate_policy_ipw,
}

DEFAULT_M_CANDIDATES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
DEFAULT_EXP_ROOT = "exp_june"
TRAIN_FRAC = 0.7
MIN_LEAF_SIZE = 5


def resolve_out_path(
    *,
    exp_root: str,
    dataset: str,
    target_col: str,
    mu_model_type: str,
) -> Path:
    """exp_june/{dataset}/{target}/{mu_model_type}/all_M/{target}.pkl"""
    out_dir = Path(exp_root) / dataset / target_col / mu_model_type / "all_M"
    return (out_dir / f"{target_col}.pkl").resolve()


def resolve_stz_pkl_path(out_path: str) -> str:
    root, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".pkl"
    return root + "_stz" + ext


def _dast_impl_key(M: int) -> str:
    return f"dast_M{int(M)}"


def _impl_actions_from_segments(seg_labels_impl, action_per_segment):
    seg = np.asarray(seg_labels_impl, dtype=int)
    action = np.asarray(action_per_segment, dtype=int)
    return action[seg].astype(int)


def _record_impl_action(impl_actions, algo, seg_labels_impl, action_per_segment=None):
    if action_per_segment is None:
        impl_actions[algo] = np.asarray(seg_labels_impl, dtype=int).copy()
    else:
        impl_actions[algo] = _impl_actions_from_segments(
            seg_labels_impl, action_per_segment
        )


def _attach_sim_implementation(
    sim_result,
    impl_actions,
    *,
    impl_customer_id,
    D_impl,
    y_impl,
    D_cohort,
    y_cohort,
):
    verify_impl_customer_alignment(
        impl_customer_id,
        D_impl,
        y_impl,
        D_cohort,
        y_cohort,
        context="run_all_M.attach_sim_implementation",
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

    sim_result["implementation"] = {
        "customer_id": customer_id,
        "D": D,
        "y": y,
        "actions": actions,
        "cohort_size": int(len(D_cohort)),
        "n_impl": int(n),
    }


def _main_payload_for_save(experiment_data: dict) -> dict:
    return {
        "params": experiment_data["params"],
        "results": [
            {k: v for k, v in run.items() if k != "implementation"}
            for run in experiment_data["results"]
        ],
    }


def _stz_payload_for_save(experiment_data: dict, stz_path: str) -> dict:
    params = experiment_data["params"]
    return {
        "params": {
            "seed_sequence": params.get("seed_sequence"),
            "seeds": list(params.get("seeds", [])),
            "N_sim": params.get("N_sim"),
            "dataset": params.get("dataset"),
            "target_col": params.get("target_col"),
            "out_path": params.get("out_path"),
            "stz_path": stz_path,
        },
        "results": [
            {"seed": run["seed"], "implementation": run["implementation"]}
            for run in experiment_data["results"]
            if "implementation" in run
        ],
    }


def _save_experiment_checkpoints(out_path: str, stz_path: str, experiment_data: dict) -> None:
    _pkl_dump(out_path, _main_payload_for_save(experiment_data))
    _pkl_dump(stz_path, _stz_payload_for_save(experiment_data, stz_path))


def _compute_dast_stz_scores(sim_result: dict, M_candidates: list[int]) -> None:
    analysis_dir = _REPO_ROOT / "analysis"
    if str(analysis_dir) not in sys.path:
        sys.path.insert(0, str(analysis_dir))
    from stz import STZ_evaluator

    stz_by_M: dict[str, dict[str, float]] = {}
    for M in M_candidates:
        dast_key = _dast_impl_key(M)
        stz_by_M[str(M)] = {}
        for algo in ALGO_LIST:
            adv = STZ_evaluator(sim_result, dast_key, algo)
            if np.isfinite(adv):
                stz_by_M[str(M)][algo] = float(adv)
    sim_result["dast"]["stz"] = stz_by_M


def _pkl_dump(path: str, data) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wb", compresslevel=6) as f:
        pickle.dump(data, f, protocol=4)


def _set_thread_env(n: int) -> None:
    n = int(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    os.environ["HILLSTORM_SKLEARN_N_JOBS"] = str(n)


def _prepare_fresh_experiment(
    out_path: str,
    expected_params: dict,
    fig_dir: Path | None,
) -> dict:
    """Start a new run, overwriting any existing checkpoint and figures."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        out.unlink()
        print(f"[OVERWRITE] Removed existing checkpoint: {out}")

    stz_path = Path(resolve_stz_pkl_path(out_path))
    if stz_path.is_file():
        stz_path.unlink()
        print(f"[OVERWRITE] Removed existing STZ sidecar: {stz_path}")

    if fig_dir is not None:
        fig_path = Path(fig_dir)
        if fig_path.is_dir():
            removed = 0
            for f in fig_path.glob("*.png"):
                f.unlink()
                removed += 1
            if removed:
                print(f"[OVERWRITE] Cleared {removed} figure(s) in {fig_path}")

    return {"params": dict(expected_params), "results": []}


def _evaluate_baseline(
    algo: str,
    X_pilot,
    D_pilot,
    y_pilot,
    X_impl,
    D_impl,
    y_impl,
    mu_pilot_models,
    action_K: int,
    seed: int,
) -> dict:
    """Fit one meta-learner baseline and return OPE metrics + runtime."""
    mu_model_type = META_LEARNER_MU_MODEL_TYPE
    t0 = time.perf_counter()
    actions_all = np.arange(action_K, dtype=int)
    action_identity = actions_all.copy()

    if algo == "t_learner":
        models = fit_t_learner(
            X_pilot, D_pilot, y_pilot,
            K=action_K, model_type=mu_model_type, random_state=seed,
        )
        mu_mat = predict_mu_t_learner_matrix(models, X_impl)
        seg_labels_impl = np.argmax(mu_mat, axis=1).astype(int)

    elif algo == "s_learner":
        model = fit_s_learner(
            X_pilot, D_pilot, y_pilot,
            K=action_K, model_type=mu_model_type, random_state=seed,
        )
        mu_mat = predict_mu_s_learner_matrix(model, X_impl, K=action_K)
        seg_labels_impl = np.argmax(mu_mat, axis=1).astype(int)

    elif algo == "x_learner":
        x_models = fit_x_learner(
            X_pilot=X_pilot,
            D_pilot=D_pilot,
            y_pilot=y_pilot,
            mu_pilot_models=mu_pilot_models,
            control_action=0,
            mu_model_type=mu_model_type,
            random_state=seed,
        )
        seg_labels_impl, _ = predict_best_action_x_learner(
            x_learner_models=x_models,
            X=X_impl,
            mu_pilot_models=mu_pilot_models,
        )

    elif algo == "dr_learner":
        pi_vec = _get_propensity_per_action(D_pilot, actions_all, propensities=None)
        if action_K > 2:
            dr_model = fit_dr_learner_k_armed(
                X=X_pilot, D=D_pilot, y=y_pilot,
                K=action_K, pi=pi_vec, baseline=0, n_folds=5,
                mu_model_type=mu_model_type,
                tau_model_type=tau_model_type_from_mu(mu_model_type),
            )
            seg_labels_impl, _ = dr_learner_policy_k_armed(dr_model, X_impl)
        else:
            dr_model = fit_dr_learner_binary(
                X=X_pilot, D=D_pilot, y=y_pilot,
                e=float(pi_vec[1]), n_folds=3,
                mu_model_type=mu_model_type,
                tau_model_type=tau_model_type_from_mu(mu_model_type),
            )
            seg_labels_impl, _ = dr_learner_policy_binary(dr_model, X_impl)
        seg_labels_impl = seg_labels_impl.astype(int)

    elif algo == "causal_forest":
        cf_model = fit_multiarm_causal_forest(
            X_pilot, y_pilot, D_pilot,
            action_levels=actions_all, num_trees=50, seed=int(seed),
        )
        seg_labels_impl, _ = predict_best_action_multiarm(cf_model, X_impl)
        seg_labels_impl = seg_labels_impl.astype(int)

    else:
        raise ValueError(f"Unknown baseline algorithm: {algo}")

    out = {}
    for ev in eval_methods:
        value = eval_classes[ev](
            X_impl, D_impl, y_impl,
            seg_labels_impl,
            mu_pilot_models,
            action_identity,
            propensities=None,
        )
        out[ev] = float(value["value_mean"])
    out["time"] = float(time.perf_counter() - t0)
    return out, seg_labels_impl.astype(int)


def run_single_all_M_experiment(
    *,
    sample_frac: float,
    pilot_frac: float,
    train_frac: float,
    dataset: str,
    target_col: str,
    mu_model_type: str,
    value_type_dast: str,
    value_type_dams: str,
    action_method: str,
    M_candidates: list[int],
    seed: int,
) -> dict:
    dataset_loaders = {
        "hillstrom": load_hillstrom,
        "criteo": load_criteo,
        "lenta": load_lenta,
    }
    if dataset not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset}")

    X, y, D = dataset_loaders[dataset](
        sample_frac=sample_frac, seed=seed, target_col=target_col
    )

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
    ) = prepare_pilot_impl(
        X, y, D, pilot_frac=pilot_frac, mu_model_type=mu_model_type,
        return_impl_customer_id=True,
    )

    action_K = Gamma_pilot.shape[1]
    impl_actions: dict[str, np.ndarray] = {}

    sim_result: dict = {"seed": int(seed)}
    for algo in ALGO_LIST:
        sim_result[algo] = {}
    sim_result["dast"] = {}

    for algo in ALGO_LIST:
        metrics, seg_labels_impl = _evaluate_baseline(
            algo,
            X_pilot, D_pilot, y_pilot,
            X_impl, D_impl, y_impl,
            mu_pilot_models,
            action_K,
            seed,
        )
        sim_result[algo] = metrics
        _record_impl_action(impl_actions, algo, seg_labels_impl)

    print("\n" + "=" * 60)
    print("DAST: sweep M and evaluate on implementation set")
    print("=" * 60)
    t0 = time.perf_counter()
    best_M, pilot_by_M = run_dast_all_M_curves(
        X_pilot, D_pilot, y_pilot,
        X_impl,
        Gamma_pilot,
        M_candidates,
        min_leaf_size=MIN_LEAF_SIZE,
        value_type_dast=value_type_dast,
        value_type_dams=value_type_dams,
        action_method=action_method,
    )
    sim_result["dast"]["best_M"] = int(best_M)
    for ev in eval_methods:
        sim_result["dast"][ev] = {}
        for M, (seg_labels_impl, action_pilot) in pilot_by_M.items():
            value = eval_classes[ev](
                X_impl, D_impl, y_impl,
                seg_labels_impl,
                mu_pilot_models,
                action_pilot,
                propensities=None,
            )
            sim_result["dast"][ev][str(M)] = float(value["value_mean"])
            _record_impl_action(
                impl_actions, _dast_impl_key(M), seg_labels_impl, action_pilot
            )
    sim_result["dast"]["time"] = float(time.perf_counter() - t0)

    _attach_sim_implementation(
        sim_result,
        impl_actions,
        impl_customer_id=impl_customer_id,
        D_impl=D_impl,
        y_impl=y_impl,
        D_cohort=D,
        y_cohort=y,
    )
    _compute_dast_stz_scores(sim_result, M_candidates)

    print("\nResult for this run:")
    for k, v in sim_result.items():
        if k == "dast":
            summary = {"best_M": v.get("best_M"), "time": v.get("time")}
            for ev in eval_methods:
                if ev in v and isinstance(v[ev], dict):
                    summary[ev] = {m: round(val, 6) for m, val in v[ev].items()}
            print(f"{k:20s}: {summary}")
        elif isinstance(v, dict):
            print(f"{k:20s}: {v}")
        else:
            print(f"{k:20s}: {v}")

    return sim_result


def _run_worker(payload: dict) -> dict:
    inner_threads = int(payload.get("inner_threads", 1))
    _set_thread_env(inner_threads)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run_single_all_M_experiment(
            sample_frac=payload["sample_frac"],
            pilot_frac=payload["pilot_frac"],
            train_frac=payload["train_frac"],
            dataset=payload["dataset"],
            target_col=payload["target_col"],
            mu_model_type=payload["mu_model_type"],
            value_type_dast=payload["value_type_dast"],
            value_type_dams=payload["value_type_dams"],
            action_method=payload["action_method"],
            M_candidates=payload["M_candidates"],
            seed=int(payload["seed"]),
        )


def _plot_run_if_enabled(
    experiment_data: dict,
    *,
    run_index: int,
    fig_dir: Path | None,
    no_plot: bool,
) -> None:
    if no_plot or fig_dir is None:
        return
    from analysis.plot_all_M import plot_experiment

    plot_experiment(
        experiment_data,
        fig_dir=fig_dir,
        run_index=run_index,
        eval_method="dual_dr",
    )
    plot_experiment(
        experiment_data,
        fig_dir=fig_dir,
        run_index=run_index,
        eval_method="stz",
    )


def run_multiple_all_M_experiments(
    *,
    N_sim: int,
    sample_frac: float,
    pilot_frac: float,
    train_frac: float,
    out_path: str,
    dataset: str,
    target_col: str,
    mu_model_type: str,
    value_type_dast: str,
    value_type_dams: str,
    action_method: str,
    M_candidates: list[int],
    seed_sequence: int | None,
    n_jobs: int,
    fig_dir: Path | None = None,
    no_plot: bool = False,
) -> dict:
    inner_threads = 1
    n_jobs = int(n_jobs)
    max_attempts = int(N_sim) * 5
    out_path = str(Path(out_path).expanduser().resolve())

    expected_params = {
        "experiment_type": "all_M",
        "seed_sequence": seed_sequence,
        "sample_frac": sample_frac,
        "pilot_frac": pilot_frac,
        "train_frac": train_frac,
        "N_sim": int(N_sim),
        "max_attempts": max_attempts,
        "dataset": dataset,
        "target_col": target_col,
        "mu_model_type": mu_model_type,
        "meta_learner_mu_model_type": META_LEARNER_MU_MODEL_TYPE,
        "value_type_dast": value_type_dast,
        "value_type_dams": value_type_dams,
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

    experiment_data = _prepare_fresh_experiment(out_path, expected_params, fig_dir)
    experiment_data["params"]["out_path"] = out_path
    stz_path = resolve_stz_pkl_path(out_path)
    experiment_data["params"]["stz_path"] = stz_path

    print("Experiment parameters:")
    for k, v in experiment_data["params"].items():
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
            "action_method": action_method,
            "M_candidates": list(M_candidates),
            "seed": int(seed),
            "inner_threads": inner_threads,
        }

    if n_jobs <= 1:
        _set_thread_env(inner_threads)
        attempts_used = int(experiment_data["params"].get("attempts_used", 0))
        while len(experiment_data["results"]) < N_sim:
            if attempts_used >= max_attempts:
                experiment_data["params"]["attempts_used"] = attempts_used
                _save_experiment_checkpoints(out_path, stz_path, experiment_data)
                raise RuntimeError(
                    f"Exceeded max_attempts={max_attempts}; "
                    f"only {len(experiment_data['results'])}/{N_sim} successes."
                )
            attempts_used += 1
            experiment_data["params"]["attempts_used"] = attempts_used
            seed = random.randint(0, 1_000_000)
            try:
                res = run_single_all_M_experiment(
                    sample_frac=sample_frac,
                    pilot_frac=pilot_frac,
                    train_frac=train_frac,
                    dataset=dataset,
                    target_col=target_col,
                    mu_model_type=mu_model_type,
                    value_type_dast=value_type_dast,
                    value_type_dams=value_type_dams,
                    action_method=action_method,
                    M_candidates=M_candidates,
                    seed=int(seed),
                )
                experiment_data["results"].append(res)
                experiment_data["params"]["seeds"].append(int(seed))
                _save_experiment_checkpoints(out_path, stz_path, experiment_data)
                run_index = len(experiment_data["results"]) - 1
                print(f'[SIM {run_index + 1}/{N_sim}] saved -> {out_path}')
                _plot_run_if_enabled(
                    experiment_data,
                    run_index=run_index,
                    fig_dir=fig_dir,
                    no_plot=no_plot,
                )
                print("-" * 60)
            except Exception:
                import traceback
                traceback.print_exc()
                _save_experiment_checkpoints(out_path, stz_path, experiment_data)
                continue
    else:
        _set_thread_env(inner_threads)
        t_start = time.perf_counter()
        pending: dict = {}
        attempts_used = int(experiment_data["params"].get("attempts_used", 0))

        def submit_one(pool_ex):
            nonlocal attempts_used
            if attempts_used >= max_attempts:
                return False
            s = random.randint(0, 1_000_000)
            fut = pool_ex.submit(_run_worker, _payload(s))
            pending[fut] = s
            attempts_used += 1
            experiment_data["params"]["attempts_used"] = attempts_used
            return True

        with cf.ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for _ in range(min(n_jobs, N_sim - len(experiment_data["results"]))):
                if not submit_one(ex):
                    break

            while len(experiment_data["results"]) < N_sim:
                if not pending:
                    if attempts_used >= max_attempts:
                        break
                    if not submit_one(ex):
                        break
                done, _ = cf.wait(set(pending.keys()), return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    seed_used = pending.pop(fut)
                    if len(experiment_data["results"]) >= N_sim:
                        continue
                    try:
                        res = fut.result()
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        _save_experiment_checkpoints(out_path, stz_path, experiment_data)
                        if len(experiment_data["results"]) < N_sim and submit_one(ex):
                            pass
                        continue
                    experiment_data["results"].append(res)
                    experiment_data["params"]["seeds"].append(int(seed_used))
                    _save_experiment_checkpoints(out_path, stz_path, experiment_data)
                    completed = len(experiment_data["results"])
                    run_index = completed - 1
                    elapsed = time.perf_counter() - t_start
                    pct = 100.0 * completed / N_sim
                    print(
                        f"\r[SIM {completed}/{N_sim}] {pct:6.2f}% | "
                        f"elapsed {elapsed:8.1f}s | saved -> {out_path}",
                        end="",
                        flush=True,
                    )
                    _plot_run_if_enabled(
                        experiment_data,
                        run_index=run_index,
                        fig_dir=fig_dir,
                        no_plot=no_plot,
                    )
                    if completed < N_sim:
                        submit_one(ex)
        print("")

    print("\nALL SIMULATIONS DONE.")
    print(f"Results saved in '{out_path}'")
    return experiment_data


def _parse_M_candidates(raw: str | None) -> list[int]:
    if raw is None:
        return list(DEFAULT_M_CANDIDATES)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --M-candidates.")
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep DAST segment count M vs meta-learner baselines; "
            "auto-plots via analysis/plot_all_M.py unless --no-plot."
        )
    )
    parser.add_argument(
        "--exp-root",
        "--exp_root",
        default=os.environ.get("EXP_ROOT", DEFAULT_EXP_ROOT),
        help=f"Experiment root directory (default: {DEFAULT_EXP_ROOT}, or $EXP_ROOT)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hillstrom", "criteo", "lenta"],
        required=True,
    )
    parser.add_argument("--target", type=str, required=True, help="Target column")
    parser.add_argument("--sample_frac", type=float, required=True)
    parser.add_argument(
        "--pilot_frac",
        type=float,
        default=0.2,
        help="Fraction of cohort used as pilot (default: 0.2)",
    )
    parser.add_argument(
        "--mu_model_type",
        type=str,
        required=True,
        choices=[
            "linear",
            "mlp_reg",
            "lightgbm_reg",
            "logistic",
            "mlp_clf",
            "lightgbm_clf",
        ],
        help="Outcome model for pilot nuisances / DAST (meta-learners fixed to mlp_reg)",
    )
    parser.add_argument(
        "--value_type_dast",
        type=str,
        default="hybrid",
        help="Value type for DAST splitting ('dr' or 'hybrid')",
    )
    parser.add_argument(
        "--value_type_dams",
        type=str,
        default="hybrid",
        help="Value type for DAMS criterion ('dr' or 'hybrid')",
    )
    parser.add_argument(
        "--action_method",
        type=str,
        choices=["diff_in_means", "gamma", "logistic"],
        required=True,
        help="Method to estimate segment-level action",
    )
    parser.add_argument(
        "--M-candidates",
        "--M_candidates",
        dest="M_candidates",
        default=None,
        help=f"Comma-separated M values (default: {DEFAULT_M_CANDIDATES})",
    )
    parser.add_argument(
        "--N-sim",
        "--N_sim",
        type=int,
        default=1,
        dest="N_sim",
        help="Number of simulation runs (default: 1)",
    )
    parser.add_argument(
        "--seed_sequence",
        type=int,
        default=None,
        help="Seed for random.seed() before drawing run seeds",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel simulation workers (default: 1)",
    )
    parser.add_argument(
        "--fig-dir",
        "--fig_dir",
        default=None,
        help="Figure output directory (default: <pkl_parent>/figures)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting after the experiment finishes",
    )
    args = parser.parse_args()

    if args.seed_sequence is not None:
        random.seed(args.seed_sequence)
        print(f"Using fixed sequence seed: {args.seed_sequence}")

    M_candidates = _parse_M_candidates(args.M_candidates)

    out_path = resolve_out_path(
        exp_root=args.exp_root,
        dataset=args.dataset,
        target_col=args.target,
        mu_model_type=args.mu_model_type,
    )
    print(f"Output path: {out_path}")
    fig_dir = Path(args.fig_dir).expanduser().resolve() if args.fig_dir else (out_path.parent / "figures")

    experiment_data = run_multiple_all_M_experiments(
        N_sim=args.N_sim,
        sample_frac=args.sample_frac,
        pilot_frac=args.pilot_frac,
        train_frac=TRAIN_FRAC,
        out_path=str(out_path),
        dataset=args.dataset,
        target_col=args.target,
        mu_model_type=args.mu_model_type,
        value_type_dast=args.value_type_dast,
        value_type_dams=args.value_type_dams,
        action_method=args.action_method,
        M_candidates=M_candidates,
        seed_sequence=args.seed_sequence,
        n_jobs=args.n_jobs,
        fig_dir=fig_dir,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
