"""
Plot DAST policy value vs. segment count M (all candidate M on one curve).

Expects pickle output from run_all_M.py:
  results[i]["dast"]["dual_dr"]["2"], ..., results[i]["dast"]["best_M"]
  baselines: results[i][algo]["dual_dr"] as scalars.

Not compatible with run_sims.py pilot_frac pkls (DAST stores a single scalar per eval).
"""

from __future__ import annotations

import argparse
import pickle
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DEFAULT_EVAL_METHODS = ("dr", "dual_dr", "ipw")
FIG_DIR = "figures"

SKIP_BASELINE_KEYS = {"seed", "dast", "random"}


def _is_run_all_m_dast(dast: dict) -> bool:
    """True if dast block looks like run_all_M per-M curves."""
    if not isinstance(dast, dict):
        return False
    for ev in DEFAULT_EVAL_METHODS:
        sub = dast.get(ev)
        if isinstance(sub, dict) and any(str(k).isdigit() for k in sub):
            return True
    return False


def load_run(data_container: dict, run_index: int) -> tuple[dict, dict]:
    if "results" not in data_container:
        raise ValueError(
            "Pickle has no 'results' key. Use output from run_all_M.py "
            "(run_multiple_simulations)."
        )
    results = data_container["results"]
    if not results:
        raise ValueError("results list is empty.")
    if run_index < 0 or run_index >= len(results):
        raise IndexError(
            f"run_index={run_index} out of range (n_results={len(results)})."
        )
    params = data_container.get("params", {})
    run = results[run_index]
    if not isinstance(run, dict):
        raise TypeError(f"results[{run_index}] is not a dict.")
    if "dast" not in run:
        raise KeyError(f"results[{run_index}] has no 'dast' entry.")
    if not _is_run_all_m_dast(run["dast"]):
        raise ValueError(
            "DAST block is not in run_all_M format (per-M nested dicts). "
            "This pickle may be from run_sims.py; use plot_trend.py instead."
        )
    return params, run


def extract_dast_curve(dast: dict, eval_method: str) -> tuple[list[int], list[float], int | None]:
    sub = dast.get(eval_method)
    if not isinstance(sub, dict):
        raise KeyError(
            f"dast['{eval_method}'] must be a dict of M -> value; got {type(sub).__name__}."
        )
    pairs: list[tuple[int, float]] = []
    for k, v in sub.items():
        try:
            m = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, (int, float)):
            pairs.append((m, float(v)))
    if not pairs:
        raise ValueError(f"No numeric M keys under dast['{eval_method}'].")
    pairs.sort(key=lambda x: x[0])
    M_values = [p[0] for p in pairs]
    y_values = [p[1] for p in pairs]
    best_M = dast.get("best_M")
    best_M = int(best_M) if best_M is not None else None
    return M_values, y_values, best_M


def baseline_algorithms(run: dict) -> list[str]:
    algos = []
    for k, v in run.items():
        if k in SKIP_BASELINE_KEYS or k.startswith("all_"):
            continue
        if k == "dast":
            continue
        if isinstance(v, dict):
            algos.append(k)
    return sorted(algos)


def target_labels(target_col: str | None) -> tuple[str, str]:
    t = (target_col or "outcome").replace("_", " ").title()
    return (
        f"Expected {t} on Implementation Set as a Function of Segment Count (M)",
        f"Expected {t}",
    )


def plot_all_M(
    params: dict,
    run: dict,
    *,
    eval_method: str,
    fig_dir: Path,
    out_stem: str,
) -> list[Path]:
    dast = run["dast"]
    M_values, y_values, best_M = extract_dast_curve(dast, eval_method)
    x_min, x_max = min(M_values), max(M_values)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    linestyles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]
    gray_levels = ["#4d4d4d", "#7a7a7a", "#a0a0a0", "#b3b3b3", "#999999", "#666666"]
    dast_color = "#4379F9"
    star_red = "#c1121f"

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, algo in enumerate(baseline_algorithms(run)):
        metrics = run[algo]
        val = metrics.get(eval_method)
        if val is None or isinstance(val, dict):
            continue
        label_name = algo.replace("_", "-").title()
        ax.hlines(
            y=float(val),
            xmin=x_min,
            xmax=x_max,
            colors=gray_levels[i % len(gray_levels)],
            linestyles=linestyles[i % len(linestyles)],
            linewidth=1.8,
            alpha=0.95,
            label=label_name,
        )

    ax.plot(
        M_values,
        y_values,
        marker="o",
        markersize=6,
        color=dast_color,
        linewidth=2.5,
        linestyle="-",
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="DAST",
        zorder=3,
    )

    if best_M is not None and best_M in M_values:
        idx = M_values.index(best_M)
        best_val = y_values[idx]
        ax.plot(
            best_M,
            best_val,
            marker="*",
            markersize=22,
            color="black",
            alpha=0.08,
            linestyle="None",
            zorder=19,
        )
        ax.plot(
            best_M,
            best_val,
            marker="*",
            markersize=22,
            color=star_red,
            linestyle="None",
            zorder=20,
            markeredgecolor="white",
            markeredgewidth=2.2,
            label=f"M chosen by DAMS = {best_M}",
        )

    title, ylabel = target_labels(params.get("target_col"))
    ope = eval_method.upper().replace("_", "-")
    ax.set_title(f"{title}\n(OPE: {ope})", pad=14)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Number of segments (M)")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.18)
    ax.margins(x=0.02)

    lgd = ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        title="Algorithms",
        handlelength=3.2,
    )

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    saved: list[Path] = []
    for ext in ("pdf", "png"):
        out_path = fig_dir / f"{out_stem}.{ext}"
        fig.savefig(out_path, bbox_extra_artists=(lgd,), bbox_inches="tight", facecolor="white")
        print(f"[OK] Saved: {out_path}")
        saved.append(out_path)
    plt.close(fig)
    return saved


def _slug_from_params(params: dict) -> str:
    dataset = str(params.get("dataset", "data")).lower()
    target = str(params.get("target_col", "target")).lower()
    return re.sub(r"[^\w]+", "_", f"{dataset}_{target}").strip("_")


def plot_experiment(
    experiment_data: dict,
    *,
    fig_dir: str | Path = FIG_DIR,
    run_index: int = 0,
    eval_method: str = "all",
) -> list[Path]:
    """
    Plot DAST-vs-M curves from an in-memory experiment dict (run_all_M format).

    Returns paths of all saved figure files.
    """
    params, run = load_run(experiment_data, run_index)
    fig_dir = Path(fig_dir).expanduser().resolve()
    eval_methods = (
        list(DEFAULT_EVAL_METHODS)
        if eval_method == "all"
        else [eval_method]
    )
    slug = _slug_from_params(params)
    saved_all: list[Path] = []
    print("\n" + "=" * 60)
    print("PLOTTING (plot_all_M)")
    print(f"  run_index={run_index}, seed={run.get('seed')}, best_M={run['dast'].get('best_M')}")
    print("=" * 60)
    for ev in eval_methods:
        out_stem = f"all_M_{slug}_{ev}"
        saved_all.extend(
            plot_all_M(params, run, eval_method=ev, fig_dir=fig_dir, out_stem=out_stem)
        )
    return saved_all


def plot_experiment_from_pkl(
    pkl_path: str | Path,
    *,
    fig_dir: str | Path = FIG_DIR,
    run_index: int = 0,
    eval_method: str = "all",
) -> list[Path]:
    """Load pickle from disk and call plot_experiment."""
    pkl_path = Path(pkl_path).expanduser().resolve()
    with open(pkl_path, "rb") as f:
        experiment_data = pickle.load(f)
    return plot_experiment(
        experiment_data,
        fig_dir=fig_dir,
        run_index=run_index,
        eval_method=eval_method,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DAST value vs. M from run_all_M.py pickle output."
    )
    parser.add_argument(
        "--pkl-path",
        "--pkl_path",
        required=True,
        help="Path to pickle saved by run_all_M.py (must contain per-M dast curves).",
    )
    parser.add_argument(
        "--fig-dir",
        "--fig_dir",
        default=FIG_DIR,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--run-index",
        "--run_index",
        type=int,
        default=0,
        help="Which run in results[] to plot (default: 0).",
    )
    parser.add_argument(
        "--eval-method",
        "--eval_method",
        default="all",
        choices=["all", *DEFAULT_EVAL_METHODS],
        help="OPE method to plot (default: all three).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.simplefilter(action="ignore", category=FutureWarning)
    plot_experiment_from_pkl(
        args.pkl_path,
        fig_dir=args.fig_dir,
        run_index=args.run_index,
        eval_method=args.eval_method,
    )


if __name__ == "__main__":
    main()
