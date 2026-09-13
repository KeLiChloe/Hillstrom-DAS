"""
Plot DAST policy value vs. segment count M (all candidate M on one curve).

Expects pickle output from run_all_M.py:
  results[i]["dast"]["dual_dr"]["2"], ..., results[i]["dast"]["best_M"]
  results[i]["dast"]["stz"]["2"]["t_learner"], ...      (STZ_basic advantage)
  results[i]["dast"]["stz_vr"]["2"]["t_learner"], ...   (STZ_VR advantage)
  baselines: results[i][algo]["dual_dr"] as scalars.

STZ implementation actions may be in a *_stz.pkl sidecar (merged at load time).
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_ANALYSIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ANALYSIS_DIR.parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plot_style import baseline_color, baseline_label, to_rgba  # noqa: E402

DEFAULT_EVAL_METHODS = ("dr", "dual_dr", "ipw", "dm", "stz", "stz_vr")
FIG_DIR = "figures"

SKIP_BASELINE_KEYS = {"seed", "dast", "random", "implementation"}
DAST_COLOR = "#000000"


def _is_run_all_m_dast(dast: dict) -> bool:
    """True if dast block looks like run_all_M per-M curves."""
    if not isinstance(dast, dict):
        return False
    for ev in DEFAULT_EVAL_METHODS:
        sub = dast.get(ev)
        if isinstance(sub, dict) and any(str(k).isdigit() for k in sub):
            return True
    return False


def _pkl_load(path: Path):
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as f:
            return pickle.load(f)


def merge_stz_sidecar(experiment_data: dict, main_pkl_path: Path) -> int:
    """Attach implementation blocks from *_stz.pkl sidecar (by seed). Returns n merged."""
    stz_path = main_pkl_path.with_name(main_pkl_path.stem + "_stz" + main_pkl_path.suffix)
    if not stz_path.is_file():
        return 0
    stz_data = _pkl_load(stz_path)
    impl_by_seed = {
        run["seed"]: run["implementation"]
        for run in stz_data.get("results", [])
        if isinstance(run, dict) and "implementation" in run
    }
    merged = 0
    for run in experiment_data.get("results", []):
        seed = run.get("seed")
        if seed in impl_by_seed and "implementation" not in run:
            run["implementation"] = impl_by_seed[seed]
            merged += 1
    return merged


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


def extract_dast_stz_curves(
    dast: dict, baselines: list[str], *, stz_key: str = "stz"
) -> dict[str, tuple[list[int], list[float]]]:
    sub = dast.get(stz_key)
    if not isinstance(sub, dict):
        raise KeyError(
            f"dast['{stz_key}'] missing; re-run with updated run_all_M.py."
        )
    curves: dict[str, tuple[list[int], list[float]]] = {}
    for b in baselines:
        pairs: list[tuple[int, float]] = []
        for k, v in sub.items():
            try:
                m = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, dict) and b in v and isinstance(v[b], (int, float)):
                pairs.append((m, float(v[b])))
        if pairs:
            pairs.sort(key=lambda x: x[0])
            curves[b] = ([p[0] for p in pairs], [p[1] for p in pairs])
    if not curves:
        raise ValueError(f"No STZ curves found under dast['{stz_key}'].")
    return curves


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
    t = (target_col or "outcome").replace("_", " ").lower()
    if t == "conversion":
        metric = "Conversion Rate"
    else:
        metric = t.title()
    title = f"{metric} on Implementation Set as a Function of Segment Count (M)"
    ylabel = f"{metric} (%)"
    return title, ylabel


def _as_percent(values) -> list[float]:
    return [100.0 * float(v) for v in values]


def _configure_plot_style() -> None:
    """Times New Roman / serif typography for paper figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "axes.labelweight": "bold",
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.35,
        }
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
    y_values = _as_percent(y_values)
    x_min, x_max = min(M_values), max(M_values)

    _configure_plot_style()

    linestyles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]
    star_red = "#c1121f"

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for i, algo in enumerate(baseline_algorithms(run)):
        metrics = run[algo]
        val = metrics.get(eval_method)
        if val is None or isinstance(val, dict):
            continue
        color = to_rgba(baseline_color(algo))
        ax.hlines(
            y=100.0 * float(val),
            xmin=x_min,
            xmax=x_max,
            colors=[color],
            linestyles=linestyles[i % len(linestyles)],
            linewidth=1.8,
            label=baseline_label(algo),
        )

    ax.plot(
        M_values,
        y_values,
        marker="o",
        markersize=5.5,
        color=DAST_COLOR,
        linewidth=1.8,
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
    ax.set_title(title, pad=14)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Number of Segments (M)")

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
    for ext in ("png", "pdf"):
        out_path = fig_dir / f"{out_stem}.{ext}"
        fig.savefig(
            out_path, bbox_extra_artists=(lgd,), bbox_inches="tight", facecolor="white"
        )
        print(f"[OK] Saved: {out_path}")
        saved.append(out_path)
    plt.close(fig)
    return saved


def plot_all_M_stz(
    params: dict,
    run: dict,
    *,
    fig_dir: Path,
    out_stem: str,
    stz_key: str = "stz",
) -> list[Path]:
    dast = run["dast"]
    baselines = baseline_algorithms(run)
    curves = extract_dast_stz_curves(dast, baselines, stz_key=stz_key)
    all_M = sorted({m for xs, _ in curves.values() for m in xs})
    x_min, x_max = min(all_M), max(all_M)
    best_M = dast.get("best_M")
    best_M = int(best_M) if best_M is not None else None

    _configure_plot_style()

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--", alpha=0.8, zorder=1)

    for algo, (M_values, y_values) in curves.items():
        color = to_rgba(baseline_color(algo))
        ax.plot(
            M_values,
            y_values,
            marker="o",
            markersize=5,
            color=color,
            linewidth=2.0,
            linestyle="-",
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=f"DAST vs {baseline_label(algo)}",
            zorder=3,
        )
        if best_M is not None and best_M in M_values:
            idx = M_values.index(best_M)
            ax.plot(
                best_M,
                y_values[idx],
                marker="*",
                markersize=18,
                color=color,
                linestyle="None",
                zorder=20,
            )

    title, _ = target_labels(params.get("target_col"))
    stz_label = "STZ-VR" if stz_key == "stz_vr" else "STZ"
    ax.set_title(title, pad=14)
    ax.set_ylabel(f"DAS {stz_label} Advantage (%)")
    ax.set_xlabel("Number of Segments (M)")
    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_facecolor("white")
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.18)
    ax.margins(x=0.02)

    lgd = ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        title="Comparators",
        handlelength=3.2,
    )

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    saved: list[Path] = []
    for ext in ("png", "pdf"):
        out_path = fig_dir / f"{out_stem}.{ext}"
        fig.savefig(
            out_path, bbox_extra_artists=(lgd,), bbox_inches="tight", facecolor="white"
        )
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
    eval_method: str = "dual_dr",
) -> list[Path]:
    """
    Plot DAST-vs-M curves from an in-memory experiment dict (run_all_M format).

    Returns paths of all saved figure files.
    """
    params, run = load_run(experiment_data, run_index)
    fig_dir = Path(fig_dir).expanduser().resolve()
    slug = _slug_from_params(params)
    seed = run.get("seed", run_index)
    out_stem = f"all_M_{slug}_seed{seed}_{eval_method}"
    print("\n" + "=" * 60)
    print("PLOTTING (plot_all_M)")
    print(f"  run_index={run_index}, seed={seed}, best_M={run['dast'].get('best_M')}")
    print("=" * 60)
    if eval_method in ("stz", "stz_vr"):
        return plot_all_M_stz(
            params, run, fig_dir=fig_dir, out_stem=out_stem, stz_key=eval_method
        )
    return plot_all_M(
        params, run, eval_method=eval_method, fig_dir=fig_dir, out_stem=out_stem
    )


def plot_experiment_from_pkl(
    pkl_path: str | Path,
    *,
    fig_dir: str | Path = FIG_DIR,
    run_index: int = 0,
    eval_method: str = "all",
) -> list[Path]:
    """Load pickle from disk and call plot_experiment."""
    pkl_path = Path(pkl_path).expanduser().resolve()
    try:
        with gzip.open(pkl_path, "rb") as f:
            experiment_data = pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(pkl_path, "rb") as f:
            experiment_data = pickle.load(f)
    n = merge_stz_sidecar(experiment_data, pkl_path)
    if n:
        print(f"[INFO] Merged STZ implementation data for {n} run(s) from sidecar.")
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
        default="dual_dr",
        choices=["dual_dr", "stz", "stz_vr", "dr", "ipw", "dm"],
        help="Score to plot: dual_dr, STZ_basic, or STZ_VR (default: dual_dr).",
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
