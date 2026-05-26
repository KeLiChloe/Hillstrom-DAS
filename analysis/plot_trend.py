"""
Trend plot: DAS improvement vs. pilot fraction or sample fraction across experiment pickles.

Per-run improvement (%) = (DAST - comparator) / comparator * 100; then mean ± 95% CI.
Sweep axis is inferred from --exp-dir (pilot_frac_* vs sample_frac_* pickles).
"""

from __future__ import annotations

import argparse
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from scipy import stats

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DAST_ALGO = "dast"  # algorithm compared against baselines (CI_plot.py)
DEFAULT_EVAL_METHODS = ["dual_dr", "dr", "ipw"]
FIG_DIR = "figures"

REQUESTED_BASELINES = [
    "random",
    "kmeans",
    "gmm",
    "clr",
    "mst",
    "causal_forest",
    "t_learner",
    "s_learner",
    "x_learner",
    "dr_learner",
]

LABEL_MAP = {
    "random": "Random",
    "kmeans": "K-Means",
    "gmm": "GMM",
    "clr": "CLR",
    "mst": "MST",
    "causal_forest": "Causal Forest",
    "t_learner": "T-learner",
    "s_learner": "S-learner",
    "x_learner": "X-learner",
    "dr_learner": "DR-learner",
}

PREFERRED_ORDER = [
    "Random",
    "K-Means",
    "GMM",
    "CLR",
    "MST",
    "Causal Forest",
    "T-learner",
    "S-learner",
    "X-learner",
    "DR-learner",
]

# Tol bright–inspired (original palette)
COMPARATOR_COLORS = {
    "Random": "#000000",
    "K-Means": "#E69F00",
    "GMM": "#56B4E9",
    "CLR": "#009E73",
    "MST": "#F0E442",
    "Causal Forest": "#0072B2",
    "T-learner": "#D55E00",
    "S-learner": "#CC79A7",
    "X-learner": "#882255",
    "DR-learner": "#999999",
}

LINE_ALPHA = 0.82  # line/marker transparency (CI_plot used ~0.6 via #RRGGBB99)

SweepKind = Literal["pilot_frac", "sample_frac"]


@dataclass(frozen=True)
class SweepConfig:
    kind: SweepKind
    fixed_pilot_frac: float | None = None

    @property
    def x_param(self) -> str:
        return self.kind

    @property
    def x_label(self) -> str:
        if self.kind == "pilot_frac":
            return "Pilot fraction (%)"
        return "Sample fraction (%)"

    @property
    def out_stem_prefix(self) -> str:
        return f"trend_{self.kind}"

    def format_fixed_pilot_frac(self) -> str:
        if self.fixed_pilot_frac is None:
            return ""
        pct = self.fixed_pilot_frac * 100.0
        if abs(pct - round(pct)) < 1e-9:
            return f"{int(round(pct))}%"
        return f"{pct:g}%"

    def subtitle(self, n_sims: int) -> str:
        base = f"Each data point: mean ± 95% CI ({n_sims} runs)"
        if self.fixed_pilot_frac is not None:
            return f"{base}; fixed pilot fraction = {self.format_fixed_pilot_frac()}"
        return base


def _with_alpha(hex_color: str, alpha: float = LINE_ALPHA) -> tuple:
    return mcolors.to_rgba(hex_color, alpha=alpha)

def configure_plot_style() -> None:
    sns.set_context("paper", font_scale=1.35)
    sns.set_style("ticks", {"axes.grid": True})
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "STIXGeneral",
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "axes.labelweight": "bold",
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
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


def safe_get_value(run: dict, algo: str, ev: str) -> float:
    """Same accessor as CI_plot.py."""
    x = run[algo]
    v = x.get(ev, np.nan)
    return float(v)


def das_improvement_ratio(vt: float, vb: float) -> float:
    """
    Improvement ratio in percent.

    (DAST - comparator) / comparator * 100; 0 means no improvement.
    """
    if not np.isfinite(vt) or not np.isfinite(vb) or abs(vb) < 1e-12:
        return np.nan
    return (vt - vb) / vb * 100.0


def detect_sweep_kind(exp_dir: Path) -> SweepKind:
    """Infer x-axis sweep from directory name or pickle filenames."""
    has_sample = bool(list(exp_dir.glob("sample_frac_*.pkl")))
    has_pilot = bool(list(exp_dir.glob("pilot_frac_*.pkl")))

    if has_pilot and not has_sample:
        return "pilot_frac"
    if has_sample and not has_pilot:
        return "sample_frac"

    # e.g. pilot_frac_with_fixed_0.1_sample_frac vs sample_frac_with_fixed_020_pilot
    dir_name = exp_dir.name.lower()
    if dir_name.startswith("pilot_frac"):
        return "pilot_frac"
    if dir_name.startswith("sample_frac"):
        return "sample_frac"

    path_lower = exp_dir.as_posix().lower()
    if "pilot_frac" in path_lower:
        return "pilot_frac"
    if "sample_frac" in path_lower:
        return "sample_frac"
    return "pilot_frac"


def discover_sweep_value(kind: SweepKind, path: Path, params: dict) -> float:
    if kind in params:
        return float(params[kind])
    m = re.search(rf"{kind}_(\d+)", path.stem)
    if m:
        return int(m.group(1)) / 100.0
    raise ValueError(f"Cannot infer {kind} from {path}")


def load_experiment_pkls(
    exp_dir: Path, min_sims: int, sweep_kind: SweepKind
) -> tuple[list[dict], list[dict]]:
    pkls = sorted(exp_dir.glob(f"{sweep_kind}_*.pkl"))
    if not pkls:
        pkls = sorted(exp_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No pickle files found in {exp_dir}")

    loaded: list[dict] = []
    incomplete: list[dict] = []
    for path in pkls:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or "results" not in data:
            warnings.warn(f"Skipping {path.name}: unexpected format.")
            continue
        params = data.get("params", {})
        n_runs = len(data["results"])
        n_sim_expected = int(params.get("N_sim", 100))
        if n_runs < n_sim_expected:
            incomplete.append(
                {
                    "path": path,
                    "n_runs": n_runs,
                    "n_sim": n_sim_expected,
                }
            )
        if n_runs < min_sims:
            warnings.warn(
                f"Skipping {path.name}: only {n_runs} runs "
                f"(need >= {min_sims})."
            )
            continue
        loaded.append(
            {
                "path": path,
                "x_value": discover_sweep_value(sweep_kind, path, params),
                "params": params,
                "results": data["results"],
                "n_sims": n_runs,
            }
        )
    if not loaded:
        raise RuntimeError(
            f"No valid experiments in {exp_dir} (min_sims={min_sims})."
        )
    loaded.sort(key=lambda x: x["x_value"])
    return loaded, incomplete


def print_incomplete_pkl_warning(
    incomplete: list[dict], loaded: list[dict], min_sims: int
) -> None:
    """Remind user about partial pkls (e.g. from interrupted run_sims)."""
    if not incomplete:
        return
    loaded_paths = {exp["path"] for exp in loaded}
    print("\n" + "=" * 60)
    print("[WARN] Incomplete pickle(s) detected (possible interrupted runs):")
    for item in incomplete:
        path = item["path"]
        n_runs = item["n_runs"]
        n_sim = item["n_sim"]
        if path in loaded_paths:
            status = "used in plot"
        else:
            status = f"skipped (min_sims={min_sims})"
        print(f"  - {path.name}: {n_runs}/{n_sim}  ({status})")
    print("=" * 60)


def discover_baselines(results_list: list[dict], das_algo: str = DAST_ALGO) -> list[str]:
    all_keys: set[str] = set()
    for run in results_list:
        if isinstance(run, dict):
            all_keys |= set(run.keys())
    if das_algo not in all_keys:
        raise KeyError(
            f"DAST algorithm '{das_algo}' not found. Example keys: "
            f"{sorted(all_keys)[:20]}"
        )
    return [b for b in REQUESTED_BASELINES if b in all_keys and b != das_algo]


def compute_lift_records(
    results_list: list[dict],
    baselines: list[str],
    eval_method: str,
    das_algo: str = DAST_ALGO,
) -> pd.DataFrame:
    records = []
    for i, run in enumerate(results_list):
        vt = safe_get_value(run, das_algo, eval_method)
        if not np.isfinite(vt):
            continue
        for b in baselines:
            vb = safe_get_value(run, b, eval_method)
            if not np.isfinite(vb):
                continue
            lift = das_improvement_ratio(vt, vb)
            if not np.isfinite(lift):
                continue
            records.append(
                {
                    "Run": i,
                    "Baseline": b,
                    "Baseline_Label": LABEL_MAP.get(b, b.replace("_", " ").title()),
                    "Lift": float(lift),
                }
            )
    return pd.DataFrame(records)


def summarize_by_sweep(
    experiments: list[dict],
    baselines: list[str],
    eval_method: str,
    das_algo: str = DAST_ALGO,
) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        df = compute_lift_records(
            exp["results"], baselines, eval_method, das_algo=das_algo
        )
        if df.empty:
            continue
        for b in baselines:
            label = LABEL_MAP.get(b, b.replace("_", " ").title())
            sub = df[df["Baseline"] == b]["Lift"]
            if sub.empty:
                continue
            n = len(sub)
            mean = float(sub.mean())
            if n > 1:
                sem = float(sub.sem())
                ci = float(sem * stats.t.ppf(0.975, n - 1))
            else:
                ci = 0.0
            rows.append(
                {
                    "x_value": exp["x_value"],
                    "Baseline": b,
                    "Baseline_Label": label,
                    "Mean": mean,
                    "CI": ci,
                    "N": n,
                    "n_sims_file": exp["n_sims"],
                }
            )
    return pd.DataFrame(rows)


def ordered_labels(labels: list[str]) -> list[str]:
    ordered = [x for x in PREFERRED_ORDER if x in labels]
    ordered += [x for x in sorted(labels) if x not in set(ordered)]
    return ordered


def format_plot_title(params: dict) -> str:
    """Title uses dataset + outcome metric (params['target_col']), e.g. conversion."""
    dataset = str(params.get("dataset", "unknown")).replace("_", " ").title()
    target_metric = str(params.get("target_col", "unknown")).replace("_", " ").title()
    return f"{dataset} – {target_metric}"


def assert_consistent_experiment_meta(
    experiments: list[dict], sweep_kind: SweepKind
) -> tuple[dict, SweepConfig]:
    """Ensure pickles share dataset/target; build sweep config (fixed pilot for sample_frac)."""
    ref = experiments[0]["params"]
    dataset = ref.get("dataset")
    target_col = ref.get("target_col")
    fixed_pilot: float | None = None

    fixed_sample: float | None = None
    if sweep_kind == "sample_frac":
        if "pilot_frac" not in ref:
            raise ValueError("sample_frac sweep requires pilot_frac in params.")
        fixed_pilot = float(ref["pilot_frac"])
    elif sweep_kind == "pilot_frac":
        if "sample_frac" not in ref:
            raise ValueError("pilot_frac sweep requires sample_frac in params.")
        fixed_sample = float(ref["sample_frac"])

    for exp in experiments[1:]:
        p = exp["params"]
        if p.get("dataset") != dataset or p.get("target_col") != target_col:
            raise ValueError(
                f"Inconsistent metadata in {exp['path'].name}: "
                f"expected dataset={dataset!r}, target_col={target_col!r}, "
                f"got dataset={p.get('dataset')!r}, target_col={p.get('target_col')!r}."
            )
        if sweep_kind == "sample_frac":
            if float(p.get("pilot_frac", -1)) != fixed_pilot:
                raise ValueError(
                    f"Inconsistent pilot_frac in {exp['path'].name}: "
                    f"expected {fixed_pilot!r}, got {p.get('pilot_frac')!r}."
                )
        elif sweep_kind == "pilot_frac" and fixed_sample is not None:
            if float(p.get("sample_frac", -1)) != fixed_sample:
                raise ValueError(
                    f"Inconsistent sample_frac in {exp['path'].name}: "
                    f"expected {fixed_sample!r}, got {p.get('sample_frac')!r}."
                )

    sweep = SweepConfig(kind=sweep_kind, fixed_pilot_frac=fixed_pilot)
    return ref, sweep


def plot_trend(
    stats_df: pd.DataFrame,
    *,
    dataset_target: str,
    sweep: SweepConfig,
    n_sims: int,
    eval_method: str,
    out_stem: str,
    fig_dir: Path,
) -> None:
    labels = ordered_labels(stats_df["Baseline_Label"].unique().tolist())
    x_vals = sorted(stats_df["x_value"].unique())
    x_pct = [100.0 * x for x in x_vals]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    for label in labels:
        sub = stats_df[stats_df["Baseline_Label"] == label].sort_values("x_value")
        if sub.empty:
            continue
        base = COMPARATOR_COLORS.get(label, "#333333")
        color = _with_alpha(base)
        x = sub["x_value"].values * 100.0
        y = sub["Mean"].values
        yerr = sub["CI"].values

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="-o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            capthick=0.9,
            markersize=3.5,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.6,
            linewidth=1.4,
            label=label,
            zorder=3,
        )

    ax.axhline(
        0.0,
        color="#C1121F",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        zorder=1,
    )

    ax.set_xlabel(sweep.x_label, fontweight="bold", labelpad=8)
    ax.set_ylabel("DAS improvement ratio", fontweight="bold", labelpad=8)
    ax.set_title(
        f"DAS improvement ratio on {dataset_target}",
        fontweight="bold",
        fontsize=13,
        pad=20,
    )
    ax.text(
        0.5,
        1.01,
        sweep.subtitle(n_sims),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
    )

    ax.set_xticks(x_pct)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax.set_axisbelow(True)
    ax.grid(True, axis="both", linestyle=":", linewidth=0.5, alpha=0.55)
    sns.despine(ax=ax, top=True, right=True)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=3,
        frameon=True,
        framealpha=0.92,
        edgecolor="#D0D0D0",
        fancybox=False,
        fontsize=7,
        title="Comparator",
        title_fontsize=8,
        borderpad=0.25,
        labelspacing=0.25,
        columnspacing=0.55,
        handletextpad=0.25,
        handlelength=0.9,
        markerscale=0.8,
    )

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_path = fig_dir / f"{out_stem}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[OK] Saved: {out_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot DAS improvement ratio vs. pilot fraction or sample fraction "
            "(inferred from exp-dir / pickle names)."
        )
    )
    parser.add_argument(
        "--exp-dir",
        "--exp_dir",
        required=True,
        help=(
            "Directory with pilot_frac_*.pkl or sample_frac_*.pkl experiment files."
        ),
    )
    parser.add_argument(
        "--fig-dir",
        "--fig_dir",
        default=FIG_DIR,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--eval-method",
        "--eval_method",
        default="dual_dr",
        choices=["all"] + DEFAULT_EVAL_METHODS,
        help=(
            "OPE eval method: dual_dr (default), dr, ipw, or all to save three figures."
        ),
    )
    parser.add_argument(
        "--min-sims",
        "--min_sims",
        type=int,
        default=1,
        help="Minimum completed simulations required per pickle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings.simplefilter(action="ignore", category=FutureWarning)
    configure_plot_style()

    exp_dir = Path(args.exp_dir).expanduser().resolve()
    fig_dir = Path(args.fig_dir).expanduser().resolve()
    sweep_kind = detect_sweep_kind(exp_dir)
    experiments, incomplete_pkls = load_experiment_pkls(
        exp_dir, min_sims=args.min_sims, sweep_kind=sweep_kind
    )
    params0, sweep = assert_consistent_experiment_meta(experiments, sweep_kind)

    baselines = discover_baselines(experiments[0]["results"])
    if not baselines:
        raise RuntimeError("No comparators found in experiment results.")

    plot_title = format_plot_title(params0)
    n_sims = int(params0.get("N_sim", 100))
    eval_methods = (
        DEFAULT_EVAL_METHODS
        if args.eval_method == "all"
        else [args.eval_method]
    )

    print(f"Loaded {len(experiments)} experiments from {exp_dir}")
    print(f"Sweep axis: {sweep.kind}")
    print(f"{sweep.x_label}: {[e['x_value'] for e in experiments]}")
    if sweep.fixed_pilot_frac is not None:
        print(f"Fixed pilot fraction: {sweep.format_fixed_pilot_frac()}")
    print(f"Comparators: {baselines}")
    print(f"Dataset: {params0.get('dataset')}")
    print(f"Target metric (target_col): {params0.get('target_col')}")
    print(f"Plot title: {plot_title}")
    print(f"Eval method(s): {eval_methods}")

    for ev in eval_methods:
        stats_df = summarize_by_sweep(experiments, baselines, ev)
        if stats_df.empty:
            print(f"[WARN] No data for eval_method={ev}; skip.")
            continue

        slug = plot_title.lower().replace(" – ", "_").replace(" ", "_")
        out_stem = f"{sweep.out_stem_prefix}_{slug}_{ev}"
        plot_trend(
            stats_df,
            dataset_target=plot_title,
            sweep=sweep,
            n_sims=n_sims,
            eval_method=ev,
            out_stem=out_stem,
            fig_dir=fig_dir,
        )

    print_incomplete_pkl_warning(incomplete_pkls, experiments, args.min_sims)


if __name__ == "__main__":
    main()
