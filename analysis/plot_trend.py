"""
Trend plot across experiment pickles.

Two metrics (--metric):
  improvement  DAS_improvement_ratio = (DAST - comparator) / comparator * 100
               Plots dual_dr OPE improvement (dr/ipw omitted).
               Optional outlier removal per (sweep point, comparator) via
               --outlier-filter; method via --outlier-method (nstd or iqr).

  advantage    DAS_advantage_ratio = STZ_evaluator (Simester, Timoshenko, and Zoumpoulis)
               Uses per-customer logged data in run["implementation"].
               Loads implementation from *_stz.pkl sidecars.
               Same optional outlier removal as improvement.

  both         (default) Saves figures for both metrics.

Sweep axis is inferred from --exp-dir (pilot_frac_* vs sample_frac_* pickles).
Optionally restrict the x-axis with --min-pilot-frac / --min-sweep-value.
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_ANALYSIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ANALYSIS_DIR.parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from scipy import stats

from stz import STZ_evaluator
from plot_style import (
        PREFERRED_ORDER,
        baseline_color,
        baseline_label,
        comparator_colors_by_label,
        to_rgba,
    )


def _pkl_load(path):
    """Load a pickle file regardless of whether it is gzip-compressed or not."""
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DAST_ALGO = "dast"  # algorithm compared against baselines (CI_plot.py)
DEFAULT_EVAL_METHODS = ["dual_dr"]

REQUESTED_BASELINES = [
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

COMPARATOR_COLORS = comparator_colors_by_label()

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

    def to_display_pct(self, x_value: float) -> float:
        """
        Convert stored parameter value to display percentage.

        pilot_frac  : stored value IS the true fraction  → × 100
        sample_frac : stored value is 10× the true fraction (e.g. 0.05 → 0.5%)
                      → ÷ 10 × 100  =  × 10
        """
        if self.kind == "pilot_frac":
            return x_value * 100.0
        return x_value * 10.0

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


_DEFAULT_OUTLIER_N_STD = 3.0
_DEFAULT_OUTLIER_IQR_K = 1.5
_DEFAULT_OUTLIER_METHOD = "nstd"
_MIN_LIFTS_FOR_OUTLIER_FILTER = 4


def _keeper_mask_n_std(
    values: np.ndarray,
    *,
    n_std: float,
) -> np.ndarray:
    """
    True = keep. Drop points with |x - mean| > n_std * sample_std.
    If sample_std = 0, use modified z-score with MAD and the same n_std.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < _MIN_LIFTS_FOR_OUTLIER_FILTER:
        return np.ones(n, dtype=bool)

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    if std > 0:
        z = np.abs((values - mean) / std)
        return z <= n_std

    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    if mad <= 0:
        return np.ones(n, dtype=bool)
    modified_z = 0.6745 * (values - med) / mad
    return np.abs(modified_z) <= n_std


def _keeper_mask_iqr(
    values: np.ndarray,
    *,
    iqr_k: float,
) -> np.ndarray:
    """
    True = keep. Tukey fences: Q1 - k*IQR <= x <= Q3 + k*IQR.
    If IQR = 0, keep all (no spread to define outliers).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < _MIN_LIFTS_FOR_OUTLIER_FILTER:
        return np.ones(n, dtype=bool)

    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    if iqr <= 0:
        return np.ones(n, dtype=bool)

    low = q1 - iqr_k * iqr
    high = q3 + iqr_k * iqr
    return (values >= low) & (values <= high)


def apply_outlier_filter(
    df: pd.DataFrame,
    value_col: str,
    *,
    method: str = _DEFAULT_OUTLIER_METHOD,
    group_cols: tuple[str, ...] = ("x_value", "Baseline"),
    n_std: float = _DEFAULT_OUTLIER_N_STD,
    iqr_k: float = _DEFAULT_OUTLIER_IQR_K,
) -> tuple[pd.DataFrame, int]:
    """Drop outliers on value_col within each group (nσ or IQR)."""
    if df.empty:
        return df, 0
    n_before = len(df)
    kept = []
    for _, grp in df.groupby(list(group_cols), sort=False):
        vals = grp[value_col].to_numpy()
        if method == "nstd":
            mask = _keeper_mask_n_std(vals, n_std=n_std)
        elif method == "iqr":
            mask = _keeper_mask_iqr(vals, iqr_k=iqr_k)
        else:
            raise ValueError(f"Unknown outlier method: {method!r}")
        kept.append(grp.iloc[mask])
    filtered = pd.concat(kept, ignore_index=True) if kept else df.iloc[0:0]
    return filtered, n_before - len(filtered)


def filter_experiments_by_min_x(
    experiments: list[dict],
    min_x: float,
    *,
    sweep_kind: SweepKind,
) -> list[dict]:
    """Keep only experiments with x_value >= min_x (improvement + STZ)."""
    kept = [e for e in experiments if e["x_value"] >= min_x]
    if not kept:
        avail = sorted({e["x_value"] for e in experiments})
        raise RuntimeError(
            f"No experiments with {sweep_kind} >= {min_x:g} "
            f"(had {len(experiments)} pickle(s); available: {avail}). "
            f"Lower the cutoff with --min-pilot-frac {min(avail):g}."
        )
    if len(kept) < len(experiments):
        print(
            f"[INFO] min {sweep_kind} {min_x:g}: "
            f"kept {len(kept)}/{len(experiments)} experiment(s) "
            f"(improvement + STZ)"
        )
    return kept


def resolve_min_sweep_value(args: argparse.Namespace) -> float | None:
    if args.min_sweep_value is not None:
        return float(args.min_sweep_value)
    return None


def _is_main_pkl(path: Path) -> bool:
    return not path.stem.endswith("_stz")


def _resolve_stz_path(main_path: Path) -> Path:
    return main_path.with_name(main_path.stem + "_stz" + main_path.suffix)


def merge_stz_sidecars(experiments: list[dict]) -> int:
    """Attach implementation blocks from *_stz.pkl sidecars (by seed)."""
    merged = 0
    for exp in experiments:
        stz_path = _resolve_stz_path(Path(exp["path"]))
        if not stz_path.is_file():
            continue
        stz_data = _pkl_load(stz_path)
        impl_by_seed = {
            run["seed"]: run["implementation"]
            for run in stz_data.get("results", [])
            if isinstance(run, dict) and "implementation" in run
        }
        for run in exp["results"]:
            seed = run.get("seed")
            if seed in impl_by_seed:
                run["implementation"] = impl_by_seed[seed]
                merged += 1
    return merged


def detect_sweep_kind(exp_dir: Path) -> SweepKind:
    """Infer x-axis sweep from directory name or pickle filenames."""
    has_sample = bool(
        list(p for p in exp_dir.glob("sample_frac_*.pkl") if _is_main_pkl(p))
    )
    has_pilot = bool(
        list(p for p in exp_dir.glob("pilot_frac_*.pkl") if _is_main_pkl(p))
    )

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
    pkls = sorted(
        p for p in exp_dir.glob(f"{sweep_kind}_*.pkl") if _is_main_pkl(p)
    )
    if not pkls:
        pkls = sorted(p for p in exp_dir.glob("*.pkl") if _is_main_pkl(p))
    if not pkls:
        raise FileNotFoundError(f"No pickle files found in {exp_dir}")

    loaded: list[dict] = []
    incomplete: list[dict] = []
    for path in pkls:
        data = _pkl_load(path)
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


def compute_stz_records(
    results_list: list[dict],
    baselines: list[str],
    das_algo: str = DAST_ALGO,
) -> pd.DataFrame:
    """Per-run STZ advantage for every (DAST, comparator) pair."""
    records = []
    for i, run in enumerate(results_list):
        for b in baselines:
            adv = STZ_evaluator(run, das_algo, b)
            if not np.isfinite(adv):
                continue
            records.append(
                {
                    "Run": i,
                    "Baseline": b,
                    "Baseline_Label": baseline_label(b),
                    "Advantage": float(adv),
                }
            )
    return pd.DataFrame(records)


def summarize_stz_by_sweep(
    experiments: list[dict],
    baselines: list[str],
    das_algo: str = DAST_ALGO,
    *,
    filter_outliers: bool = False,
    outlier_method: str = _DEFAULT_OUTLIER_METHOD,
    n_std: float = _DEFAULT_OUTLIER_N_STD,
    iqr_k: float = _DEFAULT_OUTLIER_IQR_K,
) -> tuple[pd.DataFrame, int, int]:
    """Mean ± 95% CI of STZ advantage across runs, one row per (x_value, comparator)."""
    parts: list[pd.DataFrame] = []
    for exp in experiments:
        df = compute_stz_records(exp["results"], baselines, das_algo=das_algo)
        if df.empty:
            continue
        df = df.copy()
        df["x_value"] = exp["x_value"]
        df["n_sims_file"] = exp["n_sims"]
        parts.append(df)

    if not parts:
        return pd.DataFrame(), 0, 0

    combined = pd.concat(parts, ignore_index=True)
    n_filtered = 0
    if filter_outliers:
        combined, n_filtered = apply_outlier_filter(
            combined,
            "Advantage",
            method=outlier_method,
            n_std=n_std,
            iqr_k=iqr_k,
        )
    n_kept_total = len(combined)

    rows = []
    for exp in experiments:
        xv = exp["x_value"]
        slice_x = combined[combined["x_value"] == xv]
        if slice_x.empty:
            continue
        for b in baselines:
            label = baseline_label(b)
            sub = slice_x[slice_x["Baseline"] == b]["Advantage"]
            if sub.empty:
                continue
            n = len(sub)
            mean = float(sub.mean())
            ci = float(sub.sem() * stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
            rows.append(
                {
                    "x_value": xv,
                    "Baseline": b,
                    "Baseline_Label": label,
                    "Mean": mean,
                    "CI": ci,
                    "N": n,
                    "n_sims_file": exp["n_sims"],
                }
            )
    return pd.DataFrame(rows), n_kept_total, n_filtered


def compute_lift_records(
    results_list: list[dict],
    baselines: list[str],
    eval_method: str,
    das_algo: str = DAST_ALGO,
) -> pd.DataFrame:
    """Build per-run lift records for one sweep pickle."""
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
                    "Baseline_Label": baseline_label(b),
                    "Lift": float(lift),
                }
            )
    return pd.DataFrame(records)


def summarize_by_sweep(
    experiments: list[dict],
    baselines: list[str],
    eval_method: str,
    das_algo: str = DAST_ALGO,
    *,
    filter_outliers: bool = False,
    outlier_method: str = _DEFAULT_OUTLIER_METHOD,
    n_std: float = _DEFAULT_OUTLIER_N_STD,
    iqr_k: float = _DEFAULT_OUTLIER_IQR_K,
) -> tuple[pd.DataFrame, int, int]:
    parts: list[pd.DataFrame] = []
    for exp in experiments:
        df = compute_lift_records(
            exp["results"],
            baselines,
            eval_method,
            das_algo=das_algo,
        )
        if df.empty:
            continue
        df = df.copy()
        df["x_value"] = exp["x_value"]
        df["n_sims_file"] = exp["n_sims"]
        parts.append(df)

    if not parts:
        return pd.DataFrame(), 0, 0

    combined = pd.concat(parts, ignore_index=True)
    n_before = len(combined)
    n_filtered = 0
    if filter_outliers:
        combined, n_filtered = apply_outlier_filter(
            combined,
            "Lift",
            method=outlier_method,
            n_std=n_std,
            iqr_k=iqr_k,
        )
    n_kept_total = len(combined)

    rows = []
    for exp in experiments:
        xv = exp["x_value"]
        slice_x = combined[combined["x_value"] == xv]
        if slice_x.empty:
            continue
        for b in baselines:
            label = baseline_label(b)
            sub = slice_x[slice_x["Baseline"] == b]["Lift"]
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
                    "x_value": xv,
                    "Baseline": b,
                    "Baseline_Label": label,
                    "Mean": mean,
                    "CI": ci,
                    "N": n,
                    "n_sims_file": exp["n_sims"],
                }
            )
    return pd.DataFrame(rows), n_kept_total, n_filtered


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
    out_stem: str,
    fig_dir: Path,
) -> None:
    labels = ordered_labels(stats_df["Baseline_Label"].unique().tolist())
    x_vals = sorted(stats_df["x_value"].unique())
    x_pct = [sweep.to_display_pct(x) for x in x_vals]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    for label in labels:
        sub = stats_df[stats_df["Baseline_Label"] == label].sort_values("x_value")
        if sub.empty:
            continue
        base = COMPARATOR_COLORS.get(label, "#333333AF")
        color = to_rgba(base)
        x = np.array([sweep.to_display_pct(v) for v in sub["x_value"].values])
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
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
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
    out_path = fig_dir / f"{out_stem}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[OK] Saved: {out_path}")
    plt.close(fig)


def plot_stz_trend(
    stats_df: pd.DataFrame,
    *,
    dataset_target: str,
    sweep: SweepConfig,
    n_sims: int,
    out_stem: str,
    fig_dir: Path,
) -> None:
    """Plot DAS_advantage_ratio (STZ evaluator) trend figure."""
    labels = ordered_labels(stats_df["Baseline_Label"].unique().tolist())
    x_vals = sorted(stats_df["x_value"].unique())
    x_pct = [sweep.to_display_pct(x) for x in x_vals]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    for label in labels:
        sub = stats_df[stats_df["Baseline_Label"] == label].sort_values("x_value")
        if sub.empty:
            continue
        base = COMPARATOR_COLORS.get(label, "#333333AF")
        color = to_rgba(base)
        x = np.array([sweep.to_display_pct(v) for v in sub["x_value"].values])
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
    ax.set_ylabel("DAS advantage ratio (% of comparator)", fontweight="bold", labelpad=8)
    ax.set_title(
        f"DAS advantage ratio on {dataset_target}",
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
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

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
    out_path = fig_dir / f"{out_stem}.png"
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
        "exp_dir",
        nargs="?",
        default=None,
        help="Directory with pilot_frac_*.pkl or sample_frac_*.pkl experiment files.",
    )
    parser.add_argument(
        "--exp-dir",
        "--exp_dir",
        dest="exp_dir_opt",
        default=None,
        help="Same as positional exp_dir.",
    )
    parser.add_argument(
        "--fig-dir",
        "--fig_dir",
        default=None,
        help="Output directory for figures (default: same as --exp-dir).",
    )
    parser.add_argument(
        "--min-sims",
        "--min_sims",
        type=int,
        default=1,
        help="Minimum completed simulations required per pickle.",
    )
    parser.add_argument(
        "--metric",
        default="both",
        choices=["improvement", "advantage", "both"],
        help=(
            "improvement: DAS_improvement_ratio from OPE scalars (default). "
            "advantage:   DAS_advantage_ratio via STZ evaluator (needs *_stz.pkl). "
            "both:        save both figures."
        ),
    )
    parser.add_argument(
        "--min-sweep-value",
        "--min_sweep_value",
        "--min-pilot-frac",
        "--min_pilot_frac",
        type=float,
        default=None,
        metavar="X",
        help=(
            "Only plot sweep points with x_value >= X "
            "(pilot_frac or sample_frac). Default: plot all available points."
        ),
    )
    parser.add_argument(
        "--outlier-filter",
        action="store_true",
        help=(
            "Enable statistical outlier filtering for improvement and STZ plots "
            "(per sweep point, comparator)."
        ),
    )
    parser.add_argument(
        "--outlier-method",
        choices=["nstd", "iqr"],
        default=_DEFAULT_OUTLIER_METHOD,
        help=(
            "Outlier rule within each (sweep point, comparator) group: "
            "nstd (mean±k·std, MAD if σ=0) or iqr (Tukey Q1/Q3±k·IQR)."
        ),
    )
    parser.add_argument(
        "--outlier-n-std",
        type=float,
        default=_DEFAULT_OUTLIER_N_STD,
        help=(
            "For --outlier-method nstd: drop records with |z| above this "
            "many sample std devs (default: 3)."
        ),
    )
    parser.add_argument(
        "--outlier-iqr-k",
        type=float,
        default=_DEFAULT_OUTLIER_IQR_K,
        help=(
            "For --outlier-method iqr: Tukey fence multiplier (default: 1.5)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir_arg = args.exp_dir_opt or args.exp_dir
    if not exp_dir_arg:
        raise SystemExit("error: exp_dir required (positional or --exp-dir)")
    warnings.simplefilter(action="ignore", category=FutureWarning)
    configure_plot_style()

    exp_dir = Path(exp_dir_arg).expanduser().resolve()
    fig_dir = (
        Path(args.fig_dir).expanduser().resolve()
        if args.fig_dir is not None
        else exp_dir
    )
    sweep_kind = detect_sweep_kind(exp_dir)
    experiments, incomplete_pkls = load_experiment_pkls(
        exp_dir, min_sims=args.min_sims, sweep_kind=sweep_kind
    )
    min_x = resolve_min_sweep_value(args)
    if min_x is not None:
        experiments = filter_experiments_by_min_x(
            experiments,
            min_x,
            sweep_kind=sweep_kind,
        )

    params0, sweep = assert_consistent_experiment_meta(experiments, sweep_kind)

    baselines = discover_baselines(experiments[0]["results"])
    if not baselines:
        raise RuntimeError("No comparators found in experiment results.")

    plot_title = format_plot_title(params0)
    n_sims = int(params0.get("N_sim", 100))

    print(f"Loaded {len(experiments)} experiments from {exp_dir}")
    print(f"Figures -> {fig_dir}")
    print(f"Sweep axis: {sweep.kind}")
    print(f"{sweep.x_label}: {[e['x_value'] for e in experiments]}")
    if sweep.fixed_pilot_frac is not None:
        print(f"Fixed pilot fraction: {sweep.format_fixed_pilot_frac()}")
    print(f"Comparators: {baselines}")
    print(f"Dataset: {params0.get('dataset')}")
    print(f"Target metric (target_col): {params0.get('target_col')}")
    print(f"Plot title: {plot_title}")
    print(f"Metric: {args.metric}")

    slug = plot_title.lower().replace(" – ", "_").replace(" ", "_")
    do_improvement = args.metric in ("improvement", "both")
    do_advantage   = args.metric in ("advantage",   "both")
    filter_outliers = args.outlier_filter
    outlier_method = str(args.outlier_method)
    n_std = float(args.outlier_n_std)
    iqr_k = float(args.outlier_iqr_k)
    if filter_outliers:
        if outlier_method == "iqr":
            print(
                "Outlier filter: "
                f"IQR {iqr_k:g}× per (sweep point, comparator); "
                "skip if IQR=0 or n<4"
            )
        else:
            print(
                "Outlier filter: "
                f"{n_std:g}σ per (sweep point, comparator); MAD if σ=0"
            )

    # ---- DAS improvement ratio (OPE-based, one figure per eval method) ----
    if do_improvement:
        for ev in DEFAULT_EVAL_METHODS:
            stats_df, n_kept, n_filtered = summarize_by_sweep(
                experiments,
                baselines,
                ev,
                filter_outliers=filter_outliers,
                outlier_method=outlier_method,
                n_std=n_std,
                iqr_k=iqr_k,
            )
            if filter_outliers and n_filtered:
                print(
                    f"[INFO] eval={ev}: kept {n_kept} lift records, "
                    f"filtered {n_filtered} outlier(s)"
                )
            if stats_df.empty:
                print(f"[WARN] No data for eval_method={ev}; skip improvement plot.")
                continue
            out_stem = f"{sweep.out_stem_prefix}_{slug}_{ev}"
            plot_trend(
                stats_df,
                dataset_target=plot_title,
                sweep=sweep,
                n_sims=n_sims,
                out_stem=out_stem,
                fig_dir=fig_dir,
            )

    # ---- DAS advantage ratio (STZ evaluator, needs implementation data) ----
    if do_advantage:
        n_merged = merge_stz_sidecars(experiments)
        if n_merged:
            print(f"Merged implementation data from STZ sidecars: {n_merged} runs")
        n_runs_with_impl = sum(
            1
            for exp in experiments
            for run in exp["results"]
            if run.get("implementation") is not None
        )
        if n_runs_with_impl == 0:
            print(
                "[WARN] No run has 'implementation' data. "
                "Re-run with save_offline_data=True; STZ data lives in *_stz.pkl "
                "(local sidecar, merged automatically when present)."
            )
        else:
            stz_x = sorted({e["x_value"] for e in experiments})
            print(
                f"STZ sweep points ({sweep.x_label}): "
                f"{[sweep.to_display_pct(x) for x in stz_x]}"
            )
            print(f"Runs with implementation data: {n_runs_with_impl}")
            stz_df, n_kept, n_filtered = summarize_stz_by_sweep(
                experiments,
                baselines,
                filter_outliers=filter_outliers,
                outlier_method=outlier_method,
                n_std=n_std,
                iqr_k=iqr_k,
            )
            if filter_outliers and n_filtered:
                print(
                    f"[INFO] STZ: kept {n_kept} advantage records, "
                    f"filtered {n_filtered} outlier(s)"
                )
            if stz_df.empty:
                print("[WARN] STZ evaluator returned no finite values; skip advantage plot.")
            else:
                out_stem = f"{sweep.out_stem_prefix}_{slug}_stz"
                plot_stz_trend(
                    stz_df,
                    dataset_target=plot_title,
                    sweep=sweep,
                    n_sims=n_sims,
                    out_stem=out_stem,
                    fig_dir=fig_dir,
                )

    print_incomplete_pkl_warning(incomplete_pkls, experiments, args.min_sims)


if __name__ == "__main__":
    main()
