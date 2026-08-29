"""
Trend plot across experiment pickles.

Metrics (--metric):
  improvement   DAS_improvement_ratio from OPE scalars (dual_dr, ipw, dm).
  advantage     DAS_advantage_ratio via STZ / STZ-VR (needs *_stz.pkl).
  both          improvement + advantage (default for sample/pilot sweeps).
  net_profit    DAS net-profit improvement vs comparators (needs *_stz.pkl).
  treatment_rate  Absolute treatment rate per algorithm (needs *_stz.pkl).

Treatment-cost sweeps (default when --metric omitted):
  gross/  dm + dr (AIPW) + dual_dr + ipw + stz + stz_vr  (all cost points).
  net/    net OPE + net STZ  (all cost points; x-axis includes c=0).

Sweep axis is inferred from --exp-dir:
  pilot_frac_*, sample_frac_*, or treatment_cost_* pickles.
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

from evaluation import (
    dm_net_from_gross_dm,
    evaluate_dr_net_from_impl,
    evaluate_dual_dr_net_from_impl,
    evaluate_ipw_net_from_impl,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from scipy import stats

from stz import (
    STZ_VR,
    STZ_VR_net,
    STZ_basic,
    STZ_basic_net,
    net_profit_advantage,
    treatment_rate,
)
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
DAST_LABEL = "DAST"
DAST_COLOR = "#EF660A99"
DEFAULT_EVAL_METHODS = ["dual_dr", "dr", "ipw", "dm"]
OPE_NET_EVAL_METHODS = ["dual_dr_net", "dr_net", "ipw_net", "dm_net"]
OPE_NET_FILE_NAMES = {
    "dual_dr_net": "dual_dr",
    "dr_net": "dr",
    "ipw_net": "ipw",
    "dm_net": "dm",
}

# Display names for figure titles (evaluation method).
EVAL_METHOD_TITLE = {
    "dual_dr": "mixed Y and μ",
    "dr": "AIPW (μ + Y)",
    "ipw": "IPW (Y)",
    "dm": "Direct Method (μ only)",
    "dual_dr_net": "mixed Y and μ (net)",
    "dr_net": "AIPW (μ + Y, net)",
    "ipw_net": "IPW (Y, net)",
    "dm_net": "Direct Method (μ only, net)",
    "stz": "Y only (STZ)",
    "stz_vr": "Y only (STZ-VR)",
}

# STZ variants plotted under --metric advantage / both.
STZ_VARIANTS = (
    ("stz", STZ_basic),
    ("stz_vr", STZ_VR),
)
STZ_NET_VARIANTS = (
    ("stz", STZ_basic_net),
    ("stz_vr", STZ_VR_net),
)

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

SweepKind = Literal["pilot_frac", "sample_frac", "treatment_cost"]

# Loader base sizes used when *_stz.pkl cohort_size is unavailable.
# Criteo loaders use percent10=True → base is the 10% public slice.
_DATASET_BASE_N = {
    "hillstrom": 64_000,
    "criteo": 1_397_960,
}


@dataclass(frozen=True)
class SweepConfig:
    kind: SweepKind
    fixed_pilot_frac: float | None = None
    dataset: str | None = None
    # stored sweep value (pilot_frac / sample_frac) → plotted N
    n_by_x: tuple[tuple[float, int], ...] = ()

    @property
    def x_param(self) -> str:
        return self.kind

    @property
    def x_label(self) -> str:
        if self.kind == "treatment_cost":
            return r"Treatment cost $c$ ($\times 10^{-3}$)"
        if self.kind == "pilot_frac":
            return r"Pilot size $N$ ($\times 10^{3}$)"
        return r"Sample size $N$ ($\times 10^{3}$)"

    @property
    def out_stem_prefix(self) -> str:
        return f"trend_{self.kind}"

    def to_display_x(self, x_value: float) -> float:
        """Map stored sweep value → plotted x coordinate."""
        if self.kind == "treatment_cost":
            return float(x_value) * 1000.0
        for xv, n in self.n_by_x:
            if abs(xv - float(x_value)) < 1e-9:
                return float(n) / 1000.0
        raise KeyError(
            f"No cohort/pilot size N for stored {self.kind}={x_value!r}; "
            f"known={dict(self.n_by_x)}"
        )

    # Back-compat alias used by older call sites / notebooks.
    def to_display_pct(self, x_value: float) -> float:
        return self.to_display_x(x_value)

    def format_fixed_pilot_frac(self) -> str:
        if self.fixed_pilot_frac is None:
            return ""
        pct = self.fixed_pilot_frac * 100.0
        if abs(pct - round(pct)) < 1e-9:
            return f"{int(round(pct))}%"
        return f"{pct:g}%"

    def subtitle(self, n_sims: int) -> str:
        base = f"Each data point: mean ± 95% CI ({n_sims} runs)"
        if self.kind == "treatment_cost":
            return base
        if self.fixed_pilot_frac is not None:
            return f"{base}; fixed pilot fraction = {self.format_fixed_pilot_frac()}"
        return base


def algo_display_label(algo: str) -> str:
    if algo == DAST_ALGO:
        return DAST_LABEL
    return baseline_label(algo)


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
    has_cost = bool(
        list(p for p in exp_dir.glob("treatment_cost_*.pkl") if _is_main_pkl(p))
    )
    has_sample = bool(
        list(p for p in exp_dir.glob("sample_frac_*.pkl") if _is_main_pkl(p))
    )
    has_pilot = bool(
        list(p for p in exp_dir.glob("pilot_frac_*.pkl") if _is_main_pkl(p))
    )

    if has_cost:
        return "treatment_cost"
    if has_pilot and not has_sample:
        return "pilot_frac"
    if has_sample and not has_pilot:
        return "sample_frac"

    # e.g. pilot_frac_with_fixed_0.1_sample_frac vs sample_frac_with_fixed_020_pilot
    dir_name = exp_dir.name.lower()
    if "treatment_cost" in dir_name:
        return "treatment_cost"
    if dir_name.startswith("pilot_frac"):
        return "pilot_frac"
    if dir_name.startswith("sample_frac"):
        return "sample_frac"

    path_lower = exp_dir.as_posix().lower()
    if "treatment_cost" in path_lower:
        return "treatment_cost"
    if "pilot_frac" in path_lower:
        return "pilot_frac"
    if "sample_frac" in path_lower:
        return "sample_frac"
    return "pilot_frac"


def discover_sweep_value(kind: SweepKind, path: Path, params: dict) -> float:
    if kind == "treatment_cost":
        if "treatment_cost" in params:
            return float(params["treatment_cost"])
        m = re.search(r"treatment_cost_(.+)$", path.stem)
        if m:
            return float(m.group(1).replace("p", "."))
        raise ValueError(f"Cannot infer treatment_cost from {path}")
    if kind in params:
        return float(params[kind])
    # Allow half-percent tags: sample_frac_002.5.pkl → 0.025
    m = re.search(rf"{kind}_(\d+(?:\.\d+)?)", path.stem)
    if m:
        return float(m.group(1)) / 100.0
    raise ValueError(f"Cannot infer {kind} from {path}")


def _cohort_size_from_stz(main_path: Path) -> int | None:
    """Read cohort_size from the first STZ sidecar implementation block."""
    stz_path = _resolve_stz_path(Path(main_path))
    if not stz_path.is_file():
        return None
    stz_data = _pkl_load(stz_path)
    for run in stz_data.get("results", []):
        if not isinstance(run, dict):
            continue
        impl = run.get("implementation")
        if isinstance(impl, dict) and impl.get("cohort_size") is not None:
            return int(impl["cohort_size"])
    return None


def _cohort_size_from_params(params: dict) -> int:
    """Fallback: base loader size × sample_frac (matches load_* subsampling)."""
    dataset = str(params.get("dataset", "")).lower()
    if dataset not in _DATASET_BASE_N:
        raise KeyError(
            f"Unknown dataset {dataset!r} for N fallback; "
            f"known={sorted(_DATASET_BASE_N)}"
        )
    sample_frac = float(params["sample_frac"])
    return int(_DATASET_BASE_N[dataset] * sample_frac)


def resolve_cohort_size(path: Path, params: dict) -> int:
    n = _cohort_size_from_stz(path)
    if n is not None:
        return n
    return _cohort_size_from_params(params)


def display_n_for_experiment(sweep_kind: SweepKind, exp: dict) -> int:
    """N plotted on the x-axis for one sweep point (unused for treatment_cost)."""
    if sweep_kind == "treatment_cost":
        return 0
    n_cohort = int(exp["n_cohort"])
    if sweep_kind == "sample_frac":
        return n_cohort
    # pilot_frac: show pilot size (fraction of the fixed cohort)
    return int(round(float(exp["x_value"]) * n_cohort))


def load_experiment_pkls(
    exp_dir: Path, min_sims: int, sweep_kind: SweepKind
) -> tuple[list[dict], list[dict]]:
    if sweep_kind == "treatment_cost":
        pkls = sorted(
            p for p in exp_dir.glob("treatment_cost_*.pkl") if _is_main_pkl(p)
        )
    else:
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
        x_value = discover_sweep_value(sweep_kind, path, params)
        n_cohort = resolve_cohort_size(path, params)
        exp = {
            "path": path,
            "x_value": x_value,
            "n_cohort": n_cohort,
            "params": params,
            "results": data["results"],
            "n_sims": n_runs,
        }
        exp["x_plot"] = display_n_for_experiment(sweep_kind, exp)
        loaded.append(exp)
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
    *,
    stz_fn=STZ_VR,
    treatment_cost: float | None = None,
) -> pd.DataFrame:
    """Per-run STZ advantage for every (DAST, comparator) pair."""
    records = []
    for i, run in enumerate(results_list):
        for b in baselines:
            if treatment_cost is not None:
                adv = stz_fn(run, das_algo, b, float(treatment_cost))
            else:
                adv = stz_fn(run, das_algo, b)
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
    stz_fn=STZ_VR,
    net_stz: bool = False,
    filter_outliers: bool = False,
    outlier_method: str = _DEFAULT_OUTLIER_METHOD,
    n_std: float = _DEFAULT_OUTLIER_N_STD,
    iqr_k: float = _DEFAULT_OUTLIER_IQR_K,
) -> tuple[pd.DataFrame, int, int]:
    """Mean ± 95% CI of STZ advantage across runs, one row per (x_value, comparator)."""
    parts: list[pd.DataFrame] = []
    for exp in experiments:
        tc = float(exp["params"]["treatment_cost"]) if net_stz else None
        df = compute_stz_records(
            exp["results"],
            baselines,
            das_algo=das_algo,
            stz_fn=stz_fn,
            treatment_cost=tc,
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


def compute_net_ope_value(
    run: dict,
    algo: str,
    eval_method: str,
    treatment_cost: float,
) -> float:
    """Net OPE scalar for one run, computed offline (no extra pkl fields)."""
    method = str(eval_method).lower()
    c = float(treatment_cost)
    if method == "dm_net":
        dm = safe_get_value(run, algo, "dm")
        tr = treatment_rate(run, algo)
        return dm_net_from_gross_dm(dm, tr, c)

    impl = run.get("implementation")
    if impl is None:
        return float("nan")
    try:
        if method == "dual_dr_net":
            if impl.get("Gamma") is None:
                return float("nan")
            return float(
                evaluate_dual_dr_net_from_impl(impl, algo, c)["value_mean"]
            )
        if method == "dr_net":
            if impl.get("mu_impl") is None:
                return float("nan")
            return float(
                evaluate_dr_net_from_impl(impl, algo, c)["value_mean"]
            )
        if method == "ipw_net":
            return float(
                evaluate_ipw_net_from_impl(impl, algo, c)["value_mean"]
            )
    except (KeyError, ValueError):
        return float("nan")
    return float("nan")


def compute_net_lift_records(
    results_list: list[dict],
    baselines: list[str],
    eval_method: str,
    treatment_cost: float,
    das_algo: str = DAST_ALGO,
) -> pd.DataFrame:
    """Per-run net-OPE lift records (offline from gross dm + *_stz.pkl)."""
    records = []
    for i, run in enumerate(results_list):
        vt = compute_net_ope_value(run, das_algo, eval_method, treatment_cost)
        if not np.isfinite(vt):
            continue
        for b in baselines:
            vb = compute_net_ope_value(run, b, eval_method, treatment_cost)
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


def summarize_net_ope_by_sweep(
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
    """Like summarize_by_sweep, but net OPE is derived offline at plot time."""
    parts: list[pd.DataFrame] = []
    for exp in experiments:
        cost = float(exp["params"]["treatment_cost"])
        df = compute_net_lift_records(
            exp["results"],
            baselines,
            eval_method,
            cost,
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


def compute_net_profit_records(
    results_list: list[dict],
    baselines: list[str],
    treatment_cost: float,
    das_algo: str = DAST_ALGO,
) -> pd.DataFrame:
    """Per-run net-profit improvement (DAST vs comparator) at one cost level."""
    records = []
    for i, run in enumerate(results_list):
        for b in baselines:
            adv = net_profit_advantage(run, das_algo, b, treatment_cost)
            if not np.isfinite(adv):
                continue
            records.append(
                {
                    "Run": i,
                    "Baseline": b,
                    "Baseline_Label": baseline_label(b),
                    "NetProfitLift": float(adv),
                }
            )
    return pd.DataFrame(records)


def summarize_net_profit_by_sweep(
    experiments: list[dict],
    baselines: list[str],
    das_algo: str = DAST_ALGO,
    *,
    filter_outliers: bool = False,
    outlier_method: str = _DEFAULT_OUTLIER_METHOD,
    n_std: float = _DEFAULT_OUTLIER_N_STD,
    iqr_k: float = _DEFAULT_OUTLIER_IQR_K,
) -> tuple[pd.DataFrame, int, int]:
    """Mean ± 95% CI of net-profit improvement across runs."""
    parts: list[pd.DataFrame] = []
    for exp in experiments:
        cost = float(exp["params"]["treatment_cost"])
        df = compute_net_profit_records(
            exp["results"], baselines, cost, das_algo=das_algo
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
    n_filtered = 0
    if filter_outliers:
        combined, n_filtered = apply_outlier_filter(
            combined,
            "NetProfitLift",
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
            sub = slice_x[slice_x["Baseline"] == b]["NetProfitLift"]
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


def compute_treatment_rate_records(
    results_list: list[dict],
    algos: list[str],
) -> pd.DataFrame:
    """Per-run treatment rate (fraction treated) for each algorithm."""
    records = []
    for i, run in enumerate(results_list):
        for algo in algos:
            rate = treatment_rate(run, algo)
            if not np.isfinite(rate):
                continue
            records.append(
                {
                    "Run": i,
                    "Algo": algo,
                    "Algo_Label": algo_display_label(algo),
                    "TreatmentRate": float(rate) * 100.0,
                }
            )
    return pd.DataFrame(records)


def summarize_treatment_rate_by_sweep(
    experiments: list[dict],
    algos: list[str],
    *,
    filter_outliers: bool = False,
    outlier_method: str = _DEFAULT_OUTLIER_METHOD,
    n_std: float = _DEFAULT_OUTLIER_N_STD,
    iqr_k: float = _DEFAULT_OUTLIER_IQR_K,
) -> tuple[pd.DataFrame, int, int]:
    """Mean ± 95% CI of treatment rate (%) per algorithm and sweep point."""
    parts: list[pd.DataFrame] = []
    for exp in experiments:
        df = compute_treatment_rate_records(exp["results"], algos)
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
            "TreatmentRate",
            method=outlier_method,
            group_cols=("x_value", "Algo"),
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
        for algo in algos:
            label = algo_display_label(algo)
            sub = slice_x[slice_x["Algo"] == algo]["TreatmentRate"]
            if sub.empty:
                continue
            n = len(sub)
            mean = float(sub.mean())
            ci = float(sub.sem() * stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
            rows.append(
                {
                    "x_value": xv,
                    "Algo": algo,
                    "Algo_Label": label,
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


def eval_method_title(eval_method: str) -> str:
    """Human-readable evaluation method for plot titles."""
    key = str(eval_method).lower()
    if key in EVAL_METHOD_TITLE:
        return EVAL_METHOD_TITLE[key]
    return str(eval_method).replace("_", " ")


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
    elif sweep_kind == "treatment_cost":
        if "sample_frac" not in ref or "pilot_frac" not in ref:
            raise ValueError(
                "treatment_cost sweep requires sample_frac and pilot_frac in params."
            )
        fixed_sample = float(ref["sample_frac"])
        fixed_pilot = float(ref["pilot_frac"])

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
        elif sweep_kind == "treatment_cost":
            if float(p.get("sample_frac", -1)) != fixed_sample:
                raise ValueError(
                    f"Inconsistent sample_frac in {exp['path'].name}: "
                    f"expected {fixed_sample!r}, got {p.get('sample_frac')!r}."
                )
            if float(p.get("pilot_frac", -1)) != fixed_pilot:
                raise ValueError(
                    f"Inconsistent pilot_frac in {exp['path'].name}: "
                    f"expected {fixed_pilot!r}, got {p.get('pilot_frac')!r}."
                )

    n_by_x = tuple(
        (float(exp["x_value"]), int(exp["x_plot"])) for exp in experiments
    )
    sweep = SweepConfig(
        kind=sweep_kind,
        fixed_pilot_frac=fixed_pilot,
        dataset=str(dataset) if dataset is not None else None,
        n_by_x=n_by_x,
    )
    return ref, sweep


def _format_n_tick(value: float, _pos=None) -> str:
    """Sample-size ticks in thousands (K)."""
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}"
    return f"{value:.1f}"


def _format_cost_tick(value: float, _pos=None) -> str:
    """Treatment-cost ticks (stored c × 10³)."""
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}"
    return f"{value:g}"


def _x_tick_formatter(sweep: SweepConfig):
    if sweep.kind == "treatment_cost":
        return _format_cost_tick
    return _format_n_tick


def plot_treatment_rate_trend(
    stats_df: pd.DataFrame,
    *,
    dataset_target: str,
    sweep: SweepConfig,
    n_sims: int,
    out_stem: str,
    fig_dir: Path,
) -> None:
    """Absolute treatment rate (%) vs sweep axis."""
    other_labels = ordered_labels(
        [
            algo_display_label(a)
            for a in stats_df["Algo"].unique()
            if a != DAST_ALGO
        ]
    )
    labels = [DAST_LABEL] + [lb for lb in other_labels if lb != DAST_LABEL]
    x_vals = sorted(stats_df["x_value"].unique())
    x_n = [sweep.to_display_x(x) for x in x_vals]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    for label in labels:
        sub = stats_df[stats_df["Algo_Label"] == label].sort_values("x_value")
        if sub.empty:
            continue
        if label == DAST_LABEL:
            color = to_rgba(DAST_COLOR)
            zorder = 4
            linewidth = 2.0
        else:
            color = to_rgba(COMPARATOR_COLORS.get(label, "#333333AF"))
            zorder = 3
            linewidth = 1.4
        x = np.array([sweep.to_display_x(v) for v in sub["x_value"].values])
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
            linewidth=linewidth,
            label=label,
            zorder=zorder,
        )

    ax.set_xlabel(sweep.x_label, fontweight="bold", labelpad=8)
    ax.set_ylabel("Treatment rate (%)", fontweight="bold", labelpad=8)
    ax.set_title(
        f"Treatment rate on {dataset_target}",
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

    ax.set_xticks(x_n)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_x_tick_formatter(sweep)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

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
        title="Algorithm",
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


def plot_net_profit_trend(
    stats_df: pd.DataFrame,
    *,
    dataset_target: str,
    sweep: SweepConfig,
    n_sims: int,
    out_stem: str,
    fig_dir: Path,
) -> None:
    """Net-profit improvement ratio (DAST vs comparators)."""
    plot_stz_trend(
        stats_df,
        dataset_target=dataset_target,
        sweep=sweep,
        n_sims=n_sims,
        out_stem=out_stem,
        fig_dir=fig_dir,
        eval_method="net_profit",
        ylabel="DAS net-profit improvement ratio",
        title_suffix="net profit (factual Y − cost)",
    )


def plot_trend(
    stats_df: pd.DataFrame,
    *,
    dataset_target: str,
    sweep: SweepConfig,
    n_sims: int,
    out_stem: str,
    fig_dir: Path,
    eval_method: str = "dual_dr",
) -> None:
    labels = ordered_labels(stats_df["Baseline_Label"].unique().tolist())
    x_vals = sorted(stats_df["x_value"].unique())
    x_n = [sweep.to_display_x(x) for x in x_vals]
    eval_label = eval_method_title(eval_method)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    for label in labels:
        sub = stats_df[stats_df["Baseline_Label"] == label].sort_values("x_value")
        if sub.empty:
            continue
        base = COMPARATOR_COLORS.get(label, "#333333AF")
        color = to_rgba(base)
        x = np.array([sweep.to_display_x(v) for v in sub["x_value"].values])
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
        f"DAS improvement ratio on {dataset_target} ({eval_label})",
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

    ax.set_xticks(x_n)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_x_tick_formatter(sweep)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

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
    eval_method: str = "stz_vr",
    ylabel: str | None = None,
    title_suffix: str | None = None,
) -> None:
    """Plot DAS_advantage_ratio (STZ / STZ-VR) or net-profit improvement trend."""
    labels = ordered_labels(stats_df["Baseline_Label"].unique().tolist())
    x_vals = sorted(stats_df["x_value"].unique())
    x_n = [sweep.to_display_x(x) for x in x_vals]
    if ylabel is None:
        ylabel = "DAS advantage ratio (% of comparator)"
    if title_suffix is None:
        title_suffix = eval_method_title(eval_method)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    for label in labels:
        sub = stats_df[stats_df["Baseline_Label"] == label].sort_values("x_value")
        if sub.empty:
            continue
        base = COMPARATOR_COLORS.get(label, "#333333AF")
        color = to_rgba(base)
        x = np.array([sweep.to_display_x(v) for v in sub["x_value"].values])
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
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=8)
    ax.set_title(
        f"DAS improvement on {dataset_target} ({title_suffix})",
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

    ax.set_xticks(x_n)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_x_tick_formatter(sweep)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

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
        help="Directory with experiment pickle files.",
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
        default=None,
        choices=[
            "improvement",
            "advantage",
            "both",
            "net_profit",
            "treatment_rate",
        ],
        help=(
            "improvement: OPE lift (dual_dr, ipw, dm). "
            "advantage: STZ / STZ-VR (needs *_stz.pkl). "
            "both: improvement + advantage (default for sample/pilot sweeps). "
            "net_profit / treatment_rate: cost-aware metrics (needs *_stz.pkl). "
            "Omitted on treatment_cost sweeps: figures go to gross/ and net/ only."
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


def _run_improvement_plots(
    experiments: list[dict],
    baselines: list[str],
    *,
    sweep: SweepConfig,
    plot_title: str,
    slug: str,
    n_sims: int,
    fig_dir: Path,
    filter_outliers: bool,
    outlier_method: str,
    n_std: float,
    iqr_k: float,
    eval_methods: list[str] | None = None,
    file_eval_names: dict[str, str] | None = None,
    offline_net: bool = False,
) -> None:
    methods = eval_methods or DEFAULT_EVAL_METHODS
    for ev in methods:
        if offline_net:
            stats_df, n_kept, n_filtered = summarize_net_ope_by_sweep(
                experiments,
                baselines,
                ev,
                filter_outliers=filter_outliers,
                outlier_method=outlier_method,
                n_std=n_std,
                iqr_k=iqr_k,
            )
        else:
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
                f"[INFO] OPE eval={ev}: kept {n_kept} lift records, "
                f"filtered {n_filtered} outlier(s)"
            )
        if stats_df.empty:
            print(f"[WARN] No data for eval_method={ev}; skip OPE plot.")
            continue
        file_ev = (file_eval_names or {}).get(ev, ev)
        out_stem = f"{sweep.out_stem_prefix}_{slug}_{file_ev}"
        plot_trend(
            stats_df,
            dataset_target=plot_title,
            sweep=sweep,
            n_sims=n_sims,
            out_stem=out_stem,
            fig_dir=fig_dir,
            eval_method=ev,
        )


def _run_advantage_plots(
    experiments: list[dict],
    baselines: list[str],
    *,
    sweep: SweepConfig,
    plot_title: str,
    slug: str,
    n_sims: int,
    fig_dir: Path,
    filter_outliers: bool,
    outlier_method: str,
    n_std: float,
    iqr_k: float,
    stz_variants=STZ_VARIANTS,
    net_stz: bool = False,
) -> None:
    for stz_key, stz_fn in stz_variants:
        stz_df, n_kept, n_filtered = summarize_stz_by_sweep(
            experiments,
            baselines,
            stz_fn=stz_fn,
            net_stz=net_stz,
            filter_outliers=filter_outliers,
            outlier_method=outlier_method,
            n_std=n_std,
            iqr_k=iqr_k,
        )
        if filter_outliers and n_filtered:
            print(
                f"[INFO] {stz_key}: kept {n_kept} advantage records, "
                f"filtered {n_filtered} outlier(s)"
            )
        if stz_df.empty:
            print(f"[WARN] {stz_key} returned no finite values; skip plot.")
            continue
        out_stem = f"{sweep.out_stem_prefix}_{slug}_{stz_key}"
        plot_stz_trend(
            stz_df,
            dataset_target=plot_title,
            sweep=sweep,
            n_sims=n_sims,
            out_stem=out_stem,
            fig_dir=fig_dir,
            eval_method=stz_key,
        )


def _run_treatment_cost_plots(
    experiments: list[dict],
    baselines: list[str],
    *,
    sweep: SweepConfig,
    plot_title: str,
    slug: str,
    n_sims: int,
    fig_dir: Path,
    filter_outliers: bool,
    outlier_method: str,
    n_std: float,
    iqr_k: float,
) -> None:
    """Plot all cost points into gross/ and net/ (evaluation-type subdirs only)."""
    gross_dir = fig_dir / "gross"
    net_dir = fig_dir / "net"

    # Remove legacy root-level trend figures from older plot_trend versions.
    for old in fig_dir.glob("trend_treatment_cost_*.png"):
        old.unlink()
        print(f"[INFO] Removed legacy figure: {old.name}")

    print("=== gross/ (all treatment_cost points) ===")
    _run_improvement_plots(
        experiments,
        baselines,
        sweep=sweep,
        plot_title=plot_title,
        slug=slug,
        n_sims=n_sims,
        fig_dir=gross_dir,
        filter_outliers=filter_outliers,
        outlier_method=outlier_method,
        n_std=n_std,
        iqr_k=iqr_k,
    )
    _run_advantage_plots(
        experiments,
        baselines,
        sweep=sweep,
        plot_title=plot_title,
        slug=slug,
        n_sims=n_sims,
        fig_dir=gross_dir,
        filter_outliers=filter_outliers,
        outlier_method=outlier_method,
        n_std=n_std,
        iqr_k=iqr_k,
    )

    print("=== net/ (all treatment_cost points) ===")
    _run_improvement_plots(
        experiments,
        baselines,
        sweep=sweep,
        plot_title=plot_title,
        slug=slug,
        n_sims=n_sims,
        fig_dir=net_dir,
        filter_outliers=filter_outliers,
        outlier_method=outlier_method,
        n_std=n_std,
        iqr_k=iqr_k,
        eval_methods=OPE_NET_EVAL_METHODS,
        file_eval_names=OPE_NET_FILE_NAMES,
        offline_net=True,
    )
    _run_advantage_plots(
        experiments,
        baselines,
        sweep=sweep,
        plot_title=plot_title,
        slug=slug,
        n_sims=n_sims,
        fig_dir=net_dir,
        filter_outliers=filter_outliers,
        outlier_method=outlier_method,
        n_std=n_std,
        iqr_k=iqr_k,
        stz_variants=STZ_NET_VARIANTS,
        net_stz=True,
    )


def _run_net_profit_plots(
    experiments: list[dict],
    baselines: list[str],
    *,
    sweep: SweepConfig,
    plot_title: str,
    slug: str,
    n_sims: int,
    fig_dir: Path,
    filter_outliers: bool,
    outlier_method: str,
    n_std: float,
    iqr_k: float,
) -> None:
    stats_df, n_kept, n_filtered = summarize_net_profit_by_sweep(
        experiments,
        baselines,
        filter_outliers=filter_outliers,
        outlier_method=outlier_method,
        n_std=n_std,
        iqr_k=iqr_k,
    )
    if filter_outliers and n_filtered:
        print(
            f"[INFO] net_profit: kept {n_kept} records, "
            f"filtered {n_filtered} outlier(s)"
        )
    if stats_df.empty:
        print("[WARN] No net-profit data; skip net profit plot.")
        return
    out_stem = f"{sweep.out_stem_prefix}_{slug}_net_profit"
    plot_net_profit_trend(
        stats_df,
        dataset_target=plot_title,
        sweep=sweep,
        n_sims=n_sims,
        out_stem=out_stem,
        fig_dir=fig_dir,
    )


def _run_treatment_rate_plots(
    experiments: list[dict],
    baselines: list[str],
    *,
    sweep: SweepConfig,
    plot_title: str,
    slug: str,
    n_sims: int,
    fig_dir: Path,
    filter_outliers: bool,
    outlier_method: str,
    n_std: float,
    iqr_k: float,
) -> None:
    algos = [DAST_ALGO] + baselines
    stats_df, n_kept, n_filtered = summarize_treatment_rate_by_sweep(
        experiments,
        algos,
        filter_outliers=filter_outliers,
        outlier_method=outlier_method,
        n_std=n_std,
        iqr_k=iqr_k,
    )
    if filter_outliers and n_filtered:
        print(
            f"[INFO] treatment_rate: kept {n_kept} records, "
            f"filtered {n_filtered} outlier(s)"
        )
    if stats_df.empty:
        print("[WARN] No treatment-rate data; skip treatment rate plot.")
        return
    out_stem = f"{sweep.out_stem_prefix}_{slug}_treatment_rate"
    plot_treatment_rate_trend(
        stats_df,
        dataset_target=plot_title,
        sweep=sweep,
        n_sims=n_sims,
        out_stem=out_stem,
        fig_dir=fig_dir,
    )


def _ensure_implementation_data(experiments: list[dict]) -> int:
    n_merged = merge_stz_sidecars(experiments)
    if n_merged:
        print(f"Merged implementation data from STZ sidecars: {n_merged} runs")
    return sum(
        1
        for exp in experiments
        for run in exp["results"]
        if run.get("implementation") is not None
    )


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
    metric = args.metric
    if metric is None:
        metric = "treatment_cost" if sweep_kind == "treatment_cost" else "both"
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
    print(
        f"{sweep.x_label}: "
        f"{[sweep.to_display_x(e['x_value']) for e in experiments]} "
        f"(stored {[sweep.kind]}={[e['x_value'] for e in experiments]})"
    )
    if sweep.fixed_pilot_frac is not None:
        print(f"Fixed pilot fraction: {sweep.format_fixed_pilot_frac()}")
    print(f"Comparators: {baselines}")
    print(f"Dataset: {params0.get('dataset')}")
    print(f"Target metric (target_col): {params0.get('target_col')}")
    print(f"Plot title: {plot_title}")
    print(f"Metric: {metric}")

    slug = plot_title.lower().replace(" – ", "_").replace(" ", "_")
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

    if metric == "treatment_cost":
        n_runs_with_impl = _ensure_implementation_data(experiments)
        if n_runs_with_impl == 0:
            print(
                "[WARN] No run has 'implementation' data; STZ plots will be skipped. "
                "Re-run with save_offline_data=True."
            )
        _run_treatment_cost_plots(
            experiments,
            baselines,
            sweep=sweep,
            plot_title=plot_title,
            slug=slug,
            n_sims=n_sims,
            fig_dir=fig_dir,
            filter_outliers=filter_outliers,
            outlier_method=outlier_method,
            n_std=n_std,
            iqr_k=iqr_k,
        )
        print_incomplete_pkl_warning(incomplete_pkls, experiments, args.min_sims)
        return

    do_improvement = metric in ("improvement", "both")
    do_advantage = metric in ("advantage", "both")
    do_net_profit = metric == "net_profit"
    do_treatment_rate = metric == "treatment_rate"

    # ---- OPE improvement ----
    if do_improvement:
        _run_improvement_plots(
            experiments,
            baselines,
            sweep=sweep,
            plot_title=plot_title,
            slug=slug,
            n_sims=n_sims,
            fig_dir=fig_dir,
            filter_outliers=filter_outliers,
            outlier_method=outlier_method,
            n_std=n_std,
            iqr_k=iqr_k,
        )

    needs_impl = do_advantage or do_net_profit or do_treatment_rate
    if needs_impl:
        n_runs_with_impl = _ensure_implementation_data(experiments)
        if n_runs_with_impl == 0:
            print(
                "[WARN] No run has 'implementation' data. "
                "Re-run with save_offline_data=True; STZ data lives in *_stz.pkl "
                "(local sidecar, merged automatically when present)."
            )
        else:
            if do_net_profit or do_treatment_rate:
                impl_x = sorted({e["x_value"] for e in experiments})
                print(
                    f"Impl sweep points ({sweep.x_label}): "
                    f"{[sweep.to_display_x(x) for x in impl_x]}"
                )
                print(f"Runs with implementation data: {n_runs_with_impl}")

            if do_net_profit:
                _run_net_profit_plots(
                    experiments,
                    baselines,
                    sweep=sweep,
                    plot_title=plot_title,
                    slug=slug,
                    n_sims=n_sims,
                    fig_dir=fig_dir,
                    filter_outliers=filter_outliers,
                    outlier_method=outlier_method,
                    n_std=n_std,
                    iqr_k=iqr_k,
                )

            if do_treatment_rate:
                _run_treatment_rate_plots(
                    experiments,
                    baselines,
                    sweep=sweep,
                    plot_title=plot_title,
                    slug=slug,
                    n_sims=n_sims,
                    fig_dir=fig_dir,
                    filter_outliers=filter_outliers,
                    outlier_method=outlier_method,
                    n_std=n_std,
                    iqr_k=iqr_k,
                )

    # ---- DAS advantage ratio (STZ / STZ-VR) ----
    if do_advantage:
        n_runs_with_impl = sum(
            1
            for exp in experiments
            for run in exp["results"]
            if run.get("implementation") is not None
        )
        if n_runs_with_impl == 0:
            if not needs_impl:
                _ensure_implementation_data(experiments)
            print("[WARN] STZ advantage skipped: no implementation data.")
        else:
            stz_x = sorted({e["x_value"] for e in experiments})
            print(
                f"STZ sweep points ({sweep.x_label}): "
                f"{[sweep.to_display_x(x) for x in stz_x]}"
            )
            print(f"Runs with implementation data: {n_runs_with_impl}")
            _run_advantage_plots(
                experiments,
                baselines,
                sweep=sweep,
                plot_title=plot_title,
                slug=slug,
                n_sims=n_sims,
                fig_dir=fig_dir,
                filter_outliers=filter_outliers,
                outlier_method=outlier_method,
                n_std=n_std,
                iqr_k=iqr_k,
            )

    print_incomplete_pkl_warning(incomplete_pkls, experiments, args.min_sims)


if __name__ == "__main__":
    main()
