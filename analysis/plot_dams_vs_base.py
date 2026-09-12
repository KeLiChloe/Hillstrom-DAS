"""Compare DAST vs base segmenters and vs their DAMS variants.

Y-axis matches plot_trend.py: DAS improvement ratio
  (V_dast - V_comp) / V_comp * 100

For each family (K-Means / GMM / CLR), side-by-side bars:
  Base vs +DAMS comparator.

Example:
  python analysis/plot_dams_vs_base.py \\
    exp_aug/ablation/criteo/conversion/lightgbm_reg/sample_frac_005.pkl
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats

_ANALYSIS_DIR = Path(__file__).resolve().parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from plot_style import baseline_color, baseline_label, to_rgba  # noqa: E402

DAST_ALGO = "dast"

PAIRS = (
    ("kmeans", "kmeans_dams"),
    ("gmm", "gmm_dams"),
    ("clr", "clr_dams"),
)

EVAL_METHODS = ("dual_dr", "dr", "ipw", "dm")
EVAL_TITLES = {
    "dual_dr": "",
    "dr": "AIPW",
    "ipw": "IPW",
    "dm": "Direct Method",
}

# Neutral legend swatches (bars keep per-algo colors).
_LEGEND_GRAY = "#9E9E9E"


def configure_plot_style() -> None:
    """Match analysis/plot_trend.py typography (Times / STIX serif)."""
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


def _pkl_load(path: Path):
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as f:
            return pickle.load(f)


def resolve_main_pkl(path: Path) -> Path:
    """Accept a main experiment pkl (reject *_stz.pkl)."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")
    if path.suffix != ".pkl":
        raise ValueError(f"Expected a .pkl file, got: {path}")
    if path.stem.endswith("_stz"):
        raise ValueError(
            f"Pass the main experiment pkl, not the STZ sidecar: {path.name}"
        )
    return path


def das_improvement_ratio(vt: float, vb: float) -> float:
    """Same definition as analysis/plot_trend.py."""
    if not np.isfinite(vt) or not np.isfinite(vb) or abs(vb) < 1e-12:
        return float("nan")
    return float((vt - vb) / vb * 100.0)


def mean_ci(vals: np.ndarray) -> tuple[float, float, float, int]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(vals.mean())
    if n == 1:
        return mean, mean, mean, 1
    se = float(stats.sem(vals, ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * se)
    return mean, mean - half, mean + half, n


def load_runs(pkl_path: Path, min_sims: int) -> tuple[list[dict], dict]:
    data = _pkl_load(pkl_path)
    if not isinstance(data, dict) or "results" not in data:
        raise RuntimeError(f"Unexpected pickle format: {pkl_path}")
    results = data["results"]
    if len(results) < min_sims:
        raise RuntimeError(
            f"{pkl_path.name}: {len(results)} sims < min_sims={min_sims}"
        )
    params = dict(data.get("params", {}))
    runs = [
        run
        for run in results
        if isinstance(run, dict) and DAST_ALGO in run
    ]
    if not runs:
        raise RuntimeError(f"No runs with DAST in {pkl_path}")
    return runs, params


def available_pairs(runs: list[dict]) -> list[tuple[str, str]]:
    keys: set[str] = set()
    for run in runs:
        keys |= set(run.keys())
    return [(b, d) for b, d in PAIRS if b in keys and d in keys]


def pair_family(base: str) -> str:
    return baseline_label(base)


def _title_with_eval(base: str, eval_method: str) -> str:
    label = EVAL_TITLES.get(eval_method, eval_method).strip()
    return f"{base} ({label})" if label else base


def lifts_vs(runs: list[dict], comparator: str, eval_method: str) -> np.ndarray:
    out = []
    for r in runs:
        if eval_method not in r.get(DAST_ALGO, {}):
            continue
        if eval_method not in r.get(comparator, {}):
            continue
        out.append(
            das_improvement_ratio(r[DAST_ALGO][eval_method], r[comparator][eval_method])
        )
    return np.asarray(out, dtype=float)


def plot_das_improvement(
    runs: list[dict],
    pairs: list[tuple[str, str]],
    *,
    eval_method: str,
    dataset_target: str,
    out_stem: Path,
    n_sims: int,
) -> None:
    """Side-by-side: DAS improvement vs base vs vs DAMS, per family."""
    families = [pair_family(b) for b, _ in pairs]
    x = np.arange(len(pairs))
    width = 0.36

    base_means, base_err, dams_means, dams_err = [], [], [], []
    for base, dams in pairs:
        bm, blo, bhi, _ = mean_ci(lifts_vs(runs, base, eval_method))
        dm, dlo, dhi, _ = mean_ci(lifts_vs(runs, dams, eval_method))
        base_means.append(bm)
        dams_means.append(dm)
        base_err.append((bm - blo, bhi - bm))
        dams_err.append((dm - dlo, dhi - dm))

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar(
        x - width / 2,
        base_means,
        width,
        yerr=np.asarray(base_err).T,
        color=[to_rgba(baseline_color(b)) for b, _ in pairs],
        edgecolor="white",
        linewidth=0.6,
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        dams_means,
        width,
        yerr=np.asarray(dams_err).T,
        color=[to_rgba(baseline_color(d)) for _, d in pairs],
        edgecolor="white",
        linewidth=0.6,
        hatch="//",
        capsize=3,
    )
    ax.axhline(0.0, color="#C1121F", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontweight="bold", fontsize=11)
    ax.set_ylabel("DAS improvement ratio (%)", fontweight="bold", labelpad=8)
    title = _title_with_eval(
        "Ablation study: The incremental value of DAMS", eval_method
    )
    subtitle = (
        f"Each bar: mean ± 95% CI ({n_sims} runs);\n"
        "higher = DAS better than comparator"
    )
    # Slightly more title pad than plot_trend (pad=20) so header sits a bit farther up.
    ax.set_title(title, fontweight="bold", fontsize=15, pad=34)
    ax.text(
        0.5,
        1.02,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#444444",
        linespacing=1.25,
    )
    legend_handles = [
        Patch(
            facecolor=_LEGEND_GRAY,
            edgecolor="white",
            label="Segmentation +\nstandard model selection",
        ),
        Patch(
            facecolor=_LEGEND_GRAY,
            edgecolor="white",
            hatch="//",
            label="Segmentation + DAMS",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        framealpha=0.92,
        edgecolor="#D0D0D0",
        fancybox=False,
        fontsize=8.5,
        title="DAS improvement over",
        title_fontsize=9.5,
        borderpad=0.4,
        labelspacing=0.55,
        handletextpad=0.5,
        handlelength=1.8,
    )
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.55)
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = Path(f"{out_stem}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {out}")
    plt.close(fig)


def print_summary(runs: list[dict], pairs: list[tuple[str, str]], methods: list[str]) -> None:
    n = len(runs)
    print(f"\nDAS improvement ratio: Base vs +DAMS (n={n})")
    print("-" * 72)
    for base, dams in pairs:
        print(f"\n{pair_family(base)}")
        for method in methods:
            bm, blo, bhi, _ = mean_ci(lifts_vs(runs, base, method))
            dm, dlo, dhi, _ = mean_ci(lifts_vs(runs, dams, method))
            print(
                f"  {method:8s}  vs base={bm:+6.2f}% [{blo:+.2f},{bhi:+.2f}]  "
                f"vs DAMS={dm:+6.2f}% [{dlo:+.2f},{dhi:+.2f}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot DAS improvement ratio vs base and vs DAMS comparators."
    )
    parser.add_argument(
        "pkl",
        type=str,
        nargs="?",
        default=None,
        help="Path to one main experiment .pkl (not *_stz.pkl)",
    )
    parser.add_argument(
        "--pkl",
        dest="pkl_opt",
        type=str,
        default=None,
        help="Same as positional pkl",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: same directory as the pkl)",
    )
    parser.add_argument("--min-sims", type=int, default=1)
    parser.add_argument(
        "--eval-methods",
        nargs="+",
        default=list(EVAL_METHODS),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ablation_dams",
        help="Output stem prefix (default: ablation_dams)",
    )
    args = parser.parse_args()

    pkl_arg = args.pkl_opt or args.pkl
    if not pkl_arg:
        raise SystemExit("error: pkl path required (positional or --pkl)")
    pkl_path = resolve_main_pkl(Path(pkl_arg))
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else pkl_path.parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, params = load_runs(pkl_path, min_sims=args.min_sims)
    pairs = available_pairs(runs)
    if not pairs:
        raise SystemExit("No (base, *_dams) pairs found in results.")

    dataset = str(params.get("dataset", "unknown")).replace("_", " ").title()
    target = str(params.get("target_col", "unknown")).replace("_", " ").title()
    dataset_target = f"{dataset} – {target}"
    n = len(runs)
    methods = list(args.eval_methods)

    print(f"Loaded {n} runs from {pkl_path}")
    print_summary(runs, pairs, methods)

    for method in methods:
        stem = out_dir / f"{args.prefix}_{method}"
        plot_das_improvement(
            runs,
            pairs,
            eval_method=method,
            dataset_target=dataset_target,
            out_stem=stem,
            n_sims=n,
        )


if __name__ == "__main__":
    main()
