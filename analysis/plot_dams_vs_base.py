"""Compare DAST vs base segmenters and vs their DAMS variants.

Y-axis matches plot_trend.py: DAS improvement ratio
  (V_dast - V_comp) / V_comp * 100

Default layout (main text): one stacked bar per family
  - total height  = gap vs standard model selection  (i)
  - bottom segment = part of that gap closed by DAMS (ii)
  - top segment    = remaining gap vs Segmentation+DAMS

Appendix layout (--layout side_by_side): two bars per family
  (DAS improvement vs base vs vs +DAMS).

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
from stz import STZ_VR, STZ_basic  # noqa: E402

DAST_ALGO = "dast"

PAIRS = (
    ("kmeans", "kmeans_dams"),
    ("gmm", "gmm_dams"),
    ("clr", "clr_dams"),
)

EVAL_METHODS = ("dual_dr", "dr", "ipw", "dm")
STZ_EVAL_METHODS = {
    "stz": STZ_basic,
    "stz_vr": STZ_VR,
}
EVAL_TITLES = {
    "dual_dr": "",
    "dr": "AIPW",
    "ipw": "IPW",
    "dm": "Direct Method",
    "stz": "Matched-Action",
    "stz_vr": "Matched-Action-VR",
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


def attach_stz_sidecar(runs: list[dict], pkl_path: Path) -> int:
    """Attach implementation blocks from the sibling *_stz.pkl, matched by seed."""
    stz_path = pkl_path.with_name(pkl_path.stem + "_stz" + pkl_path.suffix)
    if not stz_path.is_file():
        raise FileNotFoundError(
            f"STZ eval needs the sidecar next to the main pkl: {stz_path.name}"
        )
    stz_data = _pkl_load(stz_path)
    impl_by_seed = {
        run["seed"]: run["implementation"]
        for run in stz_data.get("results", [])
        if isinstance(run, dict) and "implementation" in run and "seed" in run
    }
    attached = 0
    for run in runs:
        impl = impl_by_seed.get(run.get("seed"))
        if impl is None:
            continue
        run["implementation"] = impl
        attached += 1
    if attached == 0:
        raise RuntimeError(f"No implementation blocks matched from {stz_path.name}")
    return attached


def lifts_vs(runs: list[dict], comparator: str, eval_method: str) -> np.ndarray:
    stz_fn = STZ_EVAL_METHODS.get(eval_method)
    if stz_fn is not None:
        out = [stz_fn(r, DAST_ALGO, comparator) for r in runs]
        return np.asarray(out, dtype=float)

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


def _family_gap_stats(
    runs: list[dict],
    pairs: list[tuple[str, str]],
    eval_method: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return families, gap_vs_base, remaining, closed (means only; no CI)."""
    families: list[str] = []
    base_means: list[float] = []
    dams_means: list[float] = []
    for base, dams in pairs:
        bm, _, _, _ = mean_ci(lifts_vs(runs, base, eval_method))
        dm, _, _, _ = mean_ci(lifts_vs(runs, dams, eval_method))
        families.append(pair_family(base))
        base_means.append(bm)
        dams_means.append(dm)
    gap_base = np.asarray(base_means, dtype=float)
    gap_dams = np.asarray(dams_means, dtype=float)
    # Stack always sums to gap_vs_standard: bottom=closed by DAMS, top=remaining.
    remaining = np.clip(gap_dams, 0.0, None)
    closed = gap_base - remaining
    worse = closed < 0  # DAMS widens the gap (rare): show full bar as remaining
    closed = np.where(worse, 0.0, closed)
    remaining = np.where(worse, gap_base, remaining)
    return families, gap_base, remaining, closed


def plot_das_improvement_stacked(
    runs: list[dict],
    pairs: list[tuple[str, str]],
    *,
    eval_method: str,
    out_stem: Path,
    n_sims: int,
) -> None:
    """Stacked bars: total = gap vs standard; bottom = closed by DAMS; top = remaining."""
    families, gap_base, remaining, closed = _family_gap_stats(
        runs, pairs, eval_method
    )
    x = np.arange(len(pairs), dtype=float)
    width = 0.62

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    closed_colors = [to_rgba(baseline_color(d)) for _, d in pairs]
    remain_colors = [to_rgba(baseline_color(b)) for b, _ in pairs]

    ax.bar(
        x,
        closed,
        width,
        color=closed_colors,
        edgecolor="white",
        linewidth=0.6,
        hatch="//",
    )
    ax.bar(
        x,
        remaining,
        width,
        bottom=closed,
        color=remain_colors,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.axhline(0.0, color="#C1121F", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontweight="bold", fontsize=11)
    if len(x) > 0:
        pad = 0.55
        ax.set_xlim(x[0] - pad, x[-1] + pad)
    ax.set_ylabel("DAS improvement ratio (%)", fontweight="bold", labelpad=8)
    title = _title_with_eval(
        "How Much of the DAS Advantage Is Closed by DAMS?",
        eval_method,
    )
    ax.set_title(title, fontweight="bold", fontsize=12.0, pad=22)
    ax.text(
        0.5,
        1.015,
        f"Mean across {n_sims} runs",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )

    # Percent-of-gap closed labels on the bottom segment when it is tall enough.
    ymax = float(np.nanmax(gap_base)) if len(gap_base) else 1.0
    ax.set_ylim(0.0, max(20.0, ymax * 1.05 if np.isfinite(ymax) else 20.0))
    for xi, c, g in zip(x, closed, gap_base):
        if not np.isfinite(c) or not np.isfinite(g) or g <= 1e-9 or c <= 0:
            continue
        pct = 100.0 * c / g
        if c < 0.08 * max(ymax, 1.0):
            continue
        ax.text(
            xi,
            c / 2.0,
            f"{pct:.0f}%\nclosed",
            ha="center",
            va="center",
            fontsize=10.5,
            color="#1a1a1a",
            fontweight="bold",
            linespacing=1.05,
        )

    legend_handles = [
        Patch(
            facecolor=_LEGEND_GRAY,
            edgecolor="white",
            hatch="//",
            label="Gap closed by DAMS",
        ),
        Patch(
            facecolor=_LEGEND_GRAY,
            edgecolor="white",
            label="Remaining gap",
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
        title=None,
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


def plot_das_improvement_side_by_side(
    runs: list[dict],
    pairs: list[tuple[str, str]],
    *,
    eval_method: str,
    out_stem: Path,
    n_sims: int,
) -> None:
    """Side-by-side: DAS improvement vs base vs vs DAMS, per family (appendix view)."""
    families = [pair_family(b) for b, _ in pairs]
    x = np.arange(len(pairs))
    width = 0.36

    base_means, dams_means = [], []
    for base, dams in pairs:
        bm, _, _, _ = mean_ci(lifts_vs(runs, base, eval_method))
        dm, _, _, _ = mean_ci(lifts_vs(runs, dams, eval_method))
        base_means.append(bm)
        dams_means.append(dm)

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar(
        x - width / 2,
        base_means,
        width,
        color=[to_rgba(baseline_color(b)) for b, _ in pairs],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        dams_means,
        width,
        color=[to_rgba(baseline_color(d)) for _, d in pairs],
        edgecolor="white",
        linewidth=0.6,
        hatch="//",
    )
    ax.axhline(0.0, color="#C1121F", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontweight="bold", fontsize=11)
    ax.set_ylabel("DAS improvement ratio (%)", fontweight="bold", labelpad=8)
    title = _title_with_eval(
        "Ablation study: The incremental value of DAMS", eval_method
    )
    subtitle = (
        f"Mean across {n_sims} runs;\n"
        "higher = DAS better than comparator"
    )
    ax.set_title(title, fontweight="bold", fontsize=13, pad=34)
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


def plot_das_improvement(
    runs: list[dict],
    pairs: list[tuple[str, str]],
    *,
    eval_method: str,
    dataset_target: str,
    out_stem: Path,
    n_sims: int,
    layout: str = "stacked",
) -> None:
    del dataset_target  # kept for call-site compatibility
    if layout == "side_by_side":
        plot_das_improvement_side_by_side(
            runs, pairs, eval_method=eval_method, out_stem=out_stem, n_sims=n_sims
        )
    elif layout == "stacked":
        plot_das_improvement_stacked(
            runs, pairs, eval_method=eval_method, out_stem=out_stem, n_sims=n_sims
        )
    else:
        raise ValueError(f"Unknown layout={layout!r}; use 'stacked' or 'side_by_side'")


def print_summary(runs: list[dict], pairs: list[tuple[str, str]], methods: list[str]) -> None:
    n = len(runs)
    print(f"\nDAS improvement ratio: Base vs +DAMS (n={n})")
    print("-" * 72)
    for base, dams in pairs:
        print(f"\n{pair_family(base)}")
        for method in methods:
            bm, blo, bhi, _ = mean_ci(lifts_vs(runs, base, method))
            dm, dlo, dhi, _ = mean_ci(lifts_vs(runs, dams, method))
            closed = bm - dm
            pct = 100.0 * closed / bm if np.isfinite(bm) and abs(bm) > 1e-12 else float("nan")
            print(
                f"  {method:8s}  vs base={bm:+6.2f}% [{blo:+.2f},{bhi:+.2f}]  "
                f"vs DAMS={dm:+6.2f}% [{dlo:+.2f},{dhi:+.2f}]  "
                f"closed={closed:+.2f}pp ({pct:.0f}% of gap)"
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
    parser.add_argument(
        "--layout",
        type=str,
        choices=("stacked", "side_by_side"),
        default="stacked",
        help="stacked = main-text gap decomposition; side_by_side = appendix view",
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
    methods = list(args.eval_methods)
    if any(m in STZ_EVAL_METHODS for m in methods):
        n_attached = attach_stz_sidecar(runs, pkl_path)
        print(f"Attached implementation data for {n_attached}/{len(runs)} runs")
    pairs = available_pairs(runs)
    if not pairs:
        raise SystemExit("No (base, *_dams) pairs found in results.")

    dataset = str(params.get("dataset", "unknown")).replace("_", " ").title()
    target = str(params.get("target_col", "unknown")).replace("_", " ").title()
    dataset_target = f"{dataset} – {target}"
    n = len(runs)

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
            layout=args.layout,
        )


if __name__ == "__main__":
    main()
