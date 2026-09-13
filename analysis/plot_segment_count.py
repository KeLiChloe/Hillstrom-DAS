"""Plot mean selected segment count (best_M) per algorithm.

Reads main experiment pickles under --exp-dir (skips *_stz.pkl).
Algorithms without best_M (meta-learners, causal_forest, ...) are skipped.

Examples:
  python analysis/plot_segment_count.py \\
    exp_aug/hillstrom/conversion/lightgbm_reg/sample_frac_with_fixed_0.40_pilot_frac

  python analysis/plot_segment_count.py exp_aug/ablation/hillstrom/conversion/lightgbm_reg
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
import numpy as np
import seaborn as sns
from scipy import stats

_ANALYSIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ANALYSIS_DIR.parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from plot_style import (  # noqa: E402
    PREFERRED_ORDER,
    baseline_color,
    baseline_label,
    to_rgba,
)

DAST_ALGO = "dast"
DAST_COLOR = "#000000"

# Segment-count display order / labels for DAMS ablation variants.
_SEGMENT_LABEL = {
    "dast": "DAST",
    "kmeans": "K-Means",
    "kmeans_dams": "K-Means + DAMS",
    "gmm": "GMM",
    "gmm_dams": "GMM + DAMS",
    "clr": "CLR",
    "clr_dams": "CLR + DAMS",
    "mst": "MST",
}

_SEGMENT_ORDER = [
    "DAST",
    "K-Means",
    "K-Means + DAMS",
    "GMM",
    "GMM + DAMS",
    "CLR",
    "CLR + DAMS",
    "MST",
]


def _pkl_load(path: Path):
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as f:
            return pickle.load(f)


def _is_main_pkl(path: Path) -> bool:
    return path.suffix == ".pkl" and not path.stem.endswith("_stz")


def algo_label(algo: str) -> str:
    if algo in _SEGMENT_LABEL:
        return _SEGMENT_LABEL[algo]
    if algo == DAST_ALGO:
        return "DAST"
    return baseline_label(algo)


def algo_color(algo: str):
    if algo == DAST_ALGO:
        return to_rgba(DAST_COLOR)
    # DAMS variants: same hue as the base method
    base = algo.replace("_dams", "") if algo.endswith("_dams") else algo
    return to_rgba(baseline_color(base))


def discover_pkls(exp_dir: Path) -> list[Path]:
    for pattern in (
        "sample_frac_*.pkl",
        "pilot_frac_*.pkl",
        "treatment_cost_*.pkl",
    ):
        found = sorted(p for p in exp_dir.glob(pattern) if _is_main_pkl(p))
        if found:
            return found
    found = sorted(p for p in exp_dir.glob("*.pkl") if _is_main_pkl(p))
    if not found:
        raise FileNotFoundError(f"No main pickle files in {exp_dir}")
    return found


def infer_x_value(path: Path, params: dict) -> float | None:
    for key in ("sample_frac", "pilot_frac", "treatment_cost"):
        if key in params and params[key] is not None:
            return float(params[key])
    for key in ("sample_frac", "pilot_frac", "treatment_cost"):
        m = re.search(rf"{key}_(\d+(?:\.\d+)?)", path.stem)
        if m:
            raw = float(m.group(1))
            return raw if key == "treatment_cost" else raw / 100.0
        m = re.search(rf"{key}_(\d+p\d+)", path.stem)
        if m and key == "treatment_cost":
            return float(m.group(1).replace("p", "."))
    return None


def infer_x_label(params_list: list[dict], paths: list[Path]) -> str:
    for params in params_list:
        for key, label in (
            ("sample_frac", "sample_frac"),
            ("pilot_frac", "pilot_frac"),
            ("treatment_cost", "treatment_cost"),
        ):
            if key in params:
                return label
    names = " ".join(p.name for p in paths)
    if "treatment_cost" in names:
        return "treatment_cost"
    if "pilot_frac" in names:
        return "pilot_frac"
    if "sample_frac" in names:
        return "sample_frac"
    return "x"


def collect_best_m(
    pkls: list[Path],
    *,
    min_sims: int,
    algo_filter: set[str] | None,
) -> tuple[list[dict], list[str]]:
    """Return (rows, algo_keys). Each row: x, algo, best_M, path."""
    rows: list[dict] = []
    algos: set[str] = set()
    for path in pkls:
        data = _pkl_load(path)
        if not isinstance(data, dict) or "results" not in data:
            warnings.warn(f"Skipping {path.name}: unexpected format")
            continue
        params = data.get("params", {})
        results = data["results"]
        if len(results) < min_sims:
            warnings.warn(
                f"Skipping {path.name}: {len(results)} sims < min_sims={min_sims}"
            )
            continue
        x = infer_x_value(path, params)
        for run in results:
            if not isinstance(run, dict):
                continue
            for algo, payload in run.items():
                if algo == "seed" or not isinstance(payload, dict):
                    continue
                if "best_M" not in payload:
                    continue
                if algo_filter is not None and algo not in algo_filter:
                    continue
                m = payload["best_M"]
                if m is None or (isinstance(m, float) and not np.isfinite(m)):
                    continue
                rows.append(
                    {
                        "x": x,
                        "algo": algo,
                        "best_M": float(m),
                        "path": path.name,
                        "params": params,
                    }
                )
                algos.add(algo)
    if not rows:
        raise RuntimeError(
            "No best_M entries found. Algorithms with segment counts "
            "(dast/kmeans/gmm/clr/mst/*_dams) may be missing from these pickles."
        )
    ordered = _order_algos(sorted(algos))
    return rows, ordered


def _order_algos(algos: list[str]) -> list[str]:
    label_rank = {lab: i for i, lab in enumerate(_SEGMENT_ORDER)}
    pref_rank = {lab: i for i, lab in enumerate(PREFERRED_ORDER)}

    def key(a: str):
        lab = algo_label(a)
        if lab in label_rank:
            return (0, label_rank[lab], a)
        if lab in pref_rank:
            return (1, pref_rank[lab], a)
        return (2, lab, a)

    return sorted(algos, key=key)


def _mean_ci(vals: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(vals.mean())
    if n == 1:
        return mean, mean, mean
    se = float(stats.sem(vals, ddof=1))
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    half = tcrit * se
    return mean, mean - half, mean + half


def summarize(rows: list[dict], algos: list[str]) -> list[dict]:
    """Aggregate mean / CI of best_M by (x, algo)."""
    from collections import defaultdict

    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        buckets[(r["x"], r["algo"])].append(r["best_M"])

    out = []
    for (x, algo), vals in sorted(buckets.items(), key=lambda t: (t[0][0] is None, t[0][0], t[0][1])):
        if algo not in algos:
            continue
        mean, lo, hi = _mean_ci(np.asarray(vals))
        out.append(
            {
                "x": x,
                "algo": algo,
                "label": algo_label(algo),
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "n": len(vals),
            }
        )
    return out


def plot_bars(summary: list[dict], *, title: str, out_stem: Path) -> None:
    """Single-point (or pooled) bar chart of mean best_M."""
    # If multiple x, pool is wrong — caller should use lines. Here assume one x
    # or caller passed already-filtered summary with unique algos.
    by_algo: dict[str, dict] = {}
    for row in summary:
        by_algo[row["algo"]] = row
    algos = _order_algos(list(by_algo.keys()))
    means = [by_algo[a]["mean"] for a in algos]
    los = [by_algo[a]["lo"] for a in algos]
    his = [by_algo[a]["hi"] for a in algos]
    labels = [algo_label(a) for a in algos]
    colors = [algo_color(a) for a in algos]
    yerr = np.vstack(
        [
            np.asarray(means) - np.asarray(los),
            np.asarray(his) - np.asarray(means),
        ]
    )

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(algos) + 2), 4.2))
    x_pos = np.arange(len(algos))
    ax.bar(x_pos, means, color=colors, edgecolor="white", linewidth=0.6, yerr=yerr, capsize=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean # segments (best_M)")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = out_stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def plot_lines(
    summary: list[dict],
    algos: list[str],
    *,
    x_label: str,
    title: str,
    out_stem: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for algo in algos:
        rows = sorted(
            (r for r in summary if r["algo"] == algo and r["x"] is not None),
            key=lambda r: r["x"],
        )
        if not rows:
            continue
        xs = np.asarray([r["x"] for r in rows], dtype=float)
        ys = np.asarray([r["mean"] for r in rows], dtype=float)
        lo = np.asarray([r["lo"] for r in rows], dtype=float)
        hi = np.asarray([r["hi"] for r in rows], dtype=float)
        color = algo_color(algo)
        ls = "--" if algo.endswith("_dams") else "-"
        marker = "s" if algo.endswith("_dams") else "o"
        ax.plot(
            xs,
            ys,
            label=algo_label(algo),
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=5,
            linewidth=1.8,
        )
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean # segments (best_M)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=True, fontsize=8, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = out_stem.with_suffix(f".{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize mean selected segment count (best_M) per algorithm."
    )
    parser.add_argument(
        "exp_dir",
        type=str,
        nargs="?",
        default=None,
        help="Experiment directory containing main *.pkl files",
    )
    parser.add_argument(
        "--exp-dir",
        "--exp_dir",
        dest="exp_dir_opt",
        type=str,
        default=None,
        help="Same as positional exp_dir",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: same as exp-dir)",
    )
    parser.add_argument(
        "--min-sims",
        type=int,
        default=1,
        help="Skip pickles with fewer than this many completed sims (default: 1)",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=None,
        help="Optional subset of algorithm keys (e.g. dast kmeans kmeans_dams)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="segment_count",
        help="Output filename stem prefix (default: segment_count)",
    )
    args = parser.parse_args()

    exp_dir_arg = args.exp_dir_opt or args.exp_dir
    if not exp_dir_arg:
        raise SystemExit("error: exp_dir required (positional or --exp-dir)")
    exp_dir = Path(exp_dir_arg).expanduser().resolve()
    if not exp_dir.is_dir():
        raise SystemExit(f"error: not a directory: {exp_dir}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else exp_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    algo_filter = set(args.algos) if args.algos else None
    pkls = discover_pkls(exp_dir)
    rows, algos = collect_best_m(
        pkls, min_sims=args.min_sims, algo_filter=algo_filter
    )
    summary = summarize(rows, algos)
    x_vals = {r["x"] for r in summary}
    x_label = infer_x_label([r["params"] for r in rows], pkls)

    dataset = rows[0]["params"].get("dataset", "")
    target = rows[0]["params"].get("target_col", "")
    title = f"Mean # segments — {dataset} / {target}".strip(" —/")

    print(
        f"Loaded {len(pkls)} pickle(s), {len(rows)} (sim,algo) best_M values; "
        f"algos={algos}"
    )
    for row in summary:
        print(
            f"  x={row['x']!s:>8}  {row['label']:<18}  "
            f"mean={row['mean']:.2f}  CI=[{row['lo']:.2f},{row['hi']:.2f}]  n={row['n']}"
        )

    out_stem = out_dir / args.prefix
    if len(x_vals) == 1 or (None in x_vals and len(x_vals) == 1):
        plot_bars(summary, title=title, out_stem=out_stem)
    elif None in x_vals:
        # Mixed / unknown x: bar of pooled means per algo
        warnings.warn("Some pickles lack a sweep x; plotting pooled bar chart.")
        pooled = []
        from collections import defaultdict

        by_algo = defaultdict(list)
        for r in rows:
            by_algo[r["algo"]].append(r["best_M"])
        for algo in algos:
            mean, lo, hi = _mean_ci(np.asarray(by_algo[algo]))
            pooled.append(
                {
                    "x": None,
                    "algo": algo,
                    "label": algo_label(algo),
                    "mean": mean,
                    "lo": lo,
                    "hi": hi,
                    "n": len(by_algo[algo]),
                }
            )
        plot_bars(pooled, title=title + " (pooled)", out_stem=out_stem)
    else:
        plot_lines(
            summary,
            algos,
            x_label=x_label,
            title=title,
            out_stem=out_stem,
        )


if __name__ == "__main__":
    main()
