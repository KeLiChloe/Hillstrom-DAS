"""
Average algorithm ranking across sample-fraction points, by evaluation method.

At each sample_frac x-point:
  1. score each algorithm by its mean metric over runs
  2. rank algorithms (1 = best)

Then average ranks over all x-points. Saves PNG + PDF.
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ANALYSIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _ANALYSIS_DIR.parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plot_style import baseline_color, baseline_label, to_rgba  # noqa: E402

DAST_KEY = "dast"
DAST_LABEL = "DAS"
DAST_COLOR = "#000000"
SKIP_KEYS = {"seed", "implementation", "random"}

METHOD_SPECS = (
    ("dual_dr", "Main"),
    ("dm", "Direct Method"),
    ("ipw", "IPW"),
    ("stz", "STZ"),
)


def _pkl_load(path: Path):
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(path, "rb") as f:
            return pickle.load(f)


def _merge_stz_sidecar(data: dict, main_pkl: Path) -> None:
    stz_path = main_pkl.with_name(main_pkl.stem + "_stz" + main_pkl.suffix)
    if not stz_path.is_file():
        return
    stz = _pkl_load(stz_path)
    by_seed = {
        run["seed"]: run["implementation"]
        for run in stz.get("results", [])
        if isinstance(run, dict) and "implementation" in run
    }
    for run in data.get("results", []):
        seed = run.get("seed")
        if seed in by_seed and "implementation" not in run:
            run["implementation"] = by_seed[seed]


def _is_main_sample_frac_pkl(path: Path) -> bool:
    name = path.name
    return name.startswith("sample_frac_") and "_stz" not in name and name.endswith(".pkl")


def load_sample_frac_experiments(exp_dir: Path) -> list[dict]:
    experiments: list[dict] = []
    for path in sorted(exp_dir.glob("sample_frac_*.pkl")):
        if not _is_main_sample_frac_pkl(path):
            continue
        data = _pkl_load(path)
        _merge_stz_sidecar(data, path)
        results = data.get("results") or []
        if not results:
            continue
        experiments.append(
            {
                "path": path,
                "params": data.get("params", {}),
                "x_value": float(data["params"]["sample_frac"]),
                "results": results,
            }
        )
    experiments.sort(key=lambda e: e["x_value"])
    return experiments


def discover_algorithms(results: list[dict]) -> list[str]:
    keys = set()
    for run in results:
        for k, v in run.items():
            if k in SKIP_KEYS or not isinstance(v, dict):
                continue
            keys.add(k)
    algos = sorted(keys)
    if DAST_KEY in algos:
        algos = [DAST_KEY] + [a for a in algos if a != DAST_KEY]
    return algos


def factual_mean(run: dict, algo: str) -> float:
    """STZ-style absolute score: mean(y | D == a_algo)."""
    impl = run.get("implementation")
    if impl is None:
        return float("nan")
    actions = impl.get("actions", {})
    if algo not in actions:
        return float("nan")
    D = np.asarray(impl["D"], dtype=int)
    y = np.asarray(impl["y"], dtype=float)
    a = np.asarray(actions[algo], dtype=int)
    mask = D == a
    if mask.sum() == 0:
        return float("nan")
    return float(y[mask].mean())


def mean_score_at_point(
    results: list[dict],
    algo: str,
    method: str,
) -> float:
    vals: list[float] = []
    for run in results:
        if method in ("stz", "stz_vr"):
            v = factual_mean(run, algo)
        else:
            block = run.get(algo)
            if not isinstance(block, dict) or method not in block:
                continue
            raw = block[method]
            if isinstance(raw, (int, float)):
                v = float(raw)
            else:
                continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def ranks_from_scores(scores: dict[str, float]) -> dict[str, float]:
    """Higher score → better → rank 1. Ties get average ranks."""
    items = [(a, s) for a, s in scores.items() if np.isfinite(s)]
    if not items:
        return {}
    items.sort(key=lambda t: t[1], reverse=True)
    ranks: dict[str, float] = {}
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and np.isclose(items[j + 1][1], items[i][1]):
            j += 1
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[items[k][0]] = float(avg_rank)
        i = j + 1
    return ranks


def average_ranks(
    experiments: list[dict],
    methods: list[str],
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, list[float]]]]:
    algos = discover_algorithms(experiments[0]["results"])
    # ensure algos present across experiments
    for exp in experiments[1:]:
        for a in discover_algorithms(exp["results"]):
            if a not in algos:
                algos.append(a)

    per_point: dict[str, dict[str, list[float]]] = {
        m: {a: [] for a in algos} for m in methods
    }
    mean_rank: dict[str, dict[str, float]] = {m: {} for m in methods}

    for exp in experiments:
        results = exp["results"]
        for method in methods:
            scores = {
                a: mean_score_at_point(results, a, method) for a in algos
            }
            ranks = ranks_from_scores(scores)
            for a, r in ranks.items():
                per_point[method][a].append(r)

    for method in methods:
        for a in algos:
            vals = per_point[method][a]
            mean_rank[method][a] = float(np.mean(vals)) if vals else float("nan")

    return algos, mean_rank, per_point


def algo_display(algo: str) -> str:
    if algo == DAST_KEY:
        return DAST_LABEL
    return baseline_label(algo)


def algo_color(algo: str):
    if algo == DAST_KEY:
        return to_rgba(DAST_COLOR)
    return to_rgba(baseline_color(algo))


def configure_style() -> None:
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
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 300,
        }
    )


def plot_mean_ranks(
    algos: list[str],
    mean_rank: dict[str, dict[str, float]],
    method_labels: dict[str, str],
    *,
    out_stem: Path,
    title: str,
) -> list[Path]:
    configure_style()
    methods = list(mean_rank.keys())

    # Order algorithms by Dual DR mean rank when available.
    ref = methods[0]
    order = sorted(
        algos,
        key=lambda a: (
            mean_rank[ref].get(a, 1e9),
            algo_display(a),
        ),
    )
    row_labels = [algo_display(a) for a in order]
    col_labels = [method_labels[m] for m in methods]
    mat = np.array(
        [[mean_rank[m].get(a, np.nan) for m in methods] for a in order],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    # Best rank (1) = green, worst = red.
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=1, vmax=len(order))

    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(order)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Evaluation Method")
    ax.set_ylabel("Algorithm")
    ax.set_title(title, pad=12)

    mid = 0.5 * (1 + len(order))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            txt_color = "white" if val >= mid else "black"
            weight = "bold" if order[i] == DAST_KEY else "normal"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=10,
                fontweight=weight,
            )

    for tick_label, algo in zip(ax.get_yticklabels(), order):
        if algo == DAST_KEY:
            tick_label.set_fontweight("bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Rank (1 = best)", fontweight="bold")

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in ("png", "pdf"):
        out = out_stem.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"[OK] Saved: {out}")
        saved.append(out)
    plt.close(fig)
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--exp-dir",
        required=True,
        help="Directory with sample_frac_*.pkl (+ optional *_stz.pkl sidecars).",
    )
    p.add_argument(
        "--out-stem",
        default=None,
        help="Output path without extension (default: <exp-dir>/mean_rank_sample_frac).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir).expanduser().resolve()
    experiments = load_sample_frac_experiments(exp_dir)
    if not experiments:
        raise SystemExit(f"No sample_frac pickles found under {exp_dir}")

    methods = [m for m, _ in METHOD_SPECS]
    method_labels = dict(METHOD_SPECS)
    algos, mean_rank, _ = average_ranks(experiments, methods)

    print(f"[INFO] {len(experiments)} sample_frac points, {len(algos)} algorithms")
    for method, label in METHOD_SPECS:
        ordered = sorted(algos, key=lambda a: mean_rank[method].get(a, 1e9))
        summary = ", ".join(
            f"{algo_display(a)}={mean_rank[method][a]:.2f}" for a in ordered[:5]
        )
        print(f"  {label}: top {summary}")

    out_stem = (
        Path(args.out_stem).expanduser().resolve()
        if args.out_stem
        else exp_dir / "mean_rank_sample_frac"
    )
    title = "Mean Algorithm Rank across Sample Sizes (lower is better)"
    plot_mean_ranks(
        algos,
        mean_rank,
        method_labels,
        out_stem=out_stem,
        title=title,
    )


if __name__ == "__main__":
    main()
