"""
Algorithm ranking heatmaps for sample_frac sweeps.

At each sample_frac point and evaluation method:
  1. score each algorithm by its mean metric over runs
  2. rank algorithms (1 = best)

Two figures:
  (A) mean_rank_sample_frac — average ranks over sample sizes; x = eval methods
  (B) mean_rank_by_sample_size — average ranks over eval methods; x = sample size

Saves PNG + PDF for each.
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
    ("stz", "Matched-Action"),
)

_DATASET_BASE_N = {
    "hillstrom": 64_000,
    "criteo": 1_397_960,
}


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
    """Restricting-style absolute score: mean(y | D == a_algo)."""
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


def collect_pointwise_ranks(
    experiments: list[dict],
    methods: list[str],
) -> tuple[list[str], list[tuple[float, dict[str, dict[str, float]]]]]:
    """Return (algos, [(sample_frac, {method: {algo: rank}}), ...])."""
    algos = discover_algorithms(experiments[0]["results"])
    for exp in experiments[1:]:
        for a in discover_algorithms(exp["results"]):
            if a not in algos:
                algos.append(a)

    point_ranks: list[tuple[float, dict[str, dict[str, float]]]] = []
    for exp in experiments:
        results = exp["results"]
        by_method: dict[str, dict[str, float]] = {}
        for method in methods:
            scores = {a: mean_score_at_point(results, a, method) for a in algos}
            by_method[method] = ranks_from_scores(scores)
        point_ranks.append((float(exp["x_value"]), by_method))
    return algos, point_ranks


def average_ranks_over_sample_sizes(
    point_ranks: list[tuple[float, dict[str, dict[str, float]]]],
    algos: list[str],
    methods: list[str],
) -> dict[str, dict[str, float]]:
    """mean_rank[method][algo] = average rank across sample-size points."""
    mean_rank: dict[str, dict[str, float]] = {m: {} for m in methods}
    for method in methods:
        for a in algos:
            vals = [
                by_m[method][a]
                for _, by_m in point_ranks
                if method in by_m and a in by_m[method]
            ]
            mean_rank[method][a] = float(np.mean(vals)) if vals else float("nan")
    return mean_rank


def average_ranks_over_methods(
    point_ranks: list[tuple[float, dict[str, dict[str, float]]]],
    algos: list[str],
    methods: list[str],
) -> tuple[list[float], dict[str, list[float]]]:
    """
    For each sample-size point, average ranks across evaluation methods.

    Returns (x_values, mean_rank_by_algo) where mean_rank_by_algo[algo][j]
    is the method-averaged rank at x_values[j].
    """
    x_values = [x for x, _ in point_ranks]
    mean_by_algo: dict[str, list[float]] = {a: [] for a in algos}
    for _, by_m in point_ranks:
        for a in algos:
            vals = [
                by_m[m][a]
                for m in methods
                if m in by_m and a in by_m[m] and np.isfinite(by_m[m][a])
            ]
            mean_by_algo[a].append(float(np.mean(vals)) if vals else float("nan"))
    return x_values, mean_by_algo


def display_sample_size_k(sample_frac: float, dataset: str | None) -> float:
    base = _DATASET_BASE_N.get(str(dataset or "").lower())
    if base is not None:
        return (base * float(sample_frac)) / 1000.0
    return 100.0 * float(sample_frac)


def _format_n_tick(value: float, _pos=None) -> str:
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}"
    return f"{value:.1f}"


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


def _order_algos_by_ref(
    algos: list[str],
    ref_scores: dict[str, float],
) -> list[str]:
    return sorted(
        algos,
        key=lambda a: (ref_scores.get(a, 1e9), algo_display(a)),
    )


def _draw_rank_heatmap(
    mat: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    order: list[str],
    xlabel: str,
    title: str,
    rotate_xticks: bool,
    out_stem: Path,
) -> list[Path]:
    configure_style()
    n_cols = mat.shape[1]
    fig_w = max(7.2, 0.55 * n_cols + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=1, vmax=len(order))

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(len(order)))
    if rotate_xticks:
        ax.set_xticklabels(col_labels, rotation=40, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Algorithm")
    ax.set_title(title, pad=12)

    mid = 0.5 * (1 + len(order))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            # Pale yellow / light cells need black text (luminance, not rank mid).
            rgba = cmap((float(val) - 1.0) / max(len(order) - 1, 1))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = "black" if lum >= 0.55 else "white"
            weight = "bold" if order[i] == DAST_KEY else "normal"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=9 if n_cols > 6 else 10,
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


def plot_mean_ranks_by_method(
    algos: list[str],
    mean_rank: dict[str, dict[str, float]],
    method_labels: dict[str, str],
    *,
    out_stem: Path,
    title: str,
) -> list[Path]:
    """Figure A: x = evaluation methods; ranks averaged over sample sizes."""
    methods = list(mean_rank.keys())
    order = _order_algos_by_ref(algos, mean_rank[methods[0]])
    mat = np.array(
        [[mean_rank[m].get(a, np.nan) for m in methods] for a in order],
        dtype=float,
    )
    return _draw_rank_heatmap(
        mat,
        row_labels=[algo_display(a) for a in order],
        col_labels=[method_labels[m] for m in methods],
        order=order,
        xlabel="Evaluation Method",
        title=title,
        rotate_xticks=False,
        out_stem=out_stem,
    )


def plot_mean_ranks_by_sample_size(
    algos: list[str],
    x_values: list[float],
    mean_by_algo: dict[str, list[float]],
    *,
    dataset: str | None,
    out_stem: Path,
    title: str,
) -> list[Path]:
    """Figure B: x = sample size; ranks averaged over evaluation methods."""
    # Order by mean rank at the largest sample size (problematic regime).
    last_scores = {
        a: mean_by_algo[a][-1] if mean_by_algo[a] else 1e9 for a in algos
    }
    order = _order_algos_by_ref(algos, last_scores)
    mat = np.array(
        [[mean_by_algo[a][j] for j in range(len(x_values))] for a in order],
        dtype=float,
    )
    xs_disp = [display_sample_size_k(x, dataset) for x in x_values]
    col_labels = [_format_n_tick(v) for v in xs_disp]
    xlabel = (
        r"Sample size $N$ ($\times 10^{3}$)"
        if dataset and str(dataset).lower() in _DATASET_BASE_N
        else "Sample fraction (%)"
    )
    return _draw_rank_heatmap(
        mat,
        row_labels=[algo_display(a) for a in order],
        col_labels=col_labels,
        order=order,
        xlabel=xlabel,
        title=title,
        rotate_xticks=True,
        out_stem=out_stem,
    )


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
        help=(
            "Stem for figure A (default: <exp-dir>/mean_rank_sample_frac). "
            "Figure B is written as <stem>_by_sample_size."
        ),
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
    algos, point_ranks = collect_pointwise_ranks(experiments, methods)
    mean_rank = average_ranks_over_sample_sizes(point_ranks, algos, methods)
    x_values, mean_by_algo = average_ranks_over_methods(point_ranks, algos, methods)
    dataset = experiments[0]["params"].get("dataset")

    print(f"[INFO] {len(experiments)} sample_frac points, {len(algos)} algorithms")
    print("[INFO] Figure A — mean rank over sample sizes (by eval method):")
    for method, label in METHOD_SPECS:
        ordered = sorted(algos, key=lambda a: mean_rank[method].get(a, 1e9))
        summary = ", ".join(
            f"{algo_display(a)}={mean_rank[method][a]:.2f}" for a in ordered[:5]
        )
        print(f"  {label}: top {summary}")

    print("[INFO] Figure B — mean rank over eval methods (by sample size):")
    for j, x in enumerate(x_values):
        scores = {a: mean_by_algo[a][j] for a in algos}
        ordered = sorted(algos, key=lambda a: scores.get(a, 1e9))
        n_k = display_sample_size_k(x, dataset)
        summary = ", ".join(
            f"{algo_display(a)}={scores[a]:.2f}" for a in ordered[:5]
        )
        print(f"  N={_format_n_tick(n_k)}k: top {summary}")

    out_a = (
        Path(args.out_stem).expanduser().resolve()
        if args.out_stem
        else exp_dir / "mean_rank_sample_frac"
    )
    out_b = out_a.with_name(out_a.name + "_by_sample_size")

    plot_mean_ranks_by_method(
        algos,
        mean_rank,
        method_labels,
        out_stem=out_a,
        title="Mean Algorithm Rank (lower is better)",
    )
    plot_mean_ranks_by_sample_size(
        algos,
        x_values,
        mean_by_algo,
        dataset=str(dataset) if dataset is not None else None,
        out_stem=out_b,
        title="Mean Algorithm Rank across Evaluation Methods (lower is better)",
    )


if __name__ == "__main__":
    main()
