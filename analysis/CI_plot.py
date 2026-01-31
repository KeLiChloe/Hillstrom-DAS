import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D

# ==========================================
# 0. I/O
# ==========================================
PKL_PATH = "exp_results_hillstrom/main/exp3.pkl"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

with open(PKL_PATH, "rb") as f:
    data_exp = pickle.load(f)

warnings.simplefilter(action="ignore", category=FutureWarning)

# ==========================================
# 1. Plot style
# ==========================================
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks", {'axes.grid': True})

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'axes.labelweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'pdf.fonttype': 42,
    'figure.dpi': 300,
})

# ==========================================
# 2. Load results
# ==========================================
if isinstance(data_exp, dict) and "results" in data_exp:
    params = data_exp.get("params", {})
    print("Experiment Params:", params)
    results_list = data_exp["results"]
else:
    results_list = data_exp

n_sims = len(results_list)
print("Number of runs:", n_sims)

target = "dast"
eval_methods = ["dual_dr", "dr", "ipw"]  # 画三张图

# 你希望出现的 comparator（如果 pkl 里没有，会自动跳过）
requested_baselines = [
    # "all_0", "all_1", "all_2",
    "random",
    "kmeans", "gmm", "clr", "mst",
    "causal_forest",
    "t_learner", "s_learner", "x_learner", 
    "dr_learner",
]

# ==========================================
# 3. Helpers: safely get value
# ==========================================
def safe_get_value(run: dict, algo: str, ev: str):
    """
    New structure:
      run[algo] is dict like {"dual_dr": float, "dr": float, "ipw": float, "time":..., ...}
    Returns float or np.nan
    """
    x = run[algo]
    v = x.get(ev, np.nan)
    v = float(v)
    return v

def pretty_name(k: str) -> str:
    return "vs. " + k.replace("_", " ").title()

label_map = {
    "all_0": "vs. All Action=0",
    "all_1": "vs. All Action=1",
    "all_2": "vs. All Action=2",
    "random": "vs. Random",
    "kmeans": "vs. K-Means",
    "gmm": "vs. GMM",
    "clr": "vs. CLR",
    "mst": "vs. MST",
    "causal_forest": "vs. Causal Forest",
    "t_learner": "vs. T-learner",
    "s_learner": "vs. S-learner",
    "x_learner": "vs. X-learner",
    "dr_learner": "vs. DR-learner",
}

preferred_order = [
    "vs. All Action=1",
    "vs. All Action=2",
    "vs. All Action=0",
    "vs. Random",
    "vs. K-Means",
    "vs. GMM",
    "vs. CLR",
    "vs. MST",
    "vs. Causal Forest",
    "vs. DR-learner",
    "vs. S-learner",
    "vs. T-learner",
    "vs. X-learner",
    
]

palette = {
    "vs. All Action=0": "#2CA02C99",
    "vs. All Action=1": "#4C72B099",
    "vs. All Action=2": "#8172B299",
    "vs. Random": "#8C613C99",
    "vs. K-Means": "#CCB97499",
    "vs. GMM": "#64B5CD99",
    "vs. CLR": "#9467BD99",
    "vs. MST": "#93786099",
    "vs. Causal Forest": "#1F77B499",
    "vs. T-learner": "#FF7F0E99",
    "vs. S-learner": "#55A86899",
    "vs. X-learner": "#4EBEC499",
    "vs. DR-learner": "#D6272899",
}

def get_sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return None

# ==========================================
# 4. Discover which algos exist
# ==========================================
all_keys = set()
for run in results_list:
    if isinstance(run, dict):
        all_keys |= set(run.keys())

if target not in all_keys:
    raise KeyError(f"Target '{target}' not found. Example keys: {sorted(list(all_keys))[:30]}")

baselines = [b for b in requested_baselines if b in all_keys and b != target]
print("Baselines actually plotted:", baselines)

# ==========================================
# 5. Build & plot for each eval_method
# ==========================================
for EV in eval_methods:
    # ------------------------------------------
    # 5.1 Build df of lifts for this EV
    # ------------------------------------------
    records = []
    pair_counts = {b: 0 for b in baselines}

    for i, run in enumerate(results_list):
        vt = safe_get_value(run, target, EV)
        if not np.isfinite(vt):
            continue

        for b in baselines:
            vb = safe_get_value(run, b, EV)
            if not np.isfinite(vb):
                continue

            # lift definition: (target - baseline)/baseline * 100
            # 防止 baseline=0
            if abs(vb) < 1e-12:
                continue

            lift = (vt - vb) / vb * 100.0 + 1.
            records.append({"Run": i, "Baseline": b, "Lift": float(lift)})
            pair_counts[b] += 1

    df = pd.DataFrame(records)
    if df.empty:
        print(f"[WARN] No valid pairs for eval={EV}. Skip plotting.")
        continue

    df["Baseline_Label"] = df["Baseline"].apply(lambda b: label_map.get(b, pretty_name(b)))

    # ------------------------------------------
    # 5.2 Paired t-test: align by Run for each baseline
    # ------------------------------------------
    p_values = {}
    for b in baselines:
        sub = df[df["Baseline"] == b].sort_values("Run")
        if sub.empty:
            continue

        runs = sub["Run"].values
        t_vals, b_vals = [], []
        for r in runs:
            run = results_list[int(r)]
            vt = safe_get_value(run, target, EV)
            vb = safe_get_value(run, b, EV)
            if np.isfinite(vt) and np.isfinite(vb):
                t_vals.append(vt)
                b_vals.append(vb)

        label = label_map.get(b, pretty_name(b))
        if len(t_vals) < 2:
            p_values[label] = np.nan
        else:
            p_values[label] = stats.ttest_rel(t_vals, b_vals).pvalue

    # ------------------------------------------
    # 5.3 Summary stats (mean lift, 95% CI)
    # ------------------------------------------
    all_labels = df["Baseline_Label"].unique().tolist()
    ordered_labels = [x for x in preferred_order if x in all_labels]
    ordered_labels += [x for x in sorted(all_labels) if x not in set(ordered_labels)]

    summary_stats = []
    for label in ordered_labels:
        subset = df[df["Baseline_Label"] == label]["Lift"]
        if subset.empty:
            continue
        mean = float(subset.mean())
        sem = float(subset.sem()) if len(subset) > 1 else 0.0
        ci = float(sem * stats.t.ppf(0.975, len(subset) - 1)) if len(subset) > 1 else 0.0
        p = p_values.get(label, np.nan)
        p_star = get_sig_star(p) if np.isfinite(p) else None
        summary_stats.append({"Label": label, "Mean": mean, "CI": ci, "P_Star": p_star, "N": int(len(subset))})

    stats_df = pd.DataFrame(summary_stats)
    if stats_df.empty:
        print(f"[WARN] stats_df empty for eval={EV}. Skip plotting.")
        continue

    # ------------------------------------------
    # 5.4 Plot
    # ------------------------------------------
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    y_pos = np.arange(len(stats_df))

    ax.grid(axis="x", color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    for i, row in stats_df.iterrows():
        color = palette.get(row["Label"], "#333333")
        ax.errorbar(
            x=row["Mean"],
            y=i,
            xerr=row["CI"],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            elinewidth=2,
            markersize=6,
        )

    ytick_labels = [
        f"{row['Label']}"
        + (f" ({row['P_Star']})" if row["P_Star"] is not None else "")
        + (f"  [n={row['N']}]" if row["N"] < n_sims else "")
        for _, row in stats_df.iterrows()
    ]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ytick_labels, fontweight="bold", fontsize=13)
    ax.tick_params(axis="y", length=0)

    ax.axvline(0, color="#E40606", linestyle="--", linewidth=1.6, alpha=0.8)

    ax.set_xlabel(
        "Averaged DAS Improvement (%) on Revenue Over Comparators",
        fontweight="bold",
        labelpad=12,
    )

    ax.invert_yaxis()
    sns.despine(left=True, top=True, right=True)

    title_map = {
        "dual_dr": "Dual DR OPE",
        "dr": "DR OPE",
        "ipw": "IPW OPE",
    }

    ax.set_title(
        f"Averaged DAS Improvement (%) — {title_map.get(EV, EV)} (Runs={n_sims})",
        fontweight="bold",
        pad=18,
        fontsize=16,
        y=1.08,
    )

    ax.annotate(
        "Positive values (>0%) indicate DAS outperforms comparators",
        xy=(0.5, 1.03),
        xycoords="axes fraction",
        fontsize=12,
        fontweight="bold",
        color="#333333",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", lw=0.5, alpha=0.8),
    )

    legend_handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="-", linewidth=2, markersize=8, label="Mean ± 95% CI"),
        Line2D([0], [0], color="none", label="*** p < 0.001\n**   p < 0.01\n*     p < 0.05"),
        Line2D([0], [0], color="#E40606", linestyle="--", linewidth=1.6, label="No Improvement (0%)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#E0E0E0",
        fancybox=False,
        fontsize=11,
        borderpad=1,
    )

    plt.tight_layout()

    out_pdf = os.path.join(FIG_DIR, f"Fig2_UTD_Style_{EV}.pdf")
    out_png = os.path.join(FIG_DIR, f"Fig2_UTD_Style_{EV}.png")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved ({EV}): {out_pdf}")
    print(f"[OK] Saved ({EV}): {out_png}")
