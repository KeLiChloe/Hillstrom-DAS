import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
import warnings

# ==========================================
# 0. 数据读取
# ==========================================
with open("exp_results/ablation/exp4.pkl", "rb") as f:
    data_exp = pickle.load(f)

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. 全局风格（UTD / Academic）
# ==========================================
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks", {'axes.grid': True})

plt.rcParams.update({
    'font.family': 'STIXGeneral',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'Times',
                   'Liberation Serif', 'DejaVu Serif'],
    'axes.labelweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'pdf.fonttype': 42,
    'figure.dpi': 300
})

# ==========================================
# 2. 数据整理
# ==========================================
if "params" in data_exp:
    params = data_exp["params"]
    print("Experiment Params:", params)
    results_list = data_exp["results"]
else:
    results_list = data_exp

n_sims = len(results_list)

target = "dast"
comparators = ["kmeans", "gmm", "clr"]

raw_data_map = {m: [] for m in comparators +
                [f"{c}_dams" for c in comparators] +
                [target]}

for run in results_list:
    raw_data_map[target].append(run[target])
    for c in comparators:
        raw_data_map[c].append(run[c])
        raw_data_map[f"{c}_dams"].append(run[f"{c}_dams"])

def lift_percent(target_vals, base_vals):
    target_vals = np.asarray(target_vals)
    base_vals = np.asarray(base_vals)
    return ((target_vals - base_vals) / base_vals) * 100

# ==========================================
# 3. 统计计算（mean, CI, significance）
# ==========================================
def get_sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

rows = []
comp_sig = {}

for comp in comparators:
    # significance：DAS vs Non-DAMS baseline
    pval = stats.ttest_rel(
        raw_data_map[target],
        raw_data_map[comp]
    )[1]
    comp_sig[comp] = get_sig_star(pval)

    for variant, base_key in [
        ("Non-DAMS", comp),
        ("DAMS", f"{comp}_dams")
    ]:
        lifts = lift_percent(
            raw_data_map[target],
            raw_data_map[base_key]
        )

        mean = lifts.mean()
        sem = stats.sem(lifts)
        ci = sem * stats.t.ppf(0.975, len(lifts) - 1)

        rows.append({
            "Comparator": comp.upper() if comp != "kmeans" else "K-Means",
            "Variant": variant,
            "Mean": mean,
            "CI": ci
        })

stats_df = pd.DataFrame(rows)

# ==========================================
# 4. 颜色 / 线型 / offset
# ==========================================
COLOR_NON_DAMS = "#000000"
COLOR_DAMS = "#C00000"
LINESTYLE_NON_DAMS = "-"
LINESTYLE_DAMS = "--"
offset = 0.15

def draw_ci_point(ax, mean, ci, y, color, linestyle, cap=0.06, lw=2, ms=9):
    x0, x1 = mean - ci, mean + ci
    ax.plot([x0, x1], [y, y], color=color, linestyle=linestyle, linewidth=lw)
    ax.plot([x0, x0], [y - cap, y + cap], color=color, linestyle=linestyle, linewidth=lw)
    ax.plot([x1, x1], [y - cap, y + cap], color=color, linestyle=linestyle, linewidth=lw)
    ax.plot(mean, y, marker="o", color=color, markersize=ms, linestyle="None")

# ==========================================
# 5. 绘图
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

comp_order = ["K-Means", "GMM", "CLR"]
comp_to_y = {c: i for i, c in enumerate(comp_order)}

ax.grid(axis="x", color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

for _, r in stats_df.iterrows():
    base_y = comp_to_y[r["Comparator"]]

    if r["Variant"] == "Non-DAMS":
        y = base_y - offset
        color = COLOR_NON_DAMS
        linestyle = LINESTYLE_NON_DAMS
    else:
        y = base_y + offset
        color = COLOR_DAMS
        linestyle = LINESTYLE_DAMS

    draw_ci_point(
        ax=ax,
        mean=r["Mean"],
        ci=r["CI"],
        y=y,
        color=color,
        linestyle=linestyle
    )

# Y 轴标签：直接带显著性
ytick_labels = [
    f"{c} ({comp_sig[c.lower()]})" if comp_sig[c.lower()] else c
    for c in comp_order
]

ax.set_yticks(range(len(comp_order)))
ax.set_yticklabels(ytick_labels, fontweight="bold", fontsize=14)
ax.tick_params(axis="y", length=0)

# 0% 基准线
ax.axvline(0, color="#183BD8", linestyle="--", linewidth=1.6, alpha=0.8)

ax.set_xlabel("Averaged DAS Improvement (%)", fontweight="bold", labelpad=12)

ax.invert_yaxis()
sns.despine(left=True, top=True, right=True)

ax.set_title(
    f"Averaged DAS Improvement (%) with 95% CI (Runs={n_sims})",
    fontweight="bold",
    fontsize=16,
    pad=20,
    y=1.12
)

ax.annotate(
    "Positive values (>0%) indicate DAS outperforms other methods",
    xy=(0.5, 1.06),
    xycoords="axes fraction",
    fontsize=12,
    fontweight="bold",
    color="#333333",
    ha="center",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", lw=0.5, alpha=0.8)
)

# ==========================================
# 6. Legend
# ==========================================
legend_handles = [
    Line2D([0], [0], color=COLOR_NON_DAMS, linestyle=LINESTYLE_NON_DAMS,
           marker="o", linewidth=2, markersize=8, label="Standard"),
    Line2D([0], [0], color=COLOR_DAMS, linestyle=LINESTYLE_DAMS,
           marker="o", linewidth=2, markersize=8, label="DAMS"),
    Line2D([0], [0], color="#183BD8", linestyle="--", linewidth=1.6,
           label="No Improvement (0%)"),
    Line2D([0], [0], color="none",
           label="*** p < 0.001\n**   p < 0.01\n*     p < 0.05"),
]

ax.legend(
    handles=legend_handles,
    loc="best",
    frameon=True,
    framealpha=0.95,
    edgecolor="#E0E0E0",
    fancybox=False,
    fontsize=11
)

plt.tight_layout()

# ==========================================
# 7. 导出
# ==========================================
plt.savefig("figures/Fig2_UTD_Style.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figures/Fig2_UTD_Style.png", dpi=300, bbox_inches="tight")
