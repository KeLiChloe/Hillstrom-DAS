import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 0. 读取数据
# ==========================================
with open("exp_results/main/exp2.pkl", "rb") as f:
    data_exp = pickle.load(f)

if isinstance(data_exp, dict) and "results" in data_exp:
    params = data_exp.get("params", {})
    print("Experiment Params:", params)
    results_list = data_exp["results"]
else:
    results_list = data_exp

n_sims = len(results_list)
print("Number of runs:", n_sims)

# ==========================================
# 1. 样式设置
# ==========================================
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks", {'axes.grid': True})

plt.rcParams.update({
    'font.family': 'STIXGeneral',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.pad': 8,
    'ytick.major.pad': 8,
    'pdf.fonttype': 42,
    'figure.dpi': 300
})

# ==========================================
# 2. 定义方法、颜色、标签
# ==========================================
methods = [
    "all_0", "all_1", "all_2",
    "random",
    "kmeans", "gmm", "clr", "mst",
    "causal_forest",
    "t_learner", "s_learner", "x_learner", "dr_learner",
    "dast"
]

label_map = {
    "all_0": "All Action=0",
    "all_1": "All Action=1",
    "all_2": "All Action=2",
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
    "dast": "DAST",
}

palette = {
    "All Action=0": "#55A86899",
    "All Action=1": "#4C72B099",
    "All Action=2": "#8172B299",
    "Random": "#C44E5299",
    "K-Means": "#CCB97499",
    "GMM": "#64B5CD99",
    "CLR": "#8C613C99",
    "MST": "#93786099",
    "Causal Forest": "#1F77B499",
    "T-learner": "#FF923499",
    "S-learner": "#2CA02C99",
    "X-learner": "#F5474799",
    "DR-learner": "#9467BD99",
    "DAST": "#EF660A99",
}

# 可选：排序顺序
constant_order = [
    "All Action=1",
    "All Action=2",
    "All Action=0",
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
    "DAST",
]

# ==========================================
# 3. 指定 evaluation method
# ==========================================
eval_method = "dual_dr"  # 你也可以换成 "dr" 或 "ipw"

# ==========================================
# 4. 构造 DataFrame
# ==========================================
records = []

for i, run in enumerate(results_list):
    for m in methods:
        if m not in run:
            continue
        score = run[m].get(eval_method, np.nan)
        if not np.isfinite(score):
            continue
        records.append({
            "Run": i,
            "Method": m,
            "Method_Label": label_map[m],
            "Value": float(score)
        })

df = pd.DataFrame(records)

# ==========================================
# 5. 绘图
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(
    x='Method_Label', y='Value', data=df, order=constant_order,
    palette=palette, showfliers=False, width=0.55, linewidth=1.2, ax=ax,
    showmeans=True,
    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 6},
    boxprops=dict(edgecolor='black', linewidth=1.2, alpha=0.9),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2),
    medianprops=dict(color="#D02613", linewidth=1.5)
)

sns.stripplot(
    x='Method_Label', y='Value', data=df, order=constant_order,
    color='#333333', alpha=0.4, size=4, jitter=False, ax=ax
)

ax.set_ylabel('Estimated Revenue per Customer ($)', fontweight='bold', labelpad=12)
ax.set_xlabel('Methods', labelpad=14, fontsize=16)

y_min, y_max = df["Value"].min(), df["Value"].max()
margin = 0.05 * (y_max - y_min if y_max != y_min else 1)
ax.set_ylim(y_min - margin, y_max + margin)

# 在绘图代码后，手动设置 tick labels（强制覆盖）
ax.set_xticks(range(len(constant_order)))  # 明确 tick 位置
ax.set_xticklabels(
    constant_order,
    rotation=30,
    ha='right',
    fontsize=14,
)


sns.despine(trim=True, offset=10)

ax.set_title(
    f'Performance Comparison ({eval_method.upper()}) across {n_sims} Runs',
    fontweight='bold', fontsize=16, y=1.12
)

legend_elements = [
    Patch(facecolor='#E0E0E0', edgecolor='black', linewidth=1.2, label='Box: IQR (25%–75%)'),
    Line2D([0], [0], color="#D02613", linewidth=1.5, label='Median'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
           markeredgecolor='black', markersize=6, linestyle='None', label='Mean'),
    Line2D([0], [0], color='black', linewidth=1.2, marker='_', markersize=15,
           markeredgewidth=1.2, markerfacecolor='black', label='Whiskers'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333',
           alpha=0.4, markersize=5, linestyle='None', label='Single run')
]

leg = ax.legend(
    handles=legend_elements, loc='best', frameon=True, framealpha=0.95,
    edgecolor='#B0B0B0', fancybox=False, fontsize=10,
    borderpad=0.8, labelspacing=0.6, handlelength=1.5
)
leg._legend_box.align = "left"

plt.tight_layout()

# ==========================================
# 6. 保存图像（确保 rotation 生效）
# ==========================================

# 在 savefig 前强制 draw + 设置 label rotation（这是关键）
plt.draw()
for label in ax.get_xticklabels():
    label.set_rotation(30)
    label.set_ha('right')
    label.set_fontsize(14)

# 不要使用 bbox_inches='tight'，它会裁剪掉旋转的 label
out_path = f'figures/Fig1_Boxplot_Distribution_{eval_method.upper()}.pdf'
plt.savefig(out_path, dpi=300)  # ❌ 不加 bbox_inches
plt.savefig(out_path.replace(".pdf", ".png"), dpi=300)
plt.close()
print(f"[OK] Saved: {out_path}")

