import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import warnings
from matplotlib.patches import Patch

warnings.simplefilter(action='ignore', category=FutureWarning)

with open("exp_results/main/exp0.pkl", "rb") as f:
    data_exp = pickle.load(f)
    
# ==========================================
# 1. 样式设置（保持与原一致）
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


# 获取 simulation results
if "params" in data_exp:
    params = data_exp["params"]
    print("Experiment Params:", params)
    results_list = data_exp["results"]
else:
    results_list = data_exp

n_sims = len(results_list)

# ==========================================
# 3. 定义方法、标签、颜色（颜色绑定用字典！）
# ==========================================

# 方法内部名称（run 存的 key）
methods = [
    'all_control', 'random', 'kmeans', 
    'gmm', 'clr', 'mst', 'policytree', 'dast'
]

# 显示用标签
label_map = {
    # 'all_treat': 'All Treat',
    'all_control': 'All Control',
    'random': 'Random',
    'kmeans': 'K-Means',
    'gmm': 'GMM',
    'clr': 'CLR',
    'mst': 'MST',
    'policytree': 'Policytree',
    'dast': 'DAST'
}

# 🎨 用字典绑定颜色，不依赖顺序
palette = {
    # 'All Treat':   "#4C72B099",
    'All Control': "#55A86899",
    'Random':      "#C44E5299",
    'K-Means':     "#8172B299",
    'GMM':         "#CCB97499",
    'CLR':         "#64B5CD99",
    'MST':         "#8C613C99",
    'Policytree':  "#93786099",
    'DAST':        "#FF7F0E99",
}


# 👉 画图顺序：你可以随便改，不影响颜色
constant_order = [
    # 'All Treat',
    'All Control',
    'Random',
    'K-Means',
    'GMM',
    'CLR',
    'MST',
    # 'Policytree',
    'DAST',
]

# ==========================================
# 4. 整理数据
# ==========================================
records = []

for i, run in enumerate(results_list):
    for m in methods:
        if m not in run:
            continue
        records.append({
            'Run': i,
            'Method': m,
            'Method_Label': label_map[m],
            'Value': run[m]
        })

df = pd.DataFrame(records)
df['Value'] = df['Value']*100
# ==========================================
# 按照 median 动态排序方法顺序
# ==========================================
median_order = (
    df.groupby("Method_Label")["Value"]
      .median()
      .sort_values(ascending=True)
      .index
      .tolist()
)
constant_order = median_order  # 直接改写你的 constant_order


# ==========================================
# 5. 绘图
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

# --- 网格 ---
ax.grid(axis='y', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

# --- 箱线图 ---
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

# --- 单次 run 的离散点 ---
sns.stripplot(
    x='Method_Label', y='Value', data=df, order=constant_order,
    color='#333333', alpha=0.4, size=4, jitter=False, ax=ax
)

# --- 坐标轴标签 ---
ax.set_ylabel('Expected Spend per Customer ($) at Implementation Stage', fontweight='bold', labelpad=12)
ax.set_xlabel('Methods', fontweight='bold', labelpad=14,fontsize=16)

# --- Y 轴自适应边距 ---
y_min = df['Value'].min()
y_max = df['Value'].max()
margin = 0.05 * (y_max - y_min if y_max != y_min else 1)
ax.set_ylim(y_min - margin, y_max + margin)

# --- X 轴标签格式 ---
ax.set_xticks(range(len(constant_order)))
ax.set_xticklabels(constant_order, rotation=30, ha='right', fontsize=14, fontweight='bold')

sns.despine(trim=True, offset=10)

# --- 标题 ---
ax.set_title(
    f'Distribution of Performance Metric (Expected Spend per Customer ($)) across {n_sims} Runs',
    fontweight='bold', fontsize=16, y=1.12
)

# --- 图例 ---
legend_elements = [
    Patch(facecolor='#E0E0E0', edgecolor='black', linewidth=1.2,
          label='Box: IQR (25%–75%)'),
    Line2D([0], [0], color="#D02613", linewidth=1.5, label='Median'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='white',
           markeredgecolor='black', markersize=6, linestyle='None',
           label='Mean'),
    Line2D([0], [0], color='black', linewidth=1.2,
           marker='_', markersize=15, markeredgewidth=1.2,
           markerfacecolor='black',
           label=r'Whiskers'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333',
           alpha=0.4, markersize=5, linestyle='None',
           label='Single run')
]

leg = ax.legend(
    handles=legend_elements, loc='best', frameon=True, framealpha=0.95,
    edgecolor='#B0B0B0', fancybox=False, fontsize=10,
    borderpad=0.8, labelspacing=0.6, handlelength=1.5
)
leg._legend_box.align = "left"

plt.tight_layout()

# ==========================================
# 6. 导出图像
# ==========================================
os.makedirs("figures", exist_ok=True)
plt.savefig('figures/Fig1_Boxplot_Method_Distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/Fig1_Boxplot_Method_Distribution.png', dpi=300, bbox_inches='tight')
plt.close()
