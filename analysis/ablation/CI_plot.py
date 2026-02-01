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

baselines = [
    'gmm', 'gmm_dams',
    'kmeans', 'kmeans_dams',
    'clr', 'clr_dams'
]
target = 'dast'

records = []
raw_data_map = {m: [] for m in baselines + [target]}

for i, run in enumerate(results_list):
    val_target = run[target]
    raw_data_map[target].append(val_target)

    for base in baselines:
        val_base = run[base]
        raw_data_map[base].append(val_base)
        lift = ((val_target - val_base) / val_base) * 100
        records.append({
            'Run': i,
            'Baseline': base,
            'Lift': lift
        })

df = pd.DataFrame(records)

label_map = {
    'kmeans': 'vs. K-Means',
    'kmeans_dams': 'vs. K-Means_DAMS',
    'gmm': 'vs. GMM',
    'gmm_dams': 'vs. GMM_DAMS',
    'clr': 'vs. CLR',
    'clr_dams': 'vs. CLR_DAMS',
}

df['Baseline_Label'] = df['Baseline'].map(label_map)

# ==========================================
# 3. 统计计算（P-value & CI）
# ==========================================
def get_sig_star(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return None

p_values = {
    label_map[b]: stats.ttest_rel(
        raw_data_map[target],
        raw_data_map[b]
    )[1]
    for b in baselines
}

constant_order = [
    'vs. K-Means',
    'vs. K-Means_DAMS',
    'vs. GMM',
    'vs. GMM_DAMS',
    'vs. CLR',
    'vs. CLR_DAMS',
]

summary_stats = []
for label in constant_order:
    subset = df[df['Baseline_Label'] == label]['Lift']
    mean = subset.mean()
    sem = subset.sem()
    ci = sem * stats.t.ppf(0.975, len(subset) - 1)

    summary_stats.append({
        'Label': label,
        'Mean': mean,
        'CI': ci,
        'P_Star': get_sig_star(p_values[label])
    })

stats_df = pd.DataFrame(summary_stats)

# ==========================================
# 4. 颜色 & 线型规则（核心修改）
# ==========================================
def is_dams(label):
    return 'DAMS' in label

color_map = {
    label: ('#C00000' if is_dams(label) else '#000000')
    for label in constant_order
}

linestyle_map = {
    label: ('--' if is_dams(label) else '-')
    for label in constant_order
}

# ==========================================
# 5. 绘图
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(stats_df))

ax.grid(axis='x', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

for i, row in stats_df.iterrows():
    label = row['Label']
    ax.errorbar(
        x=row['Mean'],
        y=i,
        xerr=row['CI'],
        fmt='o',
        color=color_map[label],
        ecolor=color_map[label],
        linestyle=linestyle_map[label],
        capsize=4,
        elinewidth=2,
        markersize=9
    )

ytick_labels = [
    f"{row['Label']}" +
    (f" ({row['P_Star']})" if row['P_Star'] else '')
    for _, row in stats_df.iterrows()
]

ax.set_yticks(y_pos)
ax.set_yticklabels(
    ytick_labels,
    fontweight='bold',
    fontsize=14,
    rotation=30
)
ax.tick_params(axis='y', length=0)

# 0% 基准线
ax.axvline(0, color="#183BD8", linestyle='--', linewidth=1.6, alpha=0.8)

ax.set_xlabel(
    'Averaged DAS Improvement (%)',
    fontweight='bold',
    labelpad=12
)

ax.invert_yaxis()
sns.despine(left=True, top=True, right=True)

# 标题
ax.set_title(
    f'Averaged DAS Improvement (%) on Profits with 95% CI (Runs={n_sims})',
    fontweight='bold',
    fontsize=16,
    pad=20,
    y=1.12
)

# 方向性说明
ax.annotate(
    'Positive values (>0%) indicate DAS outperforms other methods',
    xy=(0.5, 1.06),
    xycoords='axes fraction',
    fontsize=12,
    fontweight='bold',
    color='#333333',
    ha='center',
    va='bottom',
    bbox=dict(
        boxstyle="round,pad=0.3",
        fc="#f0f0f0",
        ec="gray",
        lw=0.5,
        alpha=0.8
    )
)

plt.tight_layout()

# ==========================================
# 6. 导出
# ==========================================
plt.savefig('figures/Fig2_UTD_Style.pdf', bbox_inches='tight')
plt.savefig('figures/Fig2_UTD_Style.png', bbox_inches='tight')
