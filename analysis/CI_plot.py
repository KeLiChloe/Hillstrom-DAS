import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
import warnings

# ==========================================
# 数据加载与处理
# ==========================================
with open("exp_results/main/exp0.pkl", "rb") as f:
    data_exp = pickle.load(f)

# 忽略不必要的警告
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. 画图风格 & 字体设置
# ==========================================
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks", {'axes.grid': True})

plt.rcParams.update({
    # 尝试 STIX，没有的话回退到 Times / 通用 serif
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'axes.labelweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'pdf.fonttype': 42,   # 矢量字体嵌入，投稿友好
    'figure.dpi': 300,
})

# 获取参数和结果
if "params" in data_exp:
    params = data_exp["params"]
    print("Experiment Params:", params)
    results_list = data_exp["results"]
else:
    results_list = data_exp
n_sims = len(results_list)

baselines = ['all_0', 'all_1', 'all_2', 'random', 'kmeans', 'gmm', 'clr', 'mst']

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
        records.append({'Run': i, 'Baseline': base, 'Lift': lift})

df = pd.DataFrame(records)

# ==============================
# 2. Label 映射（带 "vs." 前缀）
# ==============================
label_map = {
    'all_0': 'vs. All Control',
    'all_1': 'vs. All Action=1',
    'all_2': 'vs. All Action=2',
    'random': 'vs. Random', 
    'kmeans': 'vs. K-Means',
    'gmm': 'vs. GMM', 
    'clr': 'vs. CLR',
    'mst': 'vs. MST',
}

df['Baseline_Label'] = df['Baseline'].map(label_map)

# ==========================================
# 3. 统计计算 (P-value & CI)
# ==========================================
def get_sig_star(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return None

p_values = {
    label_map[b]: stats.ttest_rel(raw_data_map[target], raw_data_map[b])[1]
    for b in baselines
}

# 固定展示顺序（如果你希望按 mean 排序，这个只用于过滤）
constant_order = [
    # 'vs. All Action=1',
    # 'vs. All Action=2',
    'vs. All Control',
    'vs. Random',
    'vs. K-Means',
    'vs. GMM',
    'vs. CLR',
    'vs. MST',
]
palette = {
    # --- All-action baselines（绿色系层次） ---
    'vs. All Control':  "#81C784DD",   # 浅绿

    # --- Random baseline（红色系） ---
    'vs. Random':        "#E15759DD",

    # --- Clustering baselines（冷色系+土黄） ---
    'vs. K-Means':       "#5E3C99DD",  # 紫色
    'vs. GMM':           "#F1A340DD",  # 土黄
    'vs. CLR':           "#1F78B4DD",  # 蓝色
    'vs. MST':           "#A6761CDD",  # 棕色
}


summary_stats = []
for label in constant_order:
    subset = df[df['Baseline_Label'] == label]['Lift']
    if subset.empty:
        continue
    mean = subset.mean()
    sem = subset.sem()
    ci = sem * stats.t.ppf(0.975, len(subset) - 1)
    p_star = get_sig_star(p_values[label])
    summary_stats.append({
        'Label': label,
        'Mean': mean,
        'CI': ci,
        'P_Star': p_star,
    })

stats_df = pd.DataFrame(summary_stats)

# ==========================================
# 3.5 按 mean 从大到小排序
# ==========================================
stats_df = stats_df.sort_values('Mean', ascending=False).reset_index(drop=True)

# ==========================================
# 4. 绘图
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(stats_df))

ax.grid(axis='x', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

for i, row in stats_df.iterrows():
    color = palette.get(row['Label'], "#333333")
    ax.errorbar(
        x=row['Mean'],
        y=i,
        xerr=row['CI'],
        fmt='o',
        color=color,
        ecolor=color,
        capsize=4,
        elinewidth=2,
        markersize=9,
    )

ytick_labels = [
    f"{row['Label']}" + (f" ({row['P_Star']})" if row['P_Star'] is not None else '')
    for _, row in stats_df.iterrows()
]
ax.set_yticks(y_pos)
ax.set_yticklabels(ytick_labels, fontweight='bold', fontsize=14)
ax.tick_params(axis='y', length=0)

# 0% 基准线
ax.axvline(0, color="#E40606", linestyle='--', linewidth=1.6, alpha=0.8)

# 轴标题
ax.set_xlabel(
    'Averaged DAST Improvement (%) on Conversion Rate Over Comparators',
    fontweight='bold',
    labelpad=12,
)

# y 轴上方是表现更好的一侧 → invert
ax.invert_yaxis()
sns.despine(left=True, top=True, right=True)

# 标题
ax.set_title(
    f'Averaged DAST Improvement (%) on Conversion Rate Over Comparators with 95% CI (Runs={n_sims})',
    fontweight='bold',
    pad=20,
    fontsize=16,
    y=1.12,
)

# 说明文字
ax.annotate(
    'Positive values (>0%) indicate DAST outperforms comparators',
    xy=(0.5, 1.06),
    xycoords='axes fraction',
    fontsize=12,
    fontweight='bold',
    color='#333333',
    ha='center',
    va='bottom',
    bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", lw=0.5, alpha=0.8),
)

# ==========================================
# 5. Legend
# ==========================================
legend_handles = [
    Line2D([0], [0], color='black', marker='o', linestyle='-',
           linewidth=2, markersize=8, label='Mean ± 95% CI'),
    Line2D([0], [0], color='none',
           label='*** p < 0.001\n**   p < 0.01\n*     p < 0.05'),
    Line2D([0], [0], color="#E40606", linestyle='--',
           linewidth=1.6, label='No Improvement (0%)'),
]

ax.legend(
    handles=legend_handles,
    loc='best',
    frameon=True,
    framealpha=0.95,
    edgecolor='#E0E0E0',
    fancybox=False,
    fontsize=11,
    borderpad=1,
)

plt.tight_layout()

# ==========================================
# 6. 导出
# ==========================================
plt.savefig('figures/Fig2_UTD_Style.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/Fig2_UTD_Style.png', dpi=300, bbox_inches='tight')

