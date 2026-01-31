import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
import warnings

with open("exp_results/ablation/exp4.pkl", "rb") as f:
    data_exp = pickle.load(f)
    
# 忽略不必要的警告
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. 核心修复：强制使用 STIX/Times 风格
# ==========================================
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks", {'axes.grid': True})

plt.rcParams.update({
    'font.family': 'STIXGeneral', 
    'font.serif': ['STIXGeneral', 'Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'axes.labelweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'pdf.fonttype': 42,
    'figure.dpi': 300
})

academic_colors = ['#FF6B6B', '#4ECDC4', "#7F8893", "#9BE200", '#FFCC5C', "#79CBFB",  "#DA7BFF", "#FFB39A"]


# 获取参数
if "params" in data_exp:
    params = data_exp["params"]
    print("Experiment Params:", params) 
    # 输出: {'sample_frac': 0.01, 'pilot_frac': 0.5, 'train_frac': 0.7, 'N_sim': 100}

    # 获取结果列表
    results_list = data_exp["results"]
else:
    results_list = data_exp
n_sims = len(results_list)


baselines = ['gmm', 'gmm_dams', 'kmeans', 'kmeans_dams', 'clr', 'clr_dams']  # 'mst' 可选
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

# 【关键修改 1】：增加 "vs." 前缀，明确对比关系
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
# 3. 统计计算 (P-value & CI)
# ==========================================
def get_sig_star(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return None

p_values = {label_map[b]: stats.ttest_rel(raw_data_map[target], raw_data_map[b])[1] for b in baselines}
# median_order = df.groupby('Baseline_Label')['Lift'].median().sort_values(ascending=False).index.tolist()
constant_order = [
    'vs. K-Means', 
    'vs. K-Means_DAMS',
    'vs. GMM', 
    'vs. GMM_DAMS',
    'vs. CLR',
    'vs. CLR_DAMS',
]
palette = {label: color for label, color in zip(constant_order, academic_colors)}


summary_stats = []
for label in constant_order:
    subset = df[df['Baseline_Label'] == label]['Lift']
    mean = subset.mean()
    sem = subset.sem()
    ci = sem * stats.t.ppf(0.975, len(subset)-1)
    p_star = get_sig_star(p_values[label])
    summary_stats.append({
        'Label': label,
        'Mean': mean,
        'CI': ci,
        'P_Star': p_star,
    })

stats_df = pd.DataFrame(summary_stats)

# ==========================================
# 4. 绘图 (Drawing)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(stats_df))

ax.grid(axis='x', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

for i, row in stats_df.iterrows():
    color = palette[row['Label']]
    ax.errorbar(x=row['Mean'], y=i, xerr=row['CI'], fmt='o',
                color=color, ecolor=color,
                capsize=4, elinewidth=2, markersize=9)

ytick_labels = [f"{row['Label']}" + (f" ({row['P_Star']})" if row['P_Star'] is not None else '') for _, row in stats_df.iterrows()]
ax.set_yticks(y_pos)
ax.set_yticklabels(ytick_labels, fontweight='bold', fontsize=14,  rotation=30,)
ax.tick_params(axis='y', length=0)

# 0% 基准
ax.axvline(0, color="#183BD8", linestyle='--', linewidth=1.6, alpha=0.8)

# 【关键修改 2】：基准线文字标注
# transform=ax.get_xaxis_transform() 让 y 使用坐标轴刻度(0~1)，x 使用数据坐标
# 我们把文字放在底部或者顶部附近


# 轴标题
ax.set_xlabel('Averaged DAS Improvement (%)', fontweight='bold', labelpad=12)

ax.invert_yaxis()
sns.despine(left=True, top=True, right=True)

# 【关键修改 3】：方向性标注 (Higher is Better)
# 放在右下角空白区域，避免遮挡数据
# 1. 调整主标题位置
# y=1.12: 将标题垂直位置手动设定在轴上方 1.12 倍处 (您可以根据需要微调这个数字，比如 1.15)
# pad=...: pad 参数此时主要影响标题边框(如果有)的间距，有了 y 参数后，位置主要由 y 决定
ax.set_title(f'Averaged DAS Improvement (%) on Profits with 95% CI (Runs={n_sims})', 
             fontweight='bold', 
             pad=20,          # 如果用了 y，pad 可以稍微改小或者保持，主要靠 y 控制绝对位置
             fontsize=16, 
             y=1.12)          # <--- 【核心修改】：添加 y 参数，值越大越往上

# 2. 调整 "Higher is Better" 标注位置
# 将 xy 的 y 坐标从原来的 1.02 改为 1.06 (或者更高)，让它跟着标题一起往上走
ax.annotate('Positive values (>0%) indicate DAS outperforms other methods', 
            xy=(0.5, 1.06),             # <--- 【核心修改】：将 1.02 改为 1.06 (往上挪)
            xycoords='axes fraction',
            fontsize=12, fontweight='bold', color='#333333',
            ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", lw=0.5, alpha=0.8))
# ==========================================
# 5. 优化图例 (Legend)
# ==========================================
legend_handles = [
    Line2D([0], [0], color='gray', marker='o', linestyle='-', 
           linewidth=2, markersize=8, label='Mean ± 95% CI'),
    Line2D([0], [0], color='none', label='*** p < 0.001\n**   p < 0.01\n*     p < 0.05'),
    Line2D([0], [0], color='#183BD8', linestyle='--', linewidth=1.6, label='No Improvement (0%)')
]
# ax.legend(handles=legend_handles, 
#           loc='best',
#           frameon=True, 
#           framealpha=0.95, 
#           edgecolor='#E0E0E0', 
#           fancybox=False,      
#           fontsize=11,
#           borderpad=1)

plt.tight_layout()

# ==========================================
# 6. 导出
# ==========================================
plt.savefig('figures/Fig2_UTD_Style.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/Fig2_UTD_Style.png', dpi=300, bbox_inches='tight')
