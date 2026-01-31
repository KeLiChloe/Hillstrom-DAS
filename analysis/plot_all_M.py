import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pickle
import itertools
import os

def load_and_visualize_distinct(filename, methods=['dr', 'dual_dr', 'ipw']):
    # 1. Load data
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    with open(filename, 'rb') as f:
        data_container = pickle.load(f)
        
    params = data_container.get("params", {})
    print("Experiment Params:", params)
    
    # Handle data structure
    if isinstance(data_container, dict) and 'results' in data_container:
        data = data_container['results'][0]
    else:
        data = data_container
        
    

    # 2. Configuration for Publication-Quality Plots
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'legend.title_fontsize': 16,
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'lines.linewidth': 2.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    # Custom distinct colors (High contrast, excluding the blue used for DAST)
    distinct_colors = [
        '#e41a1c', # Red
        '#4daf4a', # Green
        '#984ea3', # Purple
        '#ff7f00', # Orange
        '#a65628', # Brown
        '#f781bf', # Pink
        '#999999', # Grey
        '#dede00', # Yellowish
        '#00ced1', # Dark Turquoise
    ]
    
    # Line styles to cycle through for better distinction
    line_styles = ['--', '-.', ':']

    for method in methods:
        fig, ax = plt.subplots()

        if 'dast' not in data:
            print(f"Warning: 'dast' not found for {method}")
            continue
            
        dast_data = data['dast'][method]
        best_M = data['dast']['best_M']
        M_values = sorted([int(k) for k in dast_data.keys()])
        y_values = [dast_data[str(m)] for m in M_values]
        x_min, x_max = min(M_values), max(M_values)

        # Prepare iterators
        color_cycle = itertools.cycle(distinct_colors)
        style_cycle = itertools.cycle(line_styles)

        # Sort comparators to ensure consistent legend order
        comparators = sorted([k for k in data.keys() if k != 'dast' and isinstance(data[k], dict)])
        
        # --- Plot Comparators ---
        for algo in comparators:
            metrics = data[algo]
            val = metrics.get(method)
            if val is not None:
                label_name = algo.replace('_', '-').title()
                current_color = next(color_cycle)
                current_style = next(style_cycle)
                
                ax.hlines(y=val, xmin=x_min, xmax=x_max,
                           colors=current_color,
                           linestyles=current_style, 
                           linewidth=2.5, 
                           label=label_name)

        # --- Plot DAST (Proposed) ---
        dast_color = "#5294c2" # Standard robust blue
        ax.plot(M_values, y_values, marker='o', markersize=8, label='DAST',
                 color=dast_color, linewidth=3, linestyle='-')

        # --- Highlight Best M ---
        if str(best_M) in dast_data:
            best_val = dast_data[str(best_M)]
            highlight_color = '#d62728' 
            # Added white edge to the star to make it pop
            ax.plot(best_M, best_val, marker='*', color=highlight_color, markersize=20,
                     label=f'M Chosen by DAMS = {best_M}', zorder=10, linestyle='None', 
                     markeredgecolor='white', markeredgewidth=1.5)

        # --- Formatting ---
        method_title = method.upper().replace('_', '-')
        if params['target_col'] == 'conversion':
            ax.set_title(f'Expected Conversion on Implementation Set (OPE - {method_title})', pad=20)
        elif params['target_col'] == 'spend':
            ax.set_title(f'Expected Spend on Implementation Set (OPE - {method_title})', pad=20)
        ax.set_xlabel('Number of Segments (M)')
        ax.set_ylabel('Expected Conversion')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Legend outside
        lgd = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                        frameon=False, title="Algorithms")
        lgd._legend_box.align = "left"

        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7)
        plt.tight_layout()

        # Save
        filename = f'figures/all_M_{method}.png'
        plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
        print(f"Saved {filename}")

# Run the function
if __name__ == "__main__":
    load_and_visualize_distinct('exp_results_hillstrom/all_M/3.pkl')