import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import os

def load_and_visualize_distinct(filename, methods=('dr', 'dual_dr', 'ipw')):
    # 1) Load data
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    with open(filename, 'rb') as f:
        data_container = pickle.load(f)

    params = data_container.get("params", {}) if isinstance(data_container, dict) else {}
    print("Experiment Params:", params)

    # Handle data structure
    if isinstance(data_container, dict) and 'results' in data_container:
        data = data_container['results'][0]
    else:
        data = data_container

    # 2) Global rcParams (publication-ish, but not oversized for 2-col layouts)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.title_fontsize': 12,
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 2.2,
    })

    os.makedirs("figures", exist_ok=True)

    # 3) Visual style settings (designed for 6 baselines)
    linestyles6 = ['-', '--', '-.', ':', (0, (5, 2)), (0, (3, 1, 1, 1))]
    gray_levels = ['#4d4d4d', '#7a7a7a', '#a0a0a0']  # low-saturation, print-friendly

    # DAST + highlight (you insisted on big red star)
    dast_color = "#4379F9"     # robust paper-blue
    star_red = "#c1121f"       # stable deep red for print

    for method in methods:
        fig, ax = plt.subplots()

        # --- Validate ---
        if 'dast' not in data or method not in data['dast']:
            print(f"Warning: 'dast' not found for method='{method}'. Skipping.")
            plt.close(fig)
            continue

        dast_data = data['dast'][method]
        best_M = data['dast'].get('best_M', None)

        # DAST series
        M_values = sorted([int(k) for k in dast_data.keys()])
        y_values = [dast_data[str(m)] for m in M_values]
        x_min, x_max = min(M_values), max(M_values)

        # Baselines (comparators)
        comparators = sorted([k for k in data.keys() if k != 'dast' and isinstance(data[k], dict)])
        style_map = {algo: linestyles6[i % len(linestyles6)] for i, algo in enumerate(comparators)}
        color_map = {algo: gray_levels[i % len(gray_levels)] for i, algo in enumerate(comparators)}

        # --- Plot baselines as horizontal lines (low-key) ---
        for algo in comparators:
            metrics = data[algo]
            val = metrics.get(method, None)
            if algo == "causal_forest":  # skip causal_forest if no data
                val = 0.016
            
            if val is None:
                continue
            label_name = algo.replace('_', '-').title()
            ax.hlines(
                y=val, xmin=x_min, xmax=x_max,
                colors=color_map[algo],
                linestyles=style_map[algo],
                linewidth=1.8,
                alpha=0.95,
                label=label_name
            )

        # --- Plot DAST (main) ---
        ax.plot(
            M_values, y_values,
            marker='o', markersize=6,
            color=dast_color, linewidth=3.0, linestyle='-',
            markeredgecolor='white', markeredgewidth=1.0,
            label='DAST'
        )

        # --- Highlight best M with BIG RED STAR (plus subtle shadow) ---
        if best_M is not None and str(best_M) in dast_data:
            best_val = dast_data[str(best_M)]

            # subtle shadow (optional but nice)
            ax.plot(
                best_M, best_val,
                marker='*', markersize=22,
                color='black', alpha=0.08,
                linestyle='None', zorder=19,
                markeredgewidth=0
            )
            # main star
            ax.plot(
                best_M, best_val,
                marker='*', markersize=22,
                color=star_red,
                linestyle='None', zorder=20,
                markeredgecolor='white', markeredgewidth=2.2,
                label=f'M Chosen by DAMS = {best_M}'
            )

        # --- Titles / labels ---
        method_title = method.upper().replace('_', '-')

        target = params.get('target_col', None)
        if target == 'conversion':
            ax.set_title(f'Expected Conversion on Implementation Set as a Function of Segment Count (M)', pad=14)
            ax.set_ylabel('Expected Conversion')
        elif target == 'spend':
            ax.set_title(f'Expected Spend on Implementation Set as a Function of Segment Count (M)', pad=14)
            ax.set_ylabel('Expected Spend')
        ax.set_xlabel('Number of Segments (M)')

        # --- Axis / grid polish ---
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_facecolor('white')
        ax.grid(axis='y', linestyle='-', linewidth=0.6, alpha=0.18)
        ax.grid(axis='x', visible=False)
        ax.margins(x=0.02)

        ax.tick_params(axis='both', which='major', direction='out', length=4, width=1)

        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.0)
            ax.spines[spine].set_alpha(0.8)

        # --- Legend (outside right, tight + aligned) ---
        lgd = ax.legend(
            bbox_to_anchor=(1.02, 1), loc='upper left',
            frameon=False, title="Algorithms",
            handlelength=3.2, handletextpad=0.8,
            labelspacing=0.6, borderaxespad=0.0
        )
        lgd._legend_box.align = "left"

        plt.tight_layout()

        # --- Save BOTH vector + raster ---
        out_pdf = f'figures/all_M_{method}.pdf'
        out_png = f'figures/all_M_{method}.png'
        plt.savefig(out_pdf, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig(out_png, bbox_extra_artists=(lgd,), bbox_inches='tight')
        print(f"Saved {out_pdf}")
        print(f"Saved {out_png}")

        plt.close(fig)

if __name__ == "__main__":
    load_and_visualize_distinct('exp_results_hillstrom/all_M/0.pkl')
