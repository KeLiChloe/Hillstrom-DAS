"""Replot plot_trend.py summary CSVs without loading experiment pickles.

Usage: python3 analysis/plot_from_csv.py CSV_OR_DIRECTORY [--fig-dir OUTPUT]
"""

import argparse
from pathlib import Path

import pandas as pd

from plot_trend import (
    SweepConfig,
    read_plot_csv,
    configure_plot_style,
    plot_net_profit_trend,
    plot_stz_trend,
    plot_trend,
    plot_treatment_rate_trend,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--fig-dir", type=Path)
    args = parser.parse_args()
    paths = (
        ([args.csv_path / "plot_data.csv"] if (args.csv_path / "plot_data.csv").exists()
         else sorted(args.csv_path.glob("*.csv")))
        if args.csv_path.is_dir()
        else [args.csv_path]
    )
    paths = [p for p in paths if not p.stem.endswith("_runs")]
    if not paths:
        parser.error("No summary CSVs found.")
    configure_plot_style()
    for path in paths:
        df = read_plot_csv(path)
        required = {"x_value", "x_display", "Mean", "CI", "N", "metric",
                    "plot_title", "dataset", "sweep_kind", "fixed_pilot_frac",
                    "n_sims_requested"}
        if df.empty or not required.issubset(df.columns):
            raise ValueError(f"Not a plot_trend summary CSV: {path}")
        groups = (df.groupby("figure_stem", sort=False)
                  if "figure_stem" in df else [(path.stem, df)])
        for stem, df in groups:
            first = df.iloc[0]
            sweep = SweepConfig(
                kind=str(first["sweep_kind"]),
                fixed_pilot_frac=(None if pd.isna(first["fixed_pilot_frac"])
                                  else float(first["fixed_pilot_frac"])),
                dataset=str(first["dataset"]),
                n_by_x=tuple((float(x), int(round(display * 1000)))
                             for x, display in df[["x_value", "x_display"]]
                             .drop_duplicates().itertuples(index=False, name=None)),
            )
            metric = str(first["metric"])
            kwargs = dict(dataset_target=str(first["plot_title"]), sweep=sweep,
                          n_sims=int(first["n_sims_requested"]), out_stem=str(stem),
                          fig_dir=args.fig_dir or path.parent)
            if metric == "treatment_rate":
                plot_treatment_rate_trend(df, **kwargs)
            elif metric == "net_profit":
                plot_net_profit_trend(df, **kwargs)
            elif metric.startswith("stz"):
                plot_stz_trend(df, eval_method=metric, **kwargs)
            else:
                plot_trend(df, eval_method=metric, **kwargs)



if __name__ == "__main__":
    main()
