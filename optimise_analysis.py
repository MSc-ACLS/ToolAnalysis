#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
optimise_analysis.py

Reads the "optimisation scenarios" CSV (with 3 header rows) and produces ONE figure:
- Left y-axis: Total impact (kg CO2-eq)
- Right y-axis: Productivity (g/L)

Usage:
  python3 optimise_analysis.py ./optimise_analysis.csv
  python3 optimise_analysis.py ./optimise_analysis.csv --outdir out --sort impact
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_float(x):
    """Convert strings like '-', ' 23.8', '16.34' to float; else NaN."""
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s in {"-", "—", "–", "nan", "NaN"}:
        return np.nan
    s = re.sub(r"[^0-9.\-]+", "", s)  # keep digits, dot, minus
    if s in {"", "-", ".", "-."}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_data(csv_path: str) -> pd.DataFrame:
    # Your CSV has 3 header rows; the 3rd row is the real header
    df = pd.read_csv(csv_path, header=2, encoding="utf-8")

    # First column header is blank/unnamed -> rename to Scenario
    first = df.columns[0]
    if str(first).strip() == "" or str(first).startswith("Unnamed"):
        df = df.rename(columns={first: "Scenario"})

    df.columns = [str(c).strip() for c in df.columns]

    if "Scenario" not in df.columns:
        raise ValueError(f"Could not find 'Scenario' column. Columns: {list(df.columns)}")

    for c in df.columns:
        if c == "Scenario":
            continue
        df[c] = df[c].map(to_float)

    needed = ["total impact", "productivity"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Columns: {list(df.columns)}")

    out = df[["Scenario", "total impact", "productivity"]].copy()
    out = out.dropna(subset=["total impact", "productivity"]).copy()
    return out


def plot_dual_axis(df: pd.DataFrame, outpath: str, sort: str):
    d = df.copy()

    if sort == "impact":
        d = d.sort_values("total impact", ascending=False)
    elif sort == "productivity":
        d = d.sort_values("productivity", ascending=False)
    # else: keep file order

    scenarios = d["Scenario"].astype(str).tolist()
    x = np.arange(len(d))
    width = 0.40

    fig, ax1 = plt.subplots(figsize=(11, 5.2))
    ax2 = ax1.twinx()

    # Pick distinct colors explicitly
    color_impact = "#1f77b4"   # blue
    color_prod = "#ff7f0e"     # orange

    # Left axis: impact
    b1 = ax1.bar(
        x - width / 2,
        d["total impact"].to_numpy(dtype=float),
        width,
        label="Total impact (kg CO$_2$-eq)",
        color=color_impact,
    )
    ax1.set_ylabel("Total impact (kg CO$_2$-eq)")

    # Right axis: productivity
    b2 = ax2.bar(
        x + width / 2,
        d["productivity"].to_numpy(dtype=float),
        width,
        label="Productivity (g/L)",
        color=color_prod,
    )
    ax2.set_ylabel("Productivity (g/L)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=30, ha="right")

    ax1.set_title("Optimisation scenarios: total impact and productivity")

    # One combined legend
    handles = [b1, b2]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=None, help="Path to optimise_analysis.csv")
    ap.add_argument("--csv", dest="csv_opt", default=None, help="Same as positional CSV path")
    ap.add_argument("--outdir", default="out", help="Output directory for PNG")
    ap.add_argument("--sort", default="impact", choices=["impact", "productivity", "none"], help="Sort order")
    args = ap.parse_args()

    csv_path = args.csv_opt or args.csv
    if not csv_path:
        ap.error("Please provide a CSV path, e.g. optimise_analysis.py ./optimise_analysis.csv or --csv ...")

    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(csv_path)

    out_png = os.path.join(args.outdir, "impact_and_productivity_dual_axis.png")
    plot_dual_axis(df, out_png, sort=args.sort)

    # Optional: dump cleaned CSV
    df.to_csv(os.path.join(args.outdir, "optimise_analysis_cleaned.csv"), index=False)

    print("Wrote:", os.path.abspath(out_png))


if __name__ == "__main__":
    main()
