#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_tool_analysis_plots.py

Plots (using protein concentration g/L as the productivity normaliser):
  1) Total impact vs protein concentration (scatter)
  2) Impact breakdown per protein concentration (stacked bar chart)

Input:  CSV exported from the tool (e.g. "tool analysis.csv")
Output: PNG files in ./out (configurable)

Notes:
- The CSV has a group header row, then the real header row, then a units row.
  We read with header=1 and drop rows where ID is missing.
- Some cells contain stray characters (e.g., "602.33Ê", "0.73�"). We strip
  non-numeric characters before converting to float.
"""

from __future__ import annotations

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_float(x) -> float:
    """Convert messy numeric strings (e.g. '1.25Ê', '-', '') to float or NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s in {"", "-", "—", "–", "nan", "NaN"}:
        return np.nan
    # keep digits, dot, minus; drop everything else
    s = re.sub(r"[^0-9.\-]+", "", s)
    if s in {"", "-", ".", "-."}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=1, encoding="cp1252")

    # Drop the units row (usually has empty ID)
    if "ID" in df.columns:
        df = df[df["ID"].notna()].copy()

    df.columns = [c.strip() for c in df.columns]

    # Numeric columns (convert if present)
    numeric_cols = [
        "Duration",
        "CO2 added",
        "DW",
        "Avg Temperature",
        "PAR",
        "PAR 2",
        "pH",

        # emissions columns (kg CO2-eq per run)
        "Reactor",
        "Electricity",
        "CO2",
        "(NH4)2SO4",
        "NaClO",
        "Total",

        # productivity
        "Areal",
        "Volumetric",
        "Biomass",
        "Proteinx",   # if present (your grams column seems renamed to this)
        "Protein",    # your LAST column (g/L)
        "Fixed CO2",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].map(to_float)

    # --- Use protein concentration g/L (last "Protein" column) ---
    protein_cols = [c for c in df.columns if c.strip() == "Protein"]
    if not protein_cols:
        raise ValueError("Expected a 'Protein' column (g/L) in the CSV export.")
    protein_gL_col = protein_cols[-1]  # take the last occurrence
    df["protein_gL"] = df[protein_gL_col].map(to_float)

    # keep only valid >0 protein concentrations
    df = df[df["protein_gL"].notna() & (df["protein_gL"] > 0)].copy()

    # Compute intensities normalised by protein concentration (g/L)
    impact_parts = ["Reactor", "Electricity", "CO2", "(NH4)2SO4", "NaClO", "Total"]
    for c in impact_parts:
        if c in df.columns:
            df[c] = df[c].map(to_float)  # force numeric again
            df[f"{c}_per_gL_protein"] = df[c] / df["protein_gL"]

    df["label"] = df["ID"].astype(str)
    return df


def plot_total_vs_protein_conc(df: pd.DataFrame, outpath: str):
    x = df["protein_gL"].values
    y = df["Total"].values

    plt.figure()
    plt.scatter(x, y)

    for _, r in df.iterrows():
        plt.annotate(
            str(r["ID"]),
            (r["protein_gL"], r["Total"]),
            fontsize=8,
            xytext=(4, 2),
            textcoords="offset points"
        )

    plt.xlabel("Protein concentration (g/L)")
    plt.ylabel("Total impact (kg CO2-eq per run)")
    plt.title("Total impact vs protein concentration")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_breakdown_stacked(df: pd.DataFrame, outpath: str):
    parts = ["Reactor", "Electricity", "CO2", "(NH4)2SO4", "NaClO"]
    per_cols = [f"{p}_per_gL_protein" for p in parts]
    missing = [c for c in per_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing computed columns for stacked breakdown: {missing}")

    # sort by ID as requested
    d = df.sort_values("ID").copy()

    plt.figure(figsize=(9, 5))
    bottom = np.zeros(len(d), dtype=float)

    for p in parts:
        vals = d[f"{p}_per_gL_protein"].to_numpy(dtype=float)
        vals = np.nan_to_num(vals, nan=0.0)
        plt.bar(d["label"], vals, bottom=bottom, label=p)
        bottom += vals

    plt.ylabel("Impact / protein concentration (kg CO2-eq · L / g protein)")
    plt.xlabel("Experiment ID")
    plt.title("Impact intensity breakdown per protein concentration")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to the tool export CSV (e.g. 'tool analysis.csv').")
    ap.add_argument("--outdir", default="out", help="Output directory for PNG files.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_and_clean(args.csv)

    plot_total_vs_protein_conc(df, os.path.join(args.outdir, "impact_vs_protein_concentration.png"))
    plot_breakdown_stacked(df, os.path.join(args.outdir, "impact_breakdown_stacked.png"))

    # Optional: dump cleaned table
    df.to_csv(os.path.join(args.outdir, "tool_analysis_cleaned.csv"), index=False)

    print("Wrote plots to:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
