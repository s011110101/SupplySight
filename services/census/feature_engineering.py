#!/usr/bin/env python3
"""
services/census/feature_engineering.py

Compute derived features for shrimp imports and write a training-ready CSV.

Features implemented (per user request):
- total_weight_mo = VES_WGT_MO + AIR_WGT_MO
- air_share = AIR_WGT_MO / total_weight_mo (0 if denom == 0)
- container_ratio = CNT_WGT_MO / VES_WGT_MO (0 if denom == 0)
- unit_value_per_kg = GEN_VAL_MO / total_weight_mo (NaN if denom == 0)
- weight_mom_pct, weight_yoy_pct, unit_value_mom_pct
- air_share_mom_delta
- rolling 3/6 month means and stds
- weight_zscore_6 handling std==0
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "database" / "processed"
IN_CSV = PROCESSED_DIR / "shrimp_imports.csv"
OUT_CSV = PROCESSED_DIR / "shrimp_features.csv"


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Input not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    # parse and sort by commodity then month
    df["MONTH"] = pd.to_datetime(df["MONTH"], format="%Y-%m")
    df = df.sort_values(["I_COMMODITY", "MONTH"]).reset_index(drop=True)

    # Ensure numeric columns exist and are numeric
    for col in ["GEN_VAL_MO", "VES_WGT_MO", "CNT_WGT_MO", "AIR_WGT_MO"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    # computation per commodity group
    def compute_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["total_weight_mo"] = g["VES_WGT_MO"].fillna(0) + g["AIR_WGT_MO"].fillna(0)
        g["air_share"] = np.where(g["total_weight_mo"] == 0, 0.0, g["AIR_WGT_MO"] / g["total_weight_mo"])
        g["container_ratio"] = np.where(g["VES_WGT_MO"] == 0, 0.0, g["CNT_WGT_MO"] / g["VES_WGT_MO"])
        g["unit_value_per_kg"] = np.where(g["total_weight_mo"] == 0, np.nan, g["GEN_VAL_MO"] / g["total_weight_mo"])
        g["weight_mom_pct"] = g["total_weight_mo"].pct_change(1)
        g["weight_yoy_pct"] = g["total_weight_mo"].pct_change(12)
        g["unit_value_mom_pct"] = g["unit_value_per_kg"].pct_change(1)
        g["air_share_mom_delta"] = g["air_share"] - g["air_share"].shift(1)
        g["weight_roll3_avg"] = g["total_weight_mo"].rolling(window=3, min_periods=1).mean()
        g["weight_roll6_avg"] = g["total_weight_mo"].rolling(window=6, min_periods=1).mean()
        g["weight_roll3_std"] = g["total_weight_mo"].rolling(window=3, min_periods=1).std().fillna(0.0)
        g["weight_roll6_std"] = g["total_weight_mo"].rolling(window=6, min_periods=1).std().fillna(0.0)
        g["weight_zscore_6"] = np.where(
            g["weight_roll6_std"] == 0,
            0.0,
            (g["total_weight_mo"] - g["weight_roll6_avg"]) / g["weight_roll6_std"],
        )
        return g

    df = df.groupby("I_COMMODITY", group_keys=False).apply(compute_group)

    # Convert MONTH back to YYYY-MM for output
    df["MONTH"] = df["MONTH"].dt.strftime("%Y-%m")

    # Select and order columns for output
    out_cols = [
        "I_COMMODITY",
        "I_COMMODITY_SDESC",
        "GEN_VAL_MO",
        "VES_WGT_MO",
        "CNT_WGT_MO",
        "AIR_WGT_MO",
        "MONTH",
        "total_weight_mo",
        "air_share",
        "container_ratio",
        "unit_value_per_kg",
        "weight_mom_pct",
        "weight_yoy_pct",
        "unit_value_mom_pct",
        "air_share_mom_delta",
        "weight_roll3_avg",
        "weight_roll6_avg",
        "weight_roll3_std",
        "weight_roll6_std",
        "weight_zscore_6",
    ]

    # Keep only columns that exist (be robust)
    out_cols = [c for c in out_cols if c in df.columns]
    df_out = df.loc[:, out_cols]

    df_out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df_out)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
