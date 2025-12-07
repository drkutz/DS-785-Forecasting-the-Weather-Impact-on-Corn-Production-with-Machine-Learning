
# -*- coding: utf-8 -*-
"""
this was a script written merely to produce visuals and understanding for the presentation 3 
and to create visuals that can help bring better understanding of the project
it could be swapped with the corn weather data annual at district level. 
It was done on the daily level since small intricaies could be measured more accurately 
"""
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- load the data abd create the output path -----------------
INPUT_CSV = r"C:\Users\drkut\OneDrive\Documents\ds 785\corn_weather_merged_district_daily_new.csv"
TARGET = "Yield_bu_per_acre"                                   
GROUP_COL = "State"                                
TIME_COL = "Year"                                  
ENSO_CSV = None                                    

# Output dir file to store all the images 
OUT_DIR = Path("./pres3_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helper functions ----------
def _num_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    return cols

def _save_fig(name: str):
    out = OUT_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")

def _safe_title(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", " ")).strip().replace(" ", "_")

# ---------- LOAD DATA ----------
print(f" Loading data: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

if TARGET not in df.columns:
    raise ValueError(f"TARGET '{TARGET}' not found in columns: {df.columns.tolist()}")

if ENSO_CSV:
    print(f"Loading ENSO: {ENSO_CSV}")
    enso = pd.read_csv(ENSO_CSV)
    if not {"Year","ENSO_phase"}.issubset(enso.columns):
        raise ValueError("ENSO_CSV must contain Year and ENSO_phase columns.")
    df = df.merge(enso[["Year","ENSO_phase"]], on="Year", how="left")

# ---------- BASIC STATS ----------
num_cols = _num_cols(df, exclude=[TARGET])
if TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]):
    num_cols_with_target = _num_cols(df)
else:
    num_cols_with_target = num_cols

desc = df[num_cols_with_target].describe().T
desc["skew"] = df[num_cols_with_target].skew(numeric_only=True)
desc["kurtosis"] = df[num_cols_with_target].kurtosis(numeric_only=True)

stats_path = OUT_DIR / "descriptive_statistics.csv"
desc.to_csv(stats_path)
print(f"{stats_path}")

# ---------- HISTOGRAMS ----------
print(" Creating histograms...")
for col in num_cols_with_target:
    s = df[col].dropna()
    if s.empty: 
        continue
    plt.figure()
    plt.hist(s.values, bins=30)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {col}")
    _save_fig(f"hist_{_safe_title(col)}")

# ---------- BOXPLOTS ----------
print(" Creating boxplots...")
for col in num_cols_with_target:
    s = df[col].dropna()
    if s.empty:
        continue
    plt.figure()
    plt.boxplot(s.values, vert=True, labels=[col])
    plt.title(f"Boxplot of {col}")
    _save_fig(f"box_{_safe_title(col)}")

# ---------- CORRELATION HEATMAP ----------
print(" Creating correlation heatmap...")
corr = df[num_cols_with_target].corr(numeric_only=True)
corr_path = OUT_DIR / "correlations.csv"
corr.to_csv(corr_path)
print(f"{corr_path}")

plt.figure()
plt.imshow(corr.values, interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation Heatmap")
_save_fig("correlation_heatmap")

# ---------- SCATTERS vs TARGET ----------
print(" Creating scatterplots vs target...")
for col in num_cols:
    if col == TARGET:
        continue
    x = df[col].values
    y = df[TARGET].values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        continue
    plt.figure()
    if "State" in df.columns:
        # Color by state using categorical codes
        states = df.loc[mask, "State"].astype("category")
        colors = states.cat.codes  # numeric color codes
        scatter = plt.scatter(x[mask], y[mask], c=colors, alpha=0.6)
        # Add legend
        handles = []
        for state, code in zip(states.cat.categories, range(len(states.cat.categories))):
            handles.append(plt.Line2D([], [], color=scatter.cmap(scatter.norm(code)), marker='o', 
                                      linestyle='', label=state))
        plt.legend(handles=handles, title="State", loc="best")
    else:
        plt.scatter(x[mask], y[mask], alpha=0.6)

    plt.xlabel(col)
    plt.ylabel(TARGET)
    plt.title(f"{TARGET} vs {col}")
    _save_fig(f"scatter_{_safe_title(TARGET)}_vs_{_safe_title(col)}")

# ---------- TREND LINE BY GROUP ----------
if GROUP_COL and TIME_COL and GROUP_COL in df.columns and TIME_COL in df.columns and TARGET in df.columns:
    print(" Creating trend line by group...")
    agg = df.groupby([GROUP_COL, TIME_COL])[TARGET].mean().reset_index()
    plt.figure()
    for g in sorted(agg[GROUP_COL].dropna().unique()):
        sub = agg[agg[GROUP_COL] == g]
        if sub.empty: 
            continue
        plt.plot(sub[TIME_COL].values, sub[TARGET].values, label=str(g))
    plt.xlabel(TIME_COL)
    plt.ylabel(TARGET)
    plt.title(f"{TARGET} Trend by {GROUP_COL}")
    plt.legend(loc="best")
    _save_fig(f"trend_{_safe_title(TARGET)}_by_{_safe_title(GROUP_COL)}")


# ---------- SHORT TEXT REPORT ----------
report_lines = []
report_lines.append(f"Generated: {pd.Timestamp.now()}")
report_lines.append("")
report_lines.append("Descriptive statistics saved to descriptive_statistics.csv")
report_lines.append("Correlations saved to correlations.csv")
report_lines.append("Figures saved as PNGs.")

# Highlight a few correlations (top |corr| with target)
if TARGET in corr.columns:
    corrs_to_target = corr[TARGET].drop(labels=[TARGET], errors="ignore").abs().sort_values(ascending=False)
    top = corrs_to_target.head(5)
    report_lines.append("")
    report_lines.append("Top absolute correlations with target:")
    for k, v in top.items():
        report_lines.append(f"  - {k}: {v:.3f}")

report_path = OUT_DIR / "summary_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"saved to {report_path}")

print("EDA visuals created. ")
