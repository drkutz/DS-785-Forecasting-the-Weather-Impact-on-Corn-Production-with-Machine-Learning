#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
By Adam Gruber
take the individual files of weather data and combine the into a State of weather data by Ag District

Outputs:
- Per-state tidy / filtered / completeness CSVs
- ALL-states tidy / filtered / completeness CSVs
"""

import argparse
import os
import re
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd


EXPECTED_COLS = [
    "Date",
    "TMAX (Degrees Fahrenheit)",
    "TMIN (Degrees Fahrenheit)",
    "PRCP (Inches)",
    "SNOW (Inches)",
    "SNWD (Inches)",
]

# ---------create the helper functions -----------------------
def extract_ag_district(filename: str) -> Optional[int]:
    #------------Return the integer ag-district from filename pattern like -------
    #I created the file names to include the district number since raw data did not include it
    base = os.path.basename(filename)
    m = re.search(r"[dD]\s*(\d+)", base)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def read_station_csv(path: str) -> pd.DataFrame:
    """
    Reads a single CSV with this format:
      - first line: station name in quotes
      - second line: header row (Date,TMAX,TMIN,PRCP,SNOW,SNWD; TAVG may appear and is ignored)
      - next lines: daily data
    Returns a DataFrame with expected columns + Station + D_Number + Year.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().strip('"')  # Station name since this 
        rest = f.read()

    df = pd.read_csv(StringIO(rest))

    # Keep expected columns
    keep = [c for c in df.columns if c in EXPECTED_COLS]
    df = df[keep].copy()
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    # --------Types ensure they are numeric 
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["TMAX (Degrees Fahrenheit)", "TMIN (Degrees Fahrenheit)", "PRCP (Inches)", "SNOW (Inches)", "SNWD (Inches)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    
    df["Year"] = df["Date"].dt.year
    df["Station"] = first_line
    df["Ag District Code"] = extract_ag_district(path)

    return df


def find_weather_files(input_dir: str) -> List[str]:
    files = []
    for name in os.listdir(input_dir):
        lower = name.lower()
        if lower.endswith(".csv") and "weather" in lower:
            files.append(os.path.join(input_dir, name))
    files.sort()
    return files


def combine_weather_dir(input_dir: str) -> pd.DataFrame:
    files = find_weather_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files with 'weather' found in: {input_dir}")
    frames = []
    for f in files:
        try:
            frames.append(read_station_csv(f))
        except Exception as e:
            print(f"Skipping {f} due to parse error: {e}")
    if not frames:
        raise RuntimeError(f"All files failed to parse under {input_dir}")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Station", "Date"]).reset_index(drop=True)
    return combined


def make_completeness_summary(df: pd.DataFrame, start_year: int, end_year: int, by_cols: List[str]) -> pd.DataFrame:
    """
    Computes completeness over the window, grouping by `by_cols` ["State", "Station"] or just ["Station"]). 
    Returns non-null counts and percent availability per variable.
    """
    window = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)].copy()
    metrics = ["TMAX (Degrees Fahrenheit)", "TMIN (Degrees Fahrenheit)", "PRCP (Inches)", "SNOW (Inches)", "SNWD (Inches)"]
    summary_rows = []
    for keys, g in window.groupby(by_cols, dropna=False):
        # keys can be a scalar or a tuple, normalize to dict
        key_dict = {}
        if isinstance(keys, tuple):
            for col, val in zip(by_cols, keys):
                key_dict[col] = val
        else:
            key_dict[by_cols[0]] = keys
        row = {**key_dict, "Rows (within window)": len(g)}
        total = len(g)
        for m in metrics:
            non_null = g[m].notna().sum()
            row[f"{m} non-null"] = int(non_null)
            row[f"{m} % available"] = round((non_null / total * 100.0) if total else 0.0, 2)
        summary_rows.append(row)
    if not summary_rows:
        return pd.DataFrame(columns=by_cols + ["Rows (within window)"])
    summary = pd.DataFrame(summary_rows).sort_values(by_cols).reset_index(drop=True)
    return summary


def _parse_state_dirs(state_dirs: Optional[Dict[str, str]], input_root: Optional[str]) -> Dict[str, str]:
    """
    Build {STATE: PATH} from either a provided mapping  or an input_root
    whose immediate subfolders are treated as states. 
    This was done to copy the system i had for downloading all the data by state 
    """
    if state_dirs:
        # Validate directories
        mapping = {}
        for state, path in state_dirs.items():
            if not os.path.isdir(path):
                raise ValueError(f"Directory not found for {state}: {path}")
            mapping[state] = path
        return mapping

    if input_root:
        if not os.path.isdir(input_root):
            raise ValueError(f"--input-root not found: {input_root}")
        mapping = {}
        for name in os.listdir(input_root):
            p = os.path.join(input_root, name)
            if os.path.isdir(p):
                mapping[name] = p
        if not mapping:
            raise ValueError(f"No subdirectories found under input_root: {input_root}")
        return mapping

    raise ValueError("Provide either state_dirs or input_root")


def run_multi(
    output_dir: str,
    state_dirs: Optional[Dict[str, str]] = None,
    input_root: Optional[str] = None,
    start_year: int = 1980,
    end_year: int = 2019,
) -> Dict[str, str]:
    """

    Args:
        output_dir: where to write CSV outputs
        
        input_root: path with subfolders 
        start_year, end_year: filter window

    Returns:
        A dict of key output file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    state_map = _parse_state_dirs(state_dirs=state_dirs, input_root=input_root)

    all_frames = []
    outputs = {}
#-----------process each state and the csv files 
    for state, path in sorted(state_map.items()):
        print(f" Processing state {state} from {path}")
        df_state = combine_weather_dir(path)
        df_state["State"] = state
        all_frames.append(df_state)

        tidy_out = os.path.join(output_dir, f"weather_combined_tidy_{state}.csv")
        df_state.to_csv(tidy_out, index=False)
        print(f" Wrote {tidy_out} (rows={len(df_state):,})")

        window = df_state[(df_state["Year"] >= start_year) & (df_state["Year"] <= end_year)].copy()
        window_out = os.path.join(output_dir, f"weather_combined_{start_year}_{end_year}_{state}.csv")
        window.to_csv(window_out, index=False)
        print(f" Wrote {window_out} (rows={len(window):,})")

        summary = make_completeness_summary(df_state, start_year, end_year, by_cols=["State", "Station"])
        summary_out = os.path.join(output_dir, f"weather_completeness_{start_year}_{end_year}_{state}.csv")
        summary.to_csv(summary_out, index=False)
        print(f" Wrote {summary_out}")

        outputs[f"tidy_{state}"] = tidy_out
        outputs[f"window_{state}"] = window_out
        outputs[f"summary_{state}"] = summary_out

    combined_all = pd.concat(all_frames, ignore_index=True)
    combined_all = combined_all.sort_values(["State", "Station", "Date"]).reset_index(drop=True)

    all_tidy_out = os.path.join(output_dir, "weather_combined_tidy_ALL.csv")
    combined_all.to_csv(all_tidy_out, index=False)
    print(f" Wrote {all_tidy_out} (rows={len(combined_all):,})")

    all_window = combined_all[(combined_all["Year"] >= start_year) & (combined_all["Year"] <= end_year)].copy()
    all_window_out = os.path.join(output_dir, f"weather_combined_{start_year}_{end_year}_ALL.csv")
    all_window.to_csv(all_window_out, index=False)
    print(f" Wrote {all_window_out} (rows={len(all_window):,})")

    all_summary = make_completeness_summary(combined_all, start_year, end_year, by_cols=["State", "Station"])
    all_summary_out = os.path.join(output_dir, f"weather_completeness_{start_year}_{end_year}_ALL.csv")
    all_summary.to_csv(all_summary_out, index=False)
    print(f" Wrote {all_summary_out}")

    outputs["all_tidy"] = all_tidy_out
    outputs["all_window"] = all_window_out
    outputs["all_summary"] = all_summary_out
    return outputs




if __name__ == "__main__":
    run_multi(
        output_dir="out",
        state_dirs={
            "IA": "IA weather",
            "IL": "IL weather",
            "NE": "NE weather",
        },
        start_year=1980,
        end_year=2019)
