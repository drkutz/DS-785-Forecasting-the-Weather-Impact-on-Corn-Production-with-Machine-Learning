# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 19:54:21 2025

@author: drkut
"""
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Helper functions ----------
def _clean_common(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize key columns and coerce types."""
    df = df.copy()

    # Normalize strings to make them uniform
    for col in ["State", "County", "Data Item", "Commodity", "Geo Level", "Domain", "Period", "Ag District"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # ----------Remove "OTHER COUNTIES" bucket-----------
    if "County" in df.columns:
        df = df[df["County"].str.upper() != "OTHER COUNTIES"]

    
    if "County" in df.columns:
        df["County"] = df["County"].str.title()

    # Coerce Year
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    
    if "Value" in df.columns:
        df["Value"] = (
            df["Value"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"(D)": None, "(Z)": None, "NA": None, "": None})
        )
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Coerce Ag District Code + ANSI columns
    if "Ag District Code" in df.columns:
        df["Ag District Code"] = pd.to_numeric(df["Ag District Code"], errors="coerce").astype("Int64")
    for c in ["State ANSI", "County ANSI"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df

    #make it a single row 
def _latest_or_single(df: pd.DataFrame, keys, valcol="Value", how="first"):

    if df.duplicated(keys).any():
        if how == "mean":
            agg = {c: "first" for c in df.columns if c not in keys + [valcol]}
            agg[valcol] = "mean"
            return df.groupby(keys, as_index=False).agg(agg)
        else:
            return df.sort_values(keys).drop_duplicates(keys, keep="first")
    return df


def prepare_yield(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Filter to corn grain yield and rename Value -> Yield."""
    df = _clean_common(df_raw)
    # Keep rows for corn grain yield 
    mask = df["Data Item"].str.contains("CORN, GRAIN", case=False, na=False) & \
           df["Data Item"].str.contains("YIELD", case=False, na=False)
    df = df[mask].copy()

    # Keep ANSI + Ag District columns too  # <<< NEW
    keep = ["Year", "State", "County", "Value"]
    for extra in ["Ag District", "Ag District Code", "State ANSI", "County ANSI"]:
        if extra in df.columns:
            keep.append(extra)

    df = df[keep]
    df = _latest_or_single(df, ["Year", "State", "County"], valcol="Value", how="first")
    df = df.rename(columns={"Value": "Yield_bu_per_acre"})
    return df


def prepare_acres(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Filter to corn acres planted and rename Value -> Acres_Planted."""
    df = _clean_common(df_raw)
    # Match "CORN - ACRES PLANTED"
    mask = df["Data Item"].str.contains("CORN", case=False, na=False) & \
           df["Data Item"].str.contains("ACRES", case=False, na=False) & \
           df["Data Item"].str.contains("PLANTED", case=False, na=False)
    df = df[mask].copy()

    # Keep ANSI + Ag District columns too  
    keep = ["Year", "State", "County", "Value"]
    for extra in ["Ag District", "Ag District Code", "State ANSI", "County ANSI"]:
        if extra in df.columns:
            keep.append(extra)

    df = df[keep]
    df = _latest_or_single(df, ["Year", "State", "County"], valcol="Value", how="first")
    df = df.rename(columns={"Value": "Acres_Planted"})
    return df


def _coalesce_columns(merged: pd.DataFrame, base_col: str, left_sfx="_x", right_sfx="_y") -> pd.DataFrame:
    """Coalesce duplicate columns created by merge (prefer left, else right)."""
    lx, ry = base_col + left_sfx, base_col + right_sfx
    if lx in merged.columns and ry in merged.columns:
        merged[base_col] = merged[lx].where(merged[lx].notna(), merged[ry])
        merged = merged.drop(columns=[lx, ry])
    elif lx in merged.columns:
        merged = merged.rename(columns={lx: base_col})
    elif ry in merged.columns:
        merged = merged.rename(columns={ry: base_col})
    return merged


def merge_state(yield_df: pd.DataFrame, acres_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join yield and acres on Year/State/County; 
    compute implied production per acre;
    keep Ag District + ANSI."""
    merged = pd.merge(
        yield_df, acres_df,
        on=["Year", "State", "County"],
        how="inner",
        suffixes=("_yld", "_acr"),
        validate="one_to_one"
    )

    # Reconcile Ag District / ANSI columns 
    #this was an issues between state data base and the federal govement data base 
    for base_col in ["Ag District", "Ag District Code", "State ANSI", "County ANSI"]:
        ycol, acol = f"{base_col}_yld", f"{base_col}_acr"
        if ycol in merged.columns and acol in merged.columns:
            merged[base_col] = merged[ycol].where(merged[ycol].notna(), merged[acol])
            merged = merged.drop(columns=[ycol, acol])
        elif ycol in merged.columns:
            merged = merged.rename(columns={ycol: base_col})
        elif acol in merged.columns:
            merged = merged.rename(columns={acol: base_col})




    # Implied production becomes actual bushels per acre 
    if "Yield_bu_per_acre" in merged.columns and "Acres_Planted" in merged.columns:
        merged["Implied_Production_bu"] = merged["Yield_bu_per_acre"] * merged["Acres_Planted"]

    # Sort (district-aware if present)  # <<< NEW
    sort_cols = [c for c in ["State", "Ag District Code", "County", "Year"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
    else:
        merged = merged.sort_values(["Year", "State", "County"]).reset_index(drop=True)

    return merged


# ---------- INPUT FILE PATHS ----------

ia_yield_path  = r"corn yield and acres inputs\IA corn by acre by county.csv"
ia_acres_path  = r"corn yield and acres inputs\iowa corn acres planted per year.csv"
il_yield_path  = r"corn yield and acres inputs\IL corn by acre by county.csv"
il_acres_path  = r"corn yield and acres inputs\illnois corn acres planted per year.csv" 
ne_acres_path  = r"corn yield and acres inputs\nebraska corn acres planted per year.csv"
ne_yield_path  = r"corn yield and acres inputs\NE corn yield per acre.csv"

# ---------- LOAD ----------
ia_yield_raw = pd.read_csv(ia_yield_path)
ia_acres_raw = pd.read_csv(ia_acres_path)
il_yield_raw = pd.read_csv(il_yield_path)
il_acres_raw = pd.read_csv(il_acres_path)
ne_yield_raw = pd.read_csv(ne_yield_path)
ne_acres_raw = pd.read_csv(ne_acres_path)

# ---------- PREPARE ----------
ia_yield = prepare_yield(ia_yield_raw)
ia_acres = prepare_acres(ia_acres_raw)
il_yield = prepare_yield(il_yield_raw)
il_acres = prepare_acres(il_acres_raw)
ne_yield = prepare_yield(ne_yield_raw)
ne_acres = prepare_acres(ne_acres_raw)

# ---------- MERGE BY STATE ----------
df_ia = merge_state(ia_yield, ia_acres)
df_il = merge_state(il_yield, il_acres)
df_ne = merge_state(ne_yield, ne_acres)

#-----------Filter Selected Years-------
df_ia = df_ia[(df_ia["Year"] >=1980) & (df_ia["Year"] <= 2019)]
df_il = df_il[(df_il["Year"] >=1980) & (df_il["Year"] <= 2019)]
df_ne = df_ne[(df_ne["Year"] >=1980) & (df_ne["Year"] <= 2019)]
# ---------- CONCATENATE ----------
df_all = pd.concat([df_ia, df_il, df_ne], ignore_index=True)

# Optional: enforce data types
df_all["Year"] = df_all["Year"].astype("Int64")
df_all["State"] = df_all["State"].astype(str)
df_all["County"] = df_all["County"].astype(str)

# Nice column order (keeps both ANSI + Ag District)  # <<< NEW
ordered_cols = [
    "Year", "State", "FIPS", "State ANSI", "County ANSI",
    "Ag District", "Ag District Code", "County",
    "Yield_bu_per_acre", "Acres_Planted", "Implied_Production_bu"
]
final_cols = [c for c in ordered_cols if c in df_all.columns] + [c for c in df_all.columns if c not in ordered_cols]
df_all = df_all[final_cols]

# ---------- OUTPUT ----------
out_path = "corn_yields_acres_planted.csv"
df_all.to_csv(out_path, index=False)

print("Rows:", len(df_all))
print("Columns:", df_all.columns.tolist())
print(df_all.head(10))
print(f"\nSaved to: {out_path}")

(df_all.isna().mean().mul(100).sort_values(ascending=False)).plot(kind='bar'); plt.ylabel('% missing'); plt.title('Missingness by column'); plt.tight_layout()

missing_data = df_all[df_all.isna().any(axis=1)]
