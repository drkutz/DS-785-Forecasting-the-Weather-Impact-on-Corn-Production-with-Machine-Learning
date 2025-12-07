# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 22:10:20 2025

@author: drkut
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os,glob

# Paths
yields_path = "corn_yields_acres_planted.csv"
weather_path = r"out\weather_combined_1980_2019_ALL.csv"
enso = "enso_events.csv"
# Load
df_yields = pd.read_csv(yields_path)
df_weather = pd.read_csv(weather_path)
enso_df = pd.read_csv(enso) #this is a simple csv file 
prices_dir = r"C:\Users\drkut\OneDrive\Documents\ds 785\prices"

#--------- standardize the states
state_map = {
    "ILLINOIS": "IL","INDIANA": "IN",
    "IOWA": "IA","NEBRASKA": "NE"
}
for df in (df_yields, df_weather):
    df["State"] = df["State"].astype(str).str.strip().str.upper().replace(state_map)

df_yields["Ag District Code"] = df_yields["Ag District Code"].astype("Int64")
df_weather["Ag District Code"] = df_weather["Ag District Code"].astype("Int64")
df_yields["Year"] = df_yields["Year"].astype("Int64")
df_weather["Year"] = df_weather["Year"].astype("Int64")

# -------- Aggregate yields to Ag District × Year----- ---
agg = (
    df_yields
    .groupby(["Year", "State", "Ag District Code"], as_index=False)
    .agg(
        Implied_Production_bu=("Implied_Production_bu", "sum"),
        Acres_Planted=("Acres_Planted", "sum"),
    )
)
agg["Yield_bu_per_acre"] = agg["Implied_Production_bu"] / agg["Acres_Planted"]

#merge the weather stations to yield
df_daily = pd.merge(
    df_weather,
    agg[["Year", "State", "Ag District Code", "Yield_bu_per_acre", "Acres_Planted"]],
    on=["Year", "State", "Ag District Code"],
    how="inner"
)

#%% Pricing section to add prices to the weather data and corn yields 


df_daily = df_daily.copy()
df_daily["Year"]  = pd.to_numeric(df_daily["Year"], errors="coerce").astype("Int64")
df_daily["Month"] = pd.to_datetime(df_daily["Date"], errors="coerce").dt.month
df_daily["State"] = df_daily["State"].astype(str).str.upper()

month_map = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
             "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
state_map = {"IOWA":"IA","ILLINOIS":"IL","INDIANA":"IN","NEBRASKA":"NE"}

def tidy(fp):
    d = pd.read_csv(fp, dtype=str)
    cols = [c for c in ["Year","Period","State","Commodity","Value"] if c in d.columns]
    d = d[cols].copy()
    d["Year"]   = pd.to_numeric(d["Year"], errors="coerce").astype("Int64")
    d["Period"] = d["Period"].str.upper().str.strip()
    d           = d[d["Period"].isin(month_map)].copy()
    d["Month"]  = d["Period"].map(month_map)
    d["State"]  = d["State"].astype(str).str.upper().replace(state_map)
    d["Commodity"] = d.get("Commodity","").astype(str).str.upper().str.strip()
    d["Price"]  = pd.to_numeric(d["Value"].str.replace(",", "", regex=False), errors="coerce")
    return d[["State","Year","Month","Commodity","Price"]]

files = glob.glob(os.path.join(prices_dir, "*.csv"))
prices = pd.concat([tidy(fp) for fp in files], ignore_index=True)

# Pivot to wide; rename columns to include $/bu
pivot = (prices
         .pivot_table(index=["State","Year","Month"], columns="Commodity", values="Price", aggfunc="mean")
         .rename(columns={
             "CORN": "Corn $/bu",
             "SOYBEANS": "Soybeans $/bu"
         })
         .reset_index())

df_daily = df_daily.merge(pivot, on=["State","Year","Month"], how="left")




#%%
#add the enso events to the weather data 
enso_df["ENSO"] = np.select(
    [
        enso_df["Event"].astype(str).str.contains("nino", case=False, na=False),
        enso_df["Event"].astype(str).str.contains("nina", case=False, na=False),
    ],
    [1, -1],
    default=0
)

df_daily = df_daily.merge(
    enso_df[["Year", "ENSO"]].rename(columns={"Year": "Year"}),
    on="Year", how="left")
# --------------- Reorder columns -------
front_cols = [
    "Date",
    "TMAX (Degrees Fahrenheit)",
    "TMIN (Degrees Fahrenheit)",
    "PRCP (Inches)",
    "SNOW (Inches)",
    "SNWD (Inches)",
    "Year",
    "Station",
    "Yield_bu_per_acre",
    "Acres_Planted",
    "ENSO"]

remaining = [c for c in df_daily.columns if c not in front_cols]
df_final = df_daily[front_cols + remaining + ["ENSO"]]

#%%
COL_TMAX = "TMAX (Degrees Fahrenheit)"
COL_TMIN = "TMIN (Degrees Fahrenheit)"
COL_PRCP = "PRCP (Inches)"

#
'''building the three created features that can help better predict the corn yield
    3 features to be built better capture the issue with corn yield
    1 growing degree days an industry standard
    2 extreme heat days over 95 degrees
    3 longest dry spell 
    
'''
# Ensure Date and Month exist/are correct
df_daily["Date"] = pd.to_datetime(df_daily["Date"], errors="coerce")
df_daily["Month"] = df_daily["Date"].dt.month
df_daily["Year"] = pd.to_numeric(df_daily["Year"], errors="coerce").astype("Int64")

# 1) Daily GDD (base 50F), clipped at 0, then seasonal sum
df_daily["AVG_TEMP"] = (df_daily[COL_TMAX] + df_daily[COL_TMIN]) / 2.0
df_daily["GDD_base50"] = (df_daily["AVG_TEMP"] - 50).clip(lower=0)

# 2) Extreme heat indicator (TMAX > 90F), then seasonal count
df_daily["ExtremeHeatDay"] = (df_daily[COL_TMAX] > 90).astype(int)

# 3) Longest dry spell (PRCP < 0.1") within the growing season
def _longest_dry_spell(prcp_series):
    dry = (prcp_series < 0.1).astype(int)
    # consecutive-run lengths
    runs = dry.groupby((dry != dry.shift()).cumsum()).cumsum()
    return runs.max()

# -----------Define the growing season (Apr–Sep)-----------------
season_mask = df_daily["Month"].between(4, 9, inclusive="both")
season_df = df_daily.loc[season_mask].copy()

# ------- Compute per STATION first ----------------
per_station = (
    season_df
    .groupby(["State", "Ag District Code", "Station", "Year"], dropna=True)
    .agg(
        GDD_total=("GDD_base50", "sum"),
        ExtremeHeatDays=("ExtremeHeatDay", "sum"),
        PRCP_total=(COL_PRCP, "sum"),                    
        TMAX_mean=(COL_TMAX, "mean"),
        TMIN_mean=(COL_TMIN, "mean"),
    )
    .reset_index())


# -------Dry spell needs the full daily sequence------------
dry_spell_station = (
    season_df
    .sort_values(["State","Ag District Code","Station","Year","Date"])
    .groupby(["State", "Ag District Code", "Station", "Year"])[COL_PRCP]
    .apply(_longest_dry_spell)
    .reset_index(name="MaxDrySpellDays"))

per_station = per_station.merge(
    dry_spell_station,
    on=["State","Ag District Code","Station","Year"],
    how="left")

# --- Average across stations within each Ag District by Year ---
features_ad = (
    per_station
    .groupby(["State", "Ag District Code", "Year"], dropna=True)
    .agg(
        GDD_total_mean=("GDD_total", "mean"),
        ExtremeHeatDays_mean=("ExtremeHeatDays", "mean"),
        MaxDrySpellDays_mean=("MaxDrySpellDays", "mean"),
        PRCP_total_mean=("PRCP_total", "mean"),
        TMAX_mean=("TMAX_mean", "mean"),
        TMIN_mean=("TMIN_mean", "mean"),
        n_stations=("Station", "nunique"),
    )
    .reset_index())

#  Keep ENSO at the year level
enso_year = df_daily[["State","Ag District Code","Year","ENSO"]].drop_duplicates(subset=["State","Ag District Code","Year"])
features_ad = features_ad.merge(enso_year, on=["State","Ag District Code","Year"], how="left")

# --- Build a modeling table at Ag District by Year (one row per district-year ---
# Start from your aggregated yields (agg) and join features
df_model = (
    agg[["Year", "State", "Ag District Code", "Yield_bu_per_acre", "Acres_Planted"]]
    .merge(features_ad, on=["Year","State","Ag District Code"], how="left")
)
#df_model.drop("n_stations")
# Save the features and model-ready table
features_out = "weather_features_GDD_ExtremeHeat_DrySpell.csv"
df_model_out = "corn_yield_weather_model_ADlevel.csv" #this is the final model that is useful for predictions for blind guesses
features_ad.to_csv(features_out, index=False)
df_model.to_csv(df_model_out, index=False)

annual_prices =(
    pivot.groupby(["State", "Year"], as_index = False)[["Corn $/bu", "Soybeans $/bu"]]
                 .mean()
                 .rename(columns={
                     "Corn $/bu": "Corn_price_bu_annual",
                     "Soybeans $/bu": "Soybeans_price_bu_annual"}))
df_model_prices = df_model.merge(annual_prices, on=["State", "Year"], how="left", validate ="m:1" )
'''
 this includes the prices that can be used for pricing models
 later on i decided to just use this for the yield model as well
 '''
df_model_prices_out = "corn_yield_weather_model_ADlevel_prices.csv" 
df_model_prices.to_csv(df_model_prices_out, index=False)

print(f"[saved] Features (Ag District by Year): {features_out}")
print(f"[saved] Model table (Ag District by Year): {df_model_out}")

#%%
''' this is the data set that is used for the EDA visuals that were used for presentation 3
    the daily csv was used for the visualizations since it was more accurate than the aggregated data
    aggreagted data was used for the modeling
'''
#graph of missingness
df_final_no_snow = df_final.drop(columns =["SNOW (Inches)", "SNWD (Inches)"]) 
miss = (df_final_no_snow.isna().mean() * 100).sort_values(ascending=False) 
miss_df = miss.reset_index()
miss_df.columns = ["Column", "Percent Missing"]

plt.figure(figsize=(12, 6))
plt.bar(miss_df["Column"], miss_df["Percent Missing"], color = "green") 
plt.title("Percent Missing by Column") 
plt.ylabel("Percent (%)") 
plt.xticks(rotation=60, ha="right") 
plt.tight_layout() 
plt.show()

#%% save to csv 

out_path = "corn_weather_merged_district_daily_new.csv" 
df_final_no_snow.to_csv(out_path, index=False) 
out_path




