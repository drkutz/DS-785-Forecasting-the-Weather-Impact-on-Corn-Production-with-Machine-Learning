


import os
import sys
import math
import json
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.utils import resample
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GroupKFold, KFold, RandomizedSearchCV, GridSearchCV, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.nonparametric.smoothers_lowess import lowess
from xgboost import XGBRegressor
XGB_AVAILABLE = True



warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.dpi"] = 120

# --------------- Config ---------------
DATA_PATH = "corn_yield_weather_model_ADlevel_prices.csv"  # only use corn at district level for this model
TARGET = "Yield_bu_per_acre"
RANDOM_STATE = 42
N_JOBS = -1
OUTPUT_DIR = "corn_yield_model_outputs"
EXCLUDE_FROM_PREDICTORS =["Soybeans_price_bu_annual", "Corn_price_bu_annual"]


os.makedirs(OUTPUT_DIR, exist_ok=True)

def timestamp() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_fig(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"[Saved] {path}")

def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file at: {path}")
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]:,} rows x {df.shape[1]:,} cols")
    return df

def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols) for modeling (excluding TARGET and excluded predictors)."""
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found in data.")

    # Separate by dtype first
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Remove target from features if present
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)
    if TARGET in categorical_cols:
        categorical_cols.remove(TARGET)

    #remove other columns if needed such as soybean price. can be used for anyone
    for c in EXCLUDE_FROM_PREDICTORS:
        if c in numeric_cols:
            numeric_cols.remove(c)
        if c in categorical_cols:
            categorical_cols.remove(c)

   
    likely_ids = ["state", "State", "ag_district", "AgDistrict", "district", "District"]
    for c in likely_ids:
        if c in df.columns and c not in categorical_cols and not np.issubdtype(df[c].dtype, np.number):
            categorical_cols.append(c)

    
    for c in df.columns:
        if c.lower() in ["enso", "elnino", "la_nina", "phase"]:
            if df[c].dtype == object and c not in categorical_cols:
                categorical_cols.append(c)
            

   
    numeric_cols = [c for c in numeric_cols if c != TARGET]
    categorical_cols = list(dict.fromkeys(categorical_cols))  
    return numeric_cols, categorical_cols


def train_test_split_by_year(df: pd.DataFrame, test_years: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the most recent years for testing. If test_years is None, use ~20% of unique years (min 3)."""
    if "Year" not in df.columns:
        
        print("Column 'year' not found — using random 80/20 split.")
        return train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    years = sorted(df["Year"].dropna().unique())
    if len(years) < 6:
        # too few unique years to time-split
        print("Few unique years — using random 80/20 split.")
        return train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    if test_years is None:
        test_years = max(3, int(round(0.2 * len(years))))

    split_years = years[-test_years:]
    mask_test = df["Year"].isin(split_years)
    df_train = df[~mask_test].copy()
    df_test  = df[mask_test].copy()
    print(f"Time-based split: train years = {years[0]}–{split_years[0]-1 if len(split_years)>0 else years[-1]}, "
          f"test years = {split_years[0]}–{split_years[-1]} "
          f"({df_train.shape[0]:,} train rows, {df_test.shape[0]:,} test rows)")
    return df_train, df_test

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], scale_numeric: bool) -> ColumnTransformer:
    """Create a ColumnTransformer for preprocessing."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()) if scale_numeric else ("passthrough", "passthrough")
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor

def evaluate_model(name: str, model, X_train, y_train, X_test, y_test, groups=None, cv_splits=5) -> Dict[str, Any]:
    """Cross-validate, fit, and evaluate on holdout. Returns metrics dict."""
    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error"
    }

    if groups is not None and cv_splits >= 3 and len(np.unique(groups)) >= cv_splits:
        cv = GroupKFold(n_splits=cv_splits)
        cv_res = cross_validate(model, X_train, y_train, scoring=scoring, cv=cv, groups=groups, n_jobs=N_JOBS)
    else:
        cv = KFold(n_splits=min(cv_splits, 5), shuffle=True, random_state=RANDOM_STATE)
        cv_res = cross_validate(model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=N_JOBS)

    # Fit and test
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "model": name,
        "cv_r2_mean": np.mean(cv_res["test_r2"]),
        "cv_rmse_mean": -np.mean(cv_res["test_rmse"]),
        "cv_mae_mean": -np.mean(cv_res["test_mae"]),
        "test_r2": r2_score(y_test, pred),
        "test_rmse": rmse(y_test, pred),
        "test_mae": mean_absolute_error(y_test, pred)
    }
    return metrics, model, pred
#-------------create the graphs--------------------
def feature_importance_bar(names: List[str], importances: np.ndarray, title: str, top_n: int = 20):
    idx = np.argsort(importances)[::-1][:top_n]
    imp = importances[idx]
    nms = [names[i] for i in idx]

    fig = plt.figure()
    plt.barh(range(len(imp)), imp[::-1])
    plt.yticks(range(len(imp)), nms[::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    return fig

def linear_coeff_bar(names: List[str], coefs: np.ndarray, title: str, top_n: int = 20):
    abs_idx = np.argsort(np.abs(coefs))[::-1][:top_n]
    vals = coefs[abs_idx]
    nms  = [names[i] for i in abs_idx]

    fig = plt.figure()
    plt.barh(range(len(vals)), vals[::-1])
    plt.yticks(range(len(vals)), nms[::-1])
    plt.xlabel("Coefficient")
    plt.title(title)
    plt.tight_layout()
    return fig

def predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    fig = plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.6)
    minv = min(np.min(y_true), np.min(y_pred))
    maxv = max(np.max(y_true), np.max(y_pred))
    plt.plot([minv, maxv], [minv, maxv], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    return fig

def residuals_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    resid = y_pred - y_true
    fig = plt.figure()
    plt.scatter(y_pred, resid, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Pred - True)")
    plt.title(title)
    plt.tight_layout()
    return fig
#---------------create the main script for modeling-------------------
def main():
    start = time.time()
    df = load_data(DATA_PATH)
    #df = df.drop("n_stations")
    if "n_stations" in df.columns:
        df.drop(columns=["n_stations"], inplace=True)
        print("Dropped column: n_stations")
    # Basic checks
    if TARGET not in df.columns:
        raise KeyError(f"Target '{TARGET}' not found in columns: {list(df.columns)}")

    # Keep a copy of id columns for error analysis (optional)
    id_cols = [c for c in ["state", "ag_district", "year"] if c in df.columns]

    # Determine columns
    numeric_cols, categorical_cols = detect_columns(df)
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols[:12]}{'...' if len(numeric_cols)>12 else ''}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    # Split
    df_train, df_test = train_test_split_by_year(df, test_years=None)
    y_train = df_train[TARGET].values
    y_test  = df_test[TARGET].values

    # Preprocessors
    preproc_scaled   = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)
    preproc_unscaled = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)

    drop_cols = [TARGET] + [c for c in EXCLUDE_FROM_PREDICTORS if c in df_train.columns]

    X_train = df_train.drop(columns=drop_cols, errors="ignore")
    X_test  = df_test.drop(columns=drop_cols, errors="ignore")

    # GroupKFold by year for CV
    groups = df_train["Year"].values if "Year" in df_train.columns else None

    results = []  # store metrics
    trained = {}  
    preds = {}    # store predictions

    # ---------------- Linear Regression (Baseline) ----------------
    lin_pipeline = Pipeline(steps=[
        ("preprocess", preproc_scaled),
        ("model", LinearRegression(n_jobs=None if "n_jobs" not in LinearRegression().get_params() else N_JOBS))
    ])
    m_lin, fitted_lin, pred_lin = evaluate_model("LinearRegression", lin_pipeline, X_train, y_train, X_test, y_test, groups)
    results.append(m_lin); trained["LinearRegression"] = fitted_lin; preds["LinearRegression"] = pred_lin

    # ---------------- Ridge Regression (tuned) ----------------
    ridge_pipeline = Pipeline(steps=[
        ("preprocess", preproc_scaled),
        ("model", Ridge(random_state=RANDOM_STATE))
    ])
    ridge_params = {"model__alpha": np.logspace(-3, 3, 21)}
    # Grid search with GroupKFold
    if groups is not None and len(np.unique(groups)) >= 5:
        cv = GroupKFold(n_splits=5)
        ridge_search = GridSearchCV(ridge_pipeline, ridge_params, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=N_JOBS)
        ridge_search.fit(X_train, y_train, **({"groups": groups} if groups is not None else {}))
    else:
        ridge_search = GridSearchCV(ridge_pipeline, ridge_params, scoring="neg_root_mean_squared_error", cv=5, n_jobs=N_JOBS)
        ridge_search.fit(X_train, y_train)
    best_ridge = ridge_search.best_estimator_
    pred_ridge = best_ridge.predict(X_test)
    m_ridge = {
        "model": "Ridge(best)",
        "cv_r2_mean": np.nan,  
        "cv_rmse_mean": -ridge_search.best_score_,
        "cv_mae_mean": np.nan,
        "test_r2": r2_score(y_test, pred_ridge),
        "test_rmse": rmse(y_test, pred_ridge),
        "test_mae": mean_absolute_error(y_test, pred_ridge)
    }
    results.append(m_ridge); trained["Ridge"] = best_ridge; preds["Ridge"] = pred_ridge

    # ---------------- Random Forest (tuned) ----------------
    from sklearn.ensemble import RandomForestRegressor
    rf_pipeline = Pipeline(steps=[
        ("preprocess", preproc_unscaled),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=N_JOBS))
    ])
    rf_param_dist = {
        "model__n_estimators": [200, 300, 400, 600, 800, 1000],
        "model__max_depth": [None, 6, 8, 12, 16, 20, 30],
        "model__min_samples_leaf": [1, 2, 3, 5],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["auto", "sqrt", "log2"]
    }
    if groups is not None and len(np.unique(groups)) >= 5:
        cv = GroupKFold(n_splits=5)
        rf_search = RandomizedSearchCV(
            rf_pipeline, rf_param_dist, n_iter=30, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=N_JOBS, verbose=1
        )
        rf_search.fit(X_train, y_train, **({"groups": groups} if groups is not None else {}))
    else:
        rf_search = RandomizedSearchCV(
            rf_pipeline, rf_param_dist, n_iter=30, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", cv=5, n_jobs=N_JOBS, verbose=1
        )
        rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    pred_rf = best_rf.predict(X_test)
    m_rf = {
        "model": "RandomForest(best)",
        "cv_r2_mean": np.nan,
        "cv_rmse_mean": -rf_search.best_score_,
        "cv_mae_mean": np.nan,
        "test_r2": r2_score(y_test, pred_rf),
        "test_rmse": rmse(y_test, pred_rf),
        "test_mae": mean_absolute_error(y_test, pred_rf)
    }
    results.append(m_rf); trained["RandomForest"] = best_rf; preds["RandomForest"] = pred_rf

    # ---------------- XGBoost (tuned) ----------------
    if XGB_AVAILABLE:
        xgb_pipeline = Pipeline(steps=[
            ("preprocess", preproc_unscaled),
            ("model", XGBRegressor(
                random_state=RANDOM_STATE, n_estimators=400, objective="reg:squarederror",
                n_jobs=N_JOBS, tree_method="hist"
            ))
        ])
        xgb_param_dist = {
            "model__n_estimators": [300, 400, 600, 800],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
            "model__min_child_weight": [1, 3, 5]
        }
        if groups is not None and len(np.unique(groups)) >= 5:
            cv = GroupKFold(n_splits=5)
            xgb_search = RandomizedSearchCV(
                xgb_pipeline, xgb_param_dist, n_iter=25, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=N_JOBS, verbose=1
            )
            xgb_search.fit(X_train, y_train, **({"groups": groups} if groups is not None else {}))
        else:
            xgb_search = RandomizedSearchCV(
                xgb_pipeline, xgb_param_dist, n_iter=25, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", cv=5, n_jobs=N_JOBS, verbose=1
            )
            xgb_search.fit(X_train, y_train)
        best_xgb = xgb_search.best_estimator_
        pred_xgb = best_xgb.predict(X_test)
        m_xgb = {
            "model": "XGBoost(best)",
            "cv_r2_mean": np.nan,
            "cv_rmse_mean": -xgb_search.best_score_,
            "cv_mae_mean": np.nan,
            "test_r2": r2_score(y_test, pred_xgb),
            "test_rmse": rmse(y_test, pred_xgb),
            "test_mae": mean_absolute_error(y_test, pred_xgb)
        }
        results.append(m_xgb); trained["XGBoost"] = best_xgb; preds["XGBoost"] = pred_xgb
    else:
        print("XGBoost not available; skipping XGB model")

 #----------------create additional graphs that were neeeded to understand the Ridge model -------------------
    
        # ---------------- Scatter plot: Corn Yield vs Temperature ----------------
    if "TMAX_mean" in df.columns and "Yield_bu_per_acre" in df.columns:
        fig = plt.figure()
        plt.scatter(df["TMAX_mean"], df["Yield_bu_per_acre"], alpha=0.5)
        plt.xlabel("Average Max Temperature (°F)")
        plt.ylabel("Corn Yield (bu/acre)")
        plt.title("Corn Yield vs TMAX_mean")
        plt.tight_layout()
        save_fig(fig, "scatter_yield_vs_tmax.png")
    #tmin vs corn yield
    if "TMIN_mean" in df.columns and "Yield_bu_per_acre" in df.columns:
        fig = plt.figure()
        plt.scatter(df["TMIN_mean"], df["Yield_bu_per_acre"], alpha=0.5)
        plt.xlabel("Average Min Temperature (°F)")
        plt.ylabel("Corn Yield (bu/acre)")
        plt.title("Corn Yield vs TMIN_mean")
        plt.tight_layout()
        save_fig(fig, "scatter_yield_vs_tmin.png")
    
    # created column of Temperture mean 
    if "TMAX_mean" in df.columns and "TMIN_mean" in df.columns:
        df["TMEAN_mean"] = (df["TMAX_mean"] + df["TMIN_mean"]) / 2
        fig = plt.figure()
        plt.scatter(df["TMEAN_mean"], df["Yield_bu_per_acre"], alpha=0.5)
        plt.xlabel("Mean Temperature (°F)")
        plt.ylabel("Corn Yield (bu/acre)")
        plt.title("Corn Yield vs TMEAN_mean")
        plt.tight_layout()
        save_fig(fig, "scatter_yield_vs_tmean.png")
        
    if "ExtremeHeatDays_mean" in df.columns and "Yield_bu_per_acre" in df.columns:
        fig = plt.figure()
        plt.scatter(df["ExtremeHeatDays_mean"], df["Yield_bu_per_acre"], alpha=0.5)
        lowess_smoothed = lowess(
            df["Yield_bu_per_acre"],
        df["ExtremeHeatDays_mean"],
        frac=0.15)
        plt.plot(
        lowess_smoothed[:, 0],
        lowess_smoothed[:, 1],
        color="red",
        linewidth=3,
        label="LOWESS trend")
        plt.xlabel("ExtremeHeatDays")
        plt.ylabel("Corn Yield (bu/acre)")
        plt.title("Corn Yield vs ExtremeHeatDays")
        plt.legend()
        plt.tight_layout()
        save_fig(fig, "scatter_yield_vs_heat.png")
        
    if "MaxDrySpellDays_mean" in df.columns and "Yield_bu_per_acre" in df.columns:
        fig = plt.figure()
        plt.scatter(df["MaxDrySpellDays_mean"], df["Yield_bu_per_acre"], alpha=0.5)
        plt.xlabel("MaxDrySpellDays")
        plt.ylabel("Corn Yield (bu/acre)")
        plt.title("Corn Yield vs Max Dry Spell Days")
        plt.tight_layout()
        save_fig(fig, "scatter_yield_vs_dryspell.png")


    # ---------------- Results Table ----------------
    res_df = pd.DataFrame(results)
    res_path_csv = os.path.join(OUTPUT_DIR, f"model_results_{timestamp()}.csv")
    res_df.to_csv(res_path_csv, index=False)
    print("\n=== Model Comparison (test set) ===")
    print(res_df[["model","test_r2","test_rmse","test_mae"]].sort_values("test_rmse"))
    print(f"[Saved] {res_path_csv}")

    # ---------------- Plots ----------------
    # Pred vs Actual + Residuals for best 
    best_row = res_df.iloc[res_df["test_rmse"].values.argmin()]
    best_name = best_row["model"]
    y_pred_best = preds[best_name if best_name in preds else list(preds.keys())[0]]

    fig1 = predicted_vs_actual(y_test, y_pred_best, f"Predicted vs Actual — {best_name}")
    save_fig(fig1, f"pred_vs_actual_{best_name.replace(' ','_')}.png")

    fig2 = residuals_plot(y_test, y_pred_best, f"Residuals — {best_name}")
    save_fig(fig2, f"residuals_{best_name.replace(' ','_')}.png")

        
        # Feature importance-style plot for Ridge 
    if "Ridge" in trained:
        ridge_pipe = trained["Ridge"]
        pre = ridge_pipe.named_steps["preprocess"]

        # Get feature names after preprocessing 
        num_features = pre.transformers_[0][2]
        cat_features = pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(
            pre.transformers_[1][2]
        )
        feature_names = list(num_features) + list(cat_features)

        ridge_model = ridge_pipe.named_steps["model"]
        coefs = np.array(ridge_model.coef_).ravel()

        # importance score 
        ridge_importance = np.abs(coefs)

        fig_ridge = feature_importance_bar(
            feature_names,
            ridge_importance,
            "Feature Importance - Ridge for Corn Yield coefficients",
            top_n=25,
        )
        save_fig(fig_ridge, "feature_importance_ridge.png")

    # Coefficients for Linear/Ridge models. 

    def get_feature_names_from_preprocessor(pre: ColumnTransformer) -> List[str]:
        num_features = pre.transformers_[0][2]
        cat_features = pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(pre.transformers_[1][2])
        return list(num_features) + list(cat_features)

    if "LinearRegression" in trained:
        lin_pipe = trained["LinearRegression"]
        pre = lin_pipe.named_steps["preprocess"]
        names = get_feature_names_from_preprocessor(pre)
        coefs = lin_pipe.named_steps["model"].coef_
        fig5 = linear_coeff_bar(names, np.array(coefs).ravel(), "Linear Regression Coefficients", top_n=25)
        save_fig(fig5, "coefficients_linear.png")

    if "Ridge" in trained:
        ridge_pipe = trained["Ridge"]
        pre = ridge_pipe.named_steps["preprocess"]
        names = get_feature_names_from_preprocessor(pre)
        coefs = ridge_pipe.named_steps["model"].coef_
        fig6 = linear_coeff_bar(names, np.array(coefs).ravel(), "Ridge Coefficients", top_n=25)
        save_fig(fig6, "coefficients_ridge.png")


    best_preds = y_pred_best
    err_df = df_test[id_cols].copy() if len(id_cols) else pd.DataFrame(index=df_test.index)
    err_df["actual"] = y_test
    err_df["predicted"] = best_preds
    err_df["residual"] = best_preds - y_test
    err_path = os.path.join(OUTPUT_DIR, f"errors_by_row_{best_name.replace(' ','_')}_{timestamp()}.csv")
    err_df.to_csv(err_path, index=False)
    print(f"[Saved] {err_path}")


    elapsed = time.time() - start
    print(f"\nDone {elapsed:.1f}s. Outputs in '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()




