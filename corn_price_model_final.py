

# -*- coding: utf-8 -*-
"""
Corn price modeling pipeline at Ag District level. It can only use aggreagted data 

This script mirrors the style of corn_yield_modeling_pipeline_final.py:

 Load the modeling table with yield, weather, and price features.
 Select features for corn price.
 Perform a time-aware train/test split (sorted by Year, 80/20).
 Build pipelines with shared preprocessing.
 Tune ALL models (Ridge RandomForest, XGBoost) via CV.
   - LinearRegression is used as a baseline
 Evaluate on the test set and save:


Run this as a stand-alone script after creating `corn_yield_weather_model_ADlevel_prices.csv using the weather joiner script. 
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# XGBoost is optional – script will still run if it's not installed not sure on system on 
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

#------------load the data and create the outputs 

DATA_PATH = "corn_yield_weather_model_ADlevel_prices.csv"
TARGET = "Corn_price_bu_annual"

# ---------features for price modeling 
BASE_FEATURES = [
    "Yield_bu_per_acre",
    "GDD_total_mean",
    "ExtremeHeatDays_mean",
    "MaxDrySpellDays_mean",
    "PRCP_total_mean",
    "TMAX_mean",
    "TMIN_mean",
    "ENSO",
    "Acres_Planted",
    "Soybeans_price_bu_annual",
    "Year",
]


OUTPUT_DIR = Path("corn_price_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_JOBS = -1 

#-------------helper functions--------------- 

def timestamp() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def load_data(path: str) -> pd.DataFrame:
    """Load the modeling table and ensure standardization"""
    df = pd.read_csv(path)
    # Ensure Year is numeric and sorted
    if "Year" not in df.columns:
        raise KeyError("Expected 'Year' column in the modeling table.")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=[TARGET, "Year"]).copy()
    df = df.sort_values(["Year", "State"]).reset_index(drop=True)
    return df


def prepare_features(df: pd.DataFrame):
    """Select feature matrix X and target y for modeling."""
    feature_cols = [c for c in BASE_FEATURES if c in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns found in dataframe.")
    X = df[feature_cols].copy()
    y = df[TARGET].astype(float).values
    return X, y, feature_cols


def time_based_train_test_split(X, y, years, test_size=0.2):
    """
    Time-aware train/test split: sort by Year and take last `test_size` for test.
    This mimics a realistic forecasting scenario.
    """
    order = np.argsort(years)
    X_sorted = X.iloc[order]
    y_sorted = y[order]
    years_sorted = years[order]

    cut = int((1.0 - test_size) * len(X_sorted))
    X_train = X_sorted.iloc[:cut]
    X_test = X_sorted.iloc[cut:]
    y_train = y_sorted[:cut]
    y_test = y_sorted[cut:]
    years_train = years_sorted[:cut]
    years_test = years_sorted[cut:]

    return X_train, X_test, y_train, y_test, years_train, years_test


def build_preprocessor(numeric_cols):
    """
    Build a ColumnTransformer that imputes and scales numeric features.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
        ],
        remainder="drop",
    )
    return preproc


def compute_metrics(y_true, y_pred):
    """Return R^2, MAE, and RMSE. These are the main ways to meausure success"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return r2, mae, rmse



# -----------------MODEL TRAINING & TUNING------------------


def run_linear_baseline(preproc, X_train, y_train, X_test, y_test):
    """
    Fit a LinearRegression baseline with CV no other parameters
    """
    pipe = Pipeline(
        steps=[
            ("preprocess", preproc),
            ("model", LinearRegression()),
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=N_JOBS,
    )
    cv_rmse_mean = -cv_scores.mean()
    cv_rmse_std = cv_scores.std()

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    r2, mae, rmse = compute_metrics(y_test, preds)

    result = {
        "Model": "LinearRegression",
        "Best_Params": {},
        "CV_RMSE_mean": cv_rmse_mean,
        "CV_RMSE_std": cv_rmse_std,
        "Test_R2": r2,
        "Test_MAE": mae,
        "Test_RMSE": rmse,
    }
    return pipe, preds, result


def tune_with_gridsearch(name, preproc, model, param_grid, X_train, y_train):
    """
    GridSearchCV wrapper for Ridge
    """
    pipe = Pipeline(
        steps=[
            ("preprocess", preproc),
            ("model", model),
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=N_JOBS,
    )
    print(f"[info] Tuning {name} with GridSearchCV...")
    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    best_params = grid.best_params_
    cv_rmse_mean = -grid.best_score_
    cv_rmse_std = grid.cv_results_["std_test_score"][grid.best_index_]

    return best_pipe, best_params, cv_rmse_mean, cv_rmse_std


def tune_with_randomsearch(name, preproc, model, param_dist, X_train, y_train, n_iter=40):
    """
    RandomizedSearchCV wrapper for RandomForest & XGBoost.  needded to give them best results possible 
    """
    pipe = Pipeline(
        steps=[
            ("preprocess", preproc),
            ("model", model),
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rnd = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    rnd.fit(X_train, y_train)

    best_pipe = rnd.best_estimator_
    best_params = rnd.best_params_
    cv_rmse_mean = -rnd.best_score_
    cv_rmse_std = rnd.cv_results_["std_test_score"][rnd.best_index_]

    return best_pipe, best_params, cv_rmse_mean, cv_rmse_std


# -----------------main function for modeling  --------------------


def main():
    
    df = load_data(DATA_PATH)
    print(f" Data shape: {df.shape}")

    
    X, y, feature_cols = prepare_features(df)
    years = df["Year"].values

    
    X_train, X_test, y_train, y_test, years_train, years_test = time_based_train_test_split(
        X, y, years, test_size=0.2
    )

    preproc = build_preprocessor(feature_cols)

    results = []
    best_model = None
    best_preds = None
    best_name = None
    best_rmse = np.inf

    # ---------------- Linear Regression (baseline model for comparision ----------------
    
    lin_pipe, lin_preds, lin_res = run_linear_baseline(
        preproc, X_train, y_train, X_test, y_test
    )
    results.append(lin_res)
    best_model = lin_pipe
    best_preds = lin_preds
    best_name = "LinearRegression"
    best_rmse = lin_res["Test_RMSE"]

    # ---------------- Ridge Regression ----------------This was the best model
    
    ridge_params = {"model__alpha": np.logspace(-3, 3, 21)}
    ridge_model = Ridge(random_state=RANDOM_STATE)
    ridge_pipe, ridge_best_params, ridge_cv_rmse_mean, ridge_cv_rmse_std = tune_with_gridsearch(
        "Ridge", preproc, ridge_model, ridge_params, X_train, y_train
    )
    ridge_preds = ridge_pipe.predict(X_test)
    ridge_r2, ridge_mae, ridge_rmse = compute_metrics(y_test, ridge_preds)
    results.append(
        {
            "Model": "Ridge",
            "Best_Params": ridge_best_params,
            "CV_RMSE_mean": ridge_cv_rmse_mean,
            "CV_RMSE_std": ridge_cv_rmse_std,
            "Test_R2": ridge_r2,
            "Test_MAE": ridge_mae,
            "Test_RMSE": ridge_rmse,
        }
    )
    if ridge_rmse < best_rmse:
        best_model = ridge_pipe
        best_preds = ridge_preds
        best_name = "Ridge"
        best_rmse = ridge_rmse


    # ---------------- Random Forest ----------------These models struggled 
    
    rf_model = RandomForestRegressor(random_state=RANDOM_STATE)
    rf_param_dist = {
        "model__n_estimators": [200, 400, 600, 800],
        "model__max_depth": [None, 4, 6, 8, 10],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5, 0.7],
    }
    rf_pipe, rf_best_params, rf_cv_rmse_mean, rf_cv_rmse_std = tune_with_randomsearch(
        "RandomForest", preproc, rf_model, rf_param_dist, X_train, y_train, n_iter=50
    )
    rf_preds = rf_pipe.predict(X_test)
    rf_r2, rf_mae, rf_rmse = compute_metrics(y_test, rf_preds)
    results.append(
        {
            "Model": "RandomForest",
            "Best_Params": rf_best_params,
            "CV_RMSE_mean": rf_cv_rmse_mean,
            "CV_RMSE_std": rf_cv_rmse_std,
            "Test_R2": rf_r2,
            "Test_MAE": rf_mae,
            "Test_RMSE": rf_rmse,
        }
    )
    if rf_rmse < best_rmse:
        best_model = rf_pipe
        best_preds = rf_preds
        best_name = "RandomForest"
        best_rmse = rf_rmse

    # ---------------- XGBoost ----------------This model struggled 
    if XGB_AVAILABLE:
        
        xgb_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=N_JOBS,
        )
        xgb_param_dist = {
            "model__n_estimators": [300, 400, 600, 800],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "model__reg_lambda": [0, 1, 5, 10],
        }
        xgb_pipe, xgb_best_params, xgb_cv_rmse_mean, xgb_cv_rmse_std = tune_with_randomsearch(
            "XGBoost", preproc, xgb_model, xgb_param_dist, X_train, y_train, n_iter=60
        )
        xgb_preds = xgb_pipe.predict(X_test)
        xgb_r2, xgb_mae, xgb_rmse = compute_metrics(y_test, xgb_preds)
        results.append(
            {
                "Model": "XGBoost",
                "Best_Params": xgb_best_params,
                "CV_RMSE_mean": xgb_cv_rmse_mean,
                "CV_RMSE_std": xgb_cv_rmse_std,
                "Test_R2": xgb_r2,
                "Test_MAE": xgb_mae,
                "Test_RMSE": xgb_rmse,
            }
        )
        if xgb_rmse < best_rmse:
            best_model = xgb_pipe
            best_preds = xgb_preds
            best_name = "XGBoost"
            best_rmse = xgb_rmse
    else:
        print("[info] XGBoost not installed; skipping XGBRegressor.")

    # -------------------save results to local folder 
    
    
    results_df = pd.DataFrame(results)
    # best_params is a dict; convert to string for CSV
    results_df["Best_Params"] = results_df["Best_Params"].astype(str)
    results_path = OUTPUT_DIR / "corn_price_model_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"[saved] {results_path}")

    

    #-------- Save predictions for the best model
    preds_df = pd.DataFrame(
        {
            "Year": years_test,
            "State": df.loc[X_test.index, "State"].values
            if "State" in df.columns
            else np.nan,
            "Actual_Corn_price_bu_annual": y_test,
            "Predicted_Corn_price_bu_annual": best_preds,
            "Model": best_name,
        }
    )
    preds_path = OUTPUT_DIR / "corn_price_best_model_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"saved {preds_path}")
    print("Corn price modeling pipeline complete.")


if __name__ == "__main__":
    main()

