Read Me for DS 785 project 
Forecasting the Weather Impact on Corn Production with Machine Learning 

By Adam Gruber 
MS in Data Science UW Green Bay 

------------------Key Findings-----------------
Ridge model performed best for both Corn yield and Corn prices
Ridge explained 30% of variation in Yield model with a MAE of 25 bushels per acre
Ridge explained 84% of the variation in the price model with a MAE of 41 cents 

----------------Data Sources--------------------
USDA quick stats for Raw Price, and corn yield by AG district ----->corn yield and acres inputs 
NOAA weather data by station and state ------> in state labeled folder with raw weather data 
USDA quick stats for commodity prices --------> Prices folder by state
ENSO data from NOAA ------> ENSO.csv in main file

No external databases with an API are currently in use. 
– Code Overview

1. ----combine_weather_multi_final.py------

Reads thousands of weather CSV files per state

Cleans and merges into single state-level datasets

Generates long-format daily weather tables

2. ----corn_yields_acres_combiner_final.py------

Ingests raw yield + acres files

Removes inconsistencies, standardizes naming

Produces a combined Ag-District-level table

3. ------weather_corn_joiner.py-------

Merges:

Weather

Yield

Acres planted

ENSO classification

Corn & soybean prices

Produces the final modeling dataset used in ML pipelines

4. ------corn_yield_modeling_pipeline_final.py-------

Trains and evaluates multiple yield prediction models:

Linear Regression

Ridge

Random Forest

XGBoost

Outputs: R², MAE, RMSE

Scatterplots + LOWESS curves

Feature importance charts

Diagnostic plots

5. ----corn_price_model_final.py-------

Predicts annual corn price

Uses yield, ENSO, soybean price, and climate variables

Compares:

Linear

Ridge

Random Forest

XGBoost

6. eda_make_visuals.py

Creates: Visuals used for presentation 3 

Scatterplots (weather → yield)

LOWESS smoothed curves

Correlation heatmaps

Distribution plots

Time trend graphs


How to run:

git clone https://github.com/drkutz/DS-785-Forecasting-the-Weather-Impact-on-Corn-Production-with-Machine-Learning cd DS-785-Forecasting-the-Weather-Impact-on-Corn-Production-with-Machine-Learning

may need to install python libraries

pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, statsmodels

Run programs in the following order 

Combine_weather_multi_final.py ---------build weather files
corn_yields_acres_combiner_final.py -------build corn yield data by state
weather_corn_joiner.py -------combine the corn, weather, prices, and ENSO

Run the models 
corn_yield_modeling_pipeline_final.py
corn_price_model_final.py

If needed visualizations file
eda_make_visuals.py 




