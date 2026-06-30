# Timeseries Project Jun 2026

Time series forecasting project using statistical models (ARIMA, SARIMA), feature engineering, hyperparameter tuning with Hyperopt, experiment tracking with MLflow, and a Streamlit dashboard.

## Setup
git clone https://github.com/valiyuneski/Time_Series_Project_No_Prophet.git

```bash
cd Time_Series_Project_No_Prophet
python3.13 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

Short presentation script for each notebook:

Week_1_eda.ipynb — Basic data cleaning and exploration. We removed outliers, filled missing values, checked stationarity (it's non-stationary), and saved a clean dataset for modeling.

Week_1_advanced_eda.ipynb — Deep dive into external drivers. Weekends boost sales 15–25%, holidays like Christmas and Black Friday spike demand, and oil prices have a weak negative correlation.

Week_2_statistical_models.ipynb — Classical time series models. We tried ARIMA, SARIMA, and Holt-Winters. Holt-Winters performed best with RMSE ~118 and R² 0.48, capturing weekly patterns well.

Week_2_feature_engineering_models.ipynb — Machine learning with engineered features. We built calendar features, oil price lags, sales lags, rolling averages, and holiday flags, then trained Linear Regression, Random Forest, and XGBoost — expecting XGBoost to win.

Week_3_Hyperopt.ipynb — Bayesian hyperparameter tuning for XGBoost. We used TPE to search 7 parameters over 50 iterations, minimizing RMSE, then trained a final optimized model.

Week_3_MLFlow.ipynb — Experiment tracking with MLflow. We logged parameters, metrics, and models for Linear Regression and Ridge, then showed how to load models back and launch the MLflow UI.

Week_4_Streamlit.py — Interactive dashboard that loads the feature-engineered data, retrains XGBoost (our best Week 2 model), and displays test-set metrics (MSE, RMSE, MAE, R²), model hyperparameters, and an interactive actual-vs-forecast plot with a date range selector.