# Timeseries Project Jun 2026


## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

> **Note:** All dependencies (including `matplotlib>=3.11.0`) are listed in `requirements.txt` and will be installed into the virtual environment. Make sure to **activate the `.venv`** before running any scripts:
>
> ```bash
> source .venv/bin/activate
> streamlit run Week_4_Streamlit.py
> ```
>
> Running `python3` without activating the virtual environment will use the system interpreter, which may not have the required packages installed.

## Week 1: IDE (Integrated Development Environment) Introduction and EDA 
Basic EDA
Import data
Outlier detection
Handling missing values
Seasonal decomposition
Stationarity analysis
Autocorrelation
Save cleaned data

## Week 2: Model Training and Forecasting
During this week, we will train several forecasting models using the time series data from 01.01.2013 to 31.12.2013. We will then evaluate the models by generating forecasts for January–March 2014 and calculating performance metrics to compare their accuracy.
We will explore two main approaches:
Statistical models (e.g., ARIMA, Exponential Smoothing, etc.)
Feature engineering + machine learning models
For the second approach, we will transform the time series into a structured dataframe with multiple columns (features). This will allow us to apply standard regression-based machine learning models of our choice.
In the live session
Statistical model approach
Use the cleaned dataset from Week 1
Prepare and adapt the data for statistical forecasting models
Train and evaluate models such as ARIMA and Exponential Smoothing
Feature engineering approach
Learn how to create a structured dataframe that captures time series information through engineered features
Build basic time-based features (lags, rolling statistics, etc.)
Convert the time series into a feature-based dataset
Apply regression models such as Linear Regression, Random Forest, XGBoost, etc., to generate predictions
Your Task
Statistical models
Train and test additional statistical models
Don’t worry about hyperparameter tuning yet (we will cover that next week)
Focus on learning how to generate forecasts using different approaches
Don’t stress if the forecasts are not very accurate at this stage
Steps
Create a new notebook called statistical_models.ipynb and import the timeseries_cleaned.csv data.
Pick a statistical model (ARIMA, SARIMA, Exponential Smoothing, etc.).
Train your model with data from 01.01.2013 – 31.12.2013.
Test the forecast of your model with the data from 01.01.2014 – 31.03.2014 and calculate a performance metric (MSE, R², MAE, etc.).
Repeat steps 2–4 with at least 2 more models (minimum 3 in total).
Feature engineering
Go beyond the basic features we created in class
Generate more interesting and creative features
Incorporate information from other tables (e.g., oil prices, holidays, etc.)
Extract additional signals directly from the time series
Steps
Create a new notebook called feature_engineering_models.ipynb and import the time series data.
Generate a dataframe using the feature engineering approach. In the live session we saw basic features—here, try to build more interesting and complex features (exogenous and endogenous).
Choose a regression model (XGBoost, LinearRegression, RandomForest, etc.).
Train your model with data from 01.01.2013 – 31.12.2013.
Test the forecast of your model with the data from 01.01.2014 – 31.03.2014 and calculate a performance metric (MSE, R², MAE, etc.).
Repeat steps 3–5 with at least 2 more models (minimum 3 in total).