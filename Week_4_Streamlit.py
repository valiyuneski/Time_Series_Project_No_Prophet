"""
Week 4: Streamlit Dashboard for Time Series Forecasting
=======================================================
Displays information about the best model from Week 2 (XGBoost)
and allows date-based selection to filter predictions vs actuals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import timedelta


def _on_slider_change():
    start, _ = st.session_state.date_range_widget
    num_days = st.session_state.num_days_slider
    end = start + timedelta(days=num_days - 1)
    if end != st.session_state.date_range_widget[1]:
        st.session_state.date_range_widget = (start, end)


def _on_date_range_change():
    val = st.session_state.date_range_widget
    if isinstance(val, tuple) and len(val) == 2:
        start, end = val
        st.session_state.num_days_slider = (end - start).days + 1

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Week 4 — Time Series Forecasting Dashboard",
    layout="wide",
)

st.title("📈 Time Series Forecasting — XGBoost (Best Model Week 2)")
st.markdown(
    "This dashboard uses **XGBoost**, which was the best-performing model "
    "in the Week 2 model comparison.  The training period is **2013-01-01 to "
    "2013-12-31** and the test period is **2014-01-01 to 2014-03-31**."
)

# ---------------------------------------------------------------------------
# Cache the expensive data-loading + feature-engineering step
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data():
    # Base feature-engineered data
    base = pd.read_csv("data/feature_engineered_timeseries.csv", parse_dates=["date"])

    # External data
    oil = pd.read_csv("data/oil.csv", parse_dates=["date"]).rename(
        columns={"dcoilwtico": "oil_price"}
    )
    holidays = pd.read_csv("data/holidays.csv", parse_dates=["date"])

    # Merge holidays and oil
    data = base.merge(
        holidays[["date", "locale", "description"]].drop_duplicates(subset=["date"]),
        on="date",
        how="left",
    )
    data = data.merge(oil, on="date", how="left")
    data["is_holiday"] = data["description"].notna().astype(int)

    national_dates = holidays[holidays["locale"] == "National"][["date"]].copy()
    national_dates["is_national_holiday"] = 1
    data = data.merge(national_dates, on="date", how="left")
    data["is_national_holiday"] = data["is_national_holiday"].fillna(0).astype(int)

    data.drop(columns=["description", "locale"], errors="ignore", inplace=True)

    # Sort and create advanced features
    df = data.sort_values("date").reset_index(drop=True).copy()

    # Calendar features
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Oil features
    df["oil_lag_1"] = df["oil_price"].shift(1)
    df["oil_lag_7"] = df["oil_price"].shift(7)
    df["oil_rolling_mean_7"] = df["oil_price"].rolling(7, min_periods=1).mean()
    df["oil_change"] = df["oil_price"].pct_change()

    # Additional sales features
    df["lag_14"] = df["unit_sales"].shift(14)
    df["lag_21"] = df["unit_sales"].shift(21)
    df["rolling_mean_7"] = df["unit_sales"].rolling(7, min_periods=1).mean()
    df["rolling_mean_30"] = df["unit_sales"].rolling(30, min_periods=1).mean()
    df["rolling_std_14"] = df["unit_sales"].rolling(14, min_periods=1).std()

    # Holiday proximity
    holiday_dates = holidays["date"].sort_values().values

    def days_to_next_holiday(d):
        future = holiday_dates[holiday_dates > np.datetime64(d)]
        if len(future) > 0:
            return (future[0] - np.datetime64(d)).astype("timedelta64[D]").astype(int)
        return np.nan

    df["days_to_next_holiday"] = df["date"].apply(days_to_next_holiday)
    df["weekend_holiday"] = df["is_weekend"] * df["is_holiday"]

    return df


@st.cache_data
def train_xgboost(_df):
    """Train XGBoost and return model, predictions, and metrics."""
    target = "unit_sales"
    features = [c for c in _df.columns if c not in ("date", target)]

    train = _df[_df["date"] < "2014-01-01"].copy()
    test = _df[(_df["date"] >= "2014-01-01") & (_df["date"] <= "2014-03-31")].copy()

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # Drop rows with NaN produced by lag/rolling features
    train_mask = X_train.notna().all(axis=1)
    test_mask = X_test.notna().all(axis=1)
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    test_dates = test.loc[test_mask, "date"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_test), 0, None)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    metrics = {
        "MSE": f"{mse:.2f}",
        "RMSE": f"{rmse:.2f}",
        "MAE": f"{mae:.2f}",
        "R²": f"{r2:.4f}",
    }

    return model, test_dates, y_test, preds, metrics, features


# ---------------------------------------------------------------------------
# Load data and train model
# ---------------------------------------------------------------------------
with st.spinner("Loading data and training XGBoost model …"):
    df = load_and_prepare_data()
    model, test_dates, y_test, preds, metrics, features = train_xgboost(df)

# ---------------------------------------------------------------------------
# Layout: two columns
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 2])

# ---- Left column: Model info & metrics ----
with col_left:
    st.subheader("📊 Model Performance (Test Set)")
    met_cols = st.columns(2)
    for i, (k, v) in enumerate(metrics.items()):
        met_cols[i % 2].metric(label=k, value=v)

    st.subheader("⚙️ Model Hyperparameters")
    st.code(
        """
XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    objective="reg:squarederror"
)
        """
    )

    st.subheader("🧩 Feature Count")
    st.write(f"{len(features)} engineered features used")

    st.subheader("📅 Data Ranges")
    st.write(f"**Training:** 2013-01-01  →  2013-12-31")
    st.write(f"**Test:**      2014-01-01  →  2014-03-31")

# ---- Right column: Date selection & plot ----
with col_right:
    st.subheader("📅 Select Date Range")

    # Build a DataFrame with test results for easy filtering
    results_df = pd.DataFrame({
        "date": test_dates,
        "actual": y_test.values,
        "predicted": preds,
    })

    min_date = results_df["date"].min()
    max_date = results_df["date"].max()

    date_col, days_col = st.columns(2)

    with date_col:
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range_widget",
            on_change=_on_date_range_change,
        )

    # Handle both single-date and tuple returns from date_input
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    with days_col:
        available_days = (max_date - pd.Timestamp(start_date)).days + 1
        current_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
        num_days = st.slider(
            "Number of days",
            min_value=1,
            max_value=max(available_days, 1),
            value=min(max(current_days, 1), max(available_days, 1)),
            key="num_days_slider",
            on_change=_on_slider_change,
        )
        end_date = pd.Timestamp(start_date) + pd.Timedelta(days=int(num_days) - 1)

    mask = (results_df["date"] >= pd.Timestamp(start_date)) & (
        results_df["date"] <= pd.Timestamp(end_date)
    )
    filtered = results_df[mask]

    st.write(f"Showing **{num_days}** days ({len(filtered)} with data) from **{start_date}** to **{end_date.date()}**")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        filtered["date"],
        filtered["actual"],
        label="Actual",
        color="orange",
        linewidth=2,
    )
    ax.plot(
        filtered["date"],
        filtered["predicted"],
        label="XGBoost Forecast",
        color="red",
        linestyle="--",
        linewidth=2,
    )
    ax.set_title("XGBoost — Actual vs Forecast", fontsize=14)
    ax.set_ylabel("Unit Sales")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)

    # Metrics for the filtered period
    if len(filtered) > 1:
        f_mse = mean_squared_error(filtered["actual"], filtered["predicted"])
        f_rmse = np.sqrt(f_mse)
        f_mae = mean_absolute_error(filtered["actual"], filtered["predicted"])
        f_r2 = r2_score(filtered["actual"], filtered["predicted"])
        st.caption(
            f"Filtered period — MSE: {f_mse:.2f}  |  "
            f"RMSE: {f_rmse:.2f}  |  "
            f"MAE: {f_mae:.2f}  |  "
            f"R²: {f_r2:.4f}"
        )
