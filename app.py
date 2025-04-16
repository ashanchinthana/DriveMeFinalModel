import streamlit as st
import pandas as pd
import torch
import joblib
from PIL import Image
import os
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Offense Data Analysis & Prediction")

# --- Load Assets ---
@st.cache_data # Cache data loading
def load_data():
    cleaned_file_path = "dataset/offense_set_cleaned.csv"
    if os.path.exists(cleaned_file_path):
        df = pd.read_csv(cleaned_file_path, parse_dates=['DateTime'])
        return df
    else:
        st.error(f"Cleaned data file not found at {cleaned_file_path}")
        return None

@st.cache_resource # Cache model and scaler loading
def load_model_assets():
    scaler_path = 'feature_scaler.joblib'
    model_path = "offense_predictor_mlp.pth"
    features_path = os.path.join("processed_data", 'features.json')
    test_data_path = os.path.join("processed_data", 'X_test.pt')
    test_target_path = os.path.join("processed_data", 'y_test.pt')

    assets = {"scaler": None, "model_def": None, "model_state": None, "features": None, "X_test": None, "y_test": None}
    try:
        assets["scaler"] = joblib.load(scaler_path)

        # Define model architecture here (needs to match the trained model)
        # Import or define the MLP class
        # (Assuming MLP class is defined similarly to train_model.py)
        from train_model import MLP # Or copy the class definition here
        with open(features_path, 'r') as f:
            assets["features"] = json.load(f)
        input_size = len(assets["features"])
        assets["model_def"] = MLP(input_size)

        assets["model_state"] = torch.load(model_path)
        assets["X_test"] = torch.load(test_data_path)
        assets["y_test"] = torch.load(test_target_path)

    except FileNotFoundError as e:
        st.error(f"Error loading model assets: {e}. Please ensure 'feature_scaler.joblib', '{model_path}', and files in 'processed_data' exist.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading model assets: {e}")
        return None

    return assets

df_cleaned = load_data()
model_assets = load_model_assets()

# --- Helper Function for Prediction ---
@st.cache_data(show_spinner="Generating predictions...")
def predict_date_range(start_date, end_date, _model_assets, _df_cleaned):
    if _model_assets is None or _df_cleaned is None:
        st.error("Model assets or cleaned data not loaded. Cannot predict.")
        return None

    scaler = _model_assets["scaler"]
    model = _model_assets["model_def"]
    model.load_state_dict(_model_assets["model_state"])
    model.eval()
    feature_names = _model_assets["features"]

    # Find the latest date in the historical data to get lag features
    # Need the daily aggregated data for lags
    df_daily_hist = _df_cleaned.set_index('DateTime').resample('D')['Offence_ID'].count().reset_index()
    df_daily_hist.columns = ['DateTime', 'offense_count']
    df_daily_hist = df_daily_hist.fillna(0)
    df_daily_hist.sort_values('DateTime', inplace=True)
    latest_hist_date = df_daily_hist['DateTime'].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    predictions = {}

    # Get the required lag values from the *end* of the historical daily data
    # Note: This assumes the required lag periods (1, 7, 14) are available in historical data
    if len(df_daily_hist) < 14:
        st.error("Not enough historical data (less than 14 days) to calculate all required lag features.")
        return None

    latest_counts = df_daily_hist.set_index('DateTime').iloc[-14:]['offense_count']

    for current_date in date_range:
        # Create features for the current_date
        features = {}

        # Calculate lags based on the *latest available* historical data
        # This is an approximation for future dates
        lag_1_date = current_date - pd.Timedelta(days=1)
        lag_7_date = current_date - pd.Timedelta(days=7)
        lag_14_date = current_date - pd.Timedelta(days=14)

        # Find the closest *past* date in latest_counts for lags
        # If prediction date is far in future, lags will rely on older data from latest_counts
        features['count_lag_1'] = latest_counts.get(latest_counts.index[latest_counts.index <= lag_1_date].max(), 0)
        features['count_lag_7'] = latest_counts.get(latest_counts.index[latest_counts.index <= lag_7_date].max(), 0)
        features['count_lag_14'] = latest_counts.get(latest_counts.index[latest_counts.index <= lag_14_date].max(), 0)

        # Time-based features
        features['year'] = current_date.year
        features['month'] = current_date.month
        features['day'] = current_date.day
        features['day_of_week'] = current_date.dayofweek
        features['day_of_year'] = current_date.dayofyear
        features['week_of_year'] = current_date.isocalendar().week

        # Ensure features are in the correct order
        feature_vector = pd.DataFrame([features])[feature_names]

        # Scale features
        scaled_features = scaler.transform(feature_vector)
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32)

        # Predict
        with torch.no_grad():
            prediction = model(features_tensor)
            # Ensure prediction is non-negative
            predicted_count = max(0, prediction.item())

        predictions[current_date.strftime('%Y-%m-%d')] = predicted_count

    return predictions

# --- App Layout ---
st.title("Analysis of Offense Data Set")

st.markdown("""
This application presents an analysis of the provided offense dataset, including exploratory data analysis (EDA),
time series analysis (TSA), and results from a predictive model trained to forecast daily offense counts.
""")

# --- Section 1: EDA ---
st.header("1. Exploratory Data Analysis (EDA)")

if df_cleaned is not None:
    st.subheader("Raw Data Sample (Cleaned)")
    st.dataframe(df_cleaned.head())

    # Display EDA Plots
    st.subheader("Data Distributions")
    try:
        img_hist = Image.open('eda_histograms.png')
        st.image(img_hist, caption='Distribution of Location and Vehicle Type IDs')
    except FileNotFoundError:
        st.warning("EDA histogram plot (`eda_histograms.png`) not found.")

    st.subheader("Offenses Over Time")
    try:
        img_ts = Image.open('eda_timeseries.png')
        st.image(img_ts, caption='Daily Offense Counts Over Time')
    except FileNotFoundError:
        st.warning("EDA time series plot (`eda_timeseries.png`) not found.")
else:
    st.warning("Could not load cleaned data for EDA.")

# --- Section 2: Time Series Analysis ---
st.header("2. Time Series Analysis (TSA)")
st.subheader("Seasonal Decomposition (Weekly)")

try:
    img_decomp = Image.open('tsa_decomposition_weekly.png')
    st.image(img_decomp, caption='Trend, Seasonality (Weekly), and Residuals of Daily Offense Counts')
except FileNotFoundError:
    st.warning("TSA decomposition plot (`tsa_decomposition_weekly.png`) not found.")

st.markdown("""
**Key Findings:**
*   **Trend:** A slight overall decreasing trend in daily offenses was observed.
*   **Seasonality:** A clear weekly pattern exists, with the highest number of offenses typically occurring on **Tuesday** and the lowest on **Sunday**.
""")

# --- Section 3: Predictive Model ---
st.header("3. Predictive Model (MLP)")
st.markdown("""
A Multi-Layer Perceptron (MLP) model was trained on engineered features (time-based features and lagged daily counts)
to predict the number of offenses for the next day.
""")

if model_assets is not None and model_assets["model_def"] is not None and model_assets["model_state"] is not None:
    st.subheader("Model Evaluation (Test Set)")

    # Load model state into definition
    model = model_assets["model_def"]
    model.load_state_dict(model_assets["model_state"])
    model.eval()

    # Make predictions on test set
    with torch.no_grad():
        test_predictions = model(model_assets["X_test"])

    # Get targets
    test_targets = model_assets["y_test"]

    # Calculate metrics
    targets_np = test_targets.numpy().flatten()
    predictions_np = test_predictions.numpy().flatten()

    rmse = np.sqrt(mean_squared_error(targets_np, predictions_np))
    mae = mean_absolute_error(targets_np, predictions_np)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
    with col2:
        st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}")

    # Plot predictions vs actuals
    st.subheader("Test Set: Predictions vs Actuals")
    df_results = pd.DataFrame({
        'Actual': targets_np,
        'Predicted': predictions_np
    })
    st.line_chart(df_results)

else:
    st.warning("Could not load model assets for evaluation.")

# --- Section 4: Forecast Date Range ---
st.header("4. Forecast Offenses for a Date Range")
st.markdown("""
Select a start and end date to get a forecast of the expected number of offenses for each day.
**Note:** This forecast uses the trained model and the latest available historical data to estimate future counts independently for each day.
""")

if model_assets is not None and df_cleaned is not None:
    # Default dates - start from the day after the last data point
    last_data_date = df_cleaned['DateTime'].max().date()
    default_start_date = last_data_date + datetime.timedelta(days=1)
    default_end_date = default_start_date + datetime.timedelta(days=6) # Default to 1 week forecast

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Select Start Date", value=default_start_date, min_value=default_start_date)
    with col2:
        end_date = st.date_input("Select End Date", value=default_end_date, min_value=start_date)

    if st.button("Generate Forecast"):
        if start_date and end_date:
            if start_date > end_date:
                st.error("Error: End date must be after start date.")
            else:
                forecast_results = predict_date_range(start_date, end_date, model_assets, df_cleaned)

                if forecast_results:
                    st.subheader(f"Forecasted Offenses from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                    df_forecast = pd.DataFrame(list(forecast_results.items()), columns=['Date', 'Predicted Offenses'])
                    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
                    df_forecast.set_index('Date', inplace=True)

                    st.line_chart(df_forecast)
                    st.dataframe(df_forecast)
        else:
            st.warning("Please select both a start and end date.")
else:
    st.warning("Model assets or cleaned data not available. Cannot generate forecast.")

st.markdown("--- ")
st.markdown("*Application demonstrating analysis of offense data.*") 