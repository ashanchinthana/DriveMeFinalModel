from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
import joblib
import os
import json
import datetime
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration & Asset Loading ---
SCALER_PATH = 'feature_scaler.joblib'
MODEL_PATH = "offense_predictor_mlp.pth"
FEATURES_PATH = os.path.join("processed_data", 'features.json')
CLEANED_DATA_PATH = "dataset/offense_set_cleaned.csv"

# --- Globals for Assets (Load once on startup) ---
scaler = None
model = None
feature_names = []
df_cleaned_global = None
df_daily_hist_global = None
latest_counts_global = None

app = FastAPI(title="Offense Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
def load_assets():
    global scaler, model, feature_names, df_cleaned_global, df_daily_hist_global, latest_counts_global
    print("Loading assets for API...")
    try:
        if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH) or not os.path.exists(CLEANED_DATA_PATH):
            raise FileNotFoundError("One or more required asset files not found.")

        scaler = joblib.load(SCALER_PATH)

        with open(FEATURES_PATH, 'r') as f:
            feature_names = json.load(f)
        input_size = len(feature_names)

        # Define model architecture (needs to match the trained model)
        # Ensure train_model.py or the definition is accessible
        try:
            from train_model import MLP
        except ImportError:
            # Alternative: Define MLP class directly here if train_model.py isn't in PYTHONPATH
            import torch.nn as nn
            class MLP(nn.Module):
                def __init__(self, input_size):
                    super(MLP, self).__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_size, 64), nn.ReLU(),
                        nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, 1)
                    )
                def forward(self, x): return self.network(x)

        model = MLP(input_size)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

        # Load and preprocess historical data for lags
        df_cleaned_global = pd.read_csv(CLEANED_DATA_PATH, parse_dates=['DateTime'])
        df_daily_hist_global = df_cleaned_global.set_index('DateTime').resample('D')['Offence_ID'].count().reset_index()
        df_daily_hist_global.columns = ['DateTime', 'offense_count']
        df_daily_hist_global = df_daily_hist_global.fillna(0)
        df_daily_hist_global.sort_values('DateTime', inplace=True)

        if len(df_daily_hist_global) < 14:
             print("Warning: Not enough historical data for all lag features.")
             latest_counts_global = None
        else:
            latest_counts_global = df_daily_hist_global.set_index('DateTime').iloc[-14:]['offense_count']

        print("Assets loaded successfully.")

    except Exception as e:
        print(f"FATAL: Failed to load assets on startup: {e}")
        # Depending on deployment, might want to exit or prevent app start
        scaler = None
        model = None
        # Raise error to potentially stop FastAPI startup if assets are critical
        raise RuntimeError(f"Failed to load critical assets: {e}") from e

# --- Request/Response Models ---
class DateRange(BaseModel):
    start_date: datetime.date
    end_date: datetime.date

# --- Prediction Logic (Helper Function) ---
def generate_prediction_for_range(start_date: datetime.date, end_date: datetime.date) -> Dict[str, float]:
    global scaler, model, feature_names, df_daily_hist_global, latest_counts_global

    if model is None or scaler is None or latest_counts_global is None:
        raise HTTPException(status_code=503, detail="Model or supporting assets not available. API might be starting up or encountered an error.")

    predictions = {}
    try:
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        for current_date in date_range:
            features = {}
            lag_1_date = current_date - pd.Timedelta(days=1)
            lag_7_date = current_date - pd.Timedelta(days=7)
            lag_14_date = current_date - pd.Timedelta(days=14)

            features['count_lag_1'] = latest_counts_global.get(latest_counts_global.index[latest_counts_global.index <= lag_1_date].max(), 0)
            features['count_lag_7'] = latest_counts_global.get(latest_counts_global.index[latest_counts_global.index <= lag_7_date].max(), 0)
            features['count_lag_14'] = latest_counts_global.get(latest_counts_global.index[latest_counts_global.index <= lag_14_date].max(), 0)

            features['year'] = current_date.year
            features['month'] = current_date.month
            features['day'] = current_date.day
            features['day_of_week'] = current_date.dayofweek
            features['day_of_year'] = current_date.dayofyear
            features['week_of_year'] = current_date.isocalendar().week

            feature_vector = pd.DataFrame([features])[feature_names]
            scaled_features = scaler.transform(feature_vector)
            features_tensor = torch.tensor(scaled_features, dtype=torch.float32)

            with torch.no_grad():
                prediction = model(features_tensor)
                predicted_count = max(0, prediction.item())

            predictions[current_date.strftime('%Y-%m-%d')] = predicted_count

        return predictions

    except Exception as e:
        print(f"Error during prediction generation: {e}")
        # Log the error details here
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

# --- API Endpoint ---
@app.post("/predict/", response_model=Dict[str, float])
def predict_range_endpoint(date_info: DateRange):
    """Receives a start and end date, returns predicted offense counts for each day in the range."""
    if date_info.start_date > date_info.end_date:
        raise HTTPException(status_code=400, detail="End date must be after start date.")

    predictions = generate_prediction_for_range(date_info.start_date, date_info.end_date)
    return predictions

@app.get("/")
def read_root():
    return {"message": "Welcome to the Offense Prediction API. Use the /predict/ endpoint."}

# --- To Run (in terminal) ---
# uvicorn api:app --reload 