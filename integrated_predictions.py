import pandas as pd
import numpy as np
import torch
import joblib
import json
import datetime
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load model and data
def load_model_assets():
    scaler_path = 'feature_scaler.joblib'
    model_path = "offense_predictor_mlp.pth"
    features_path = "processed_data/features.json"
    
    try:
        # Load scaler and features
        scaler = joblib.load(scaler_path)
        with open(features_path, 'r') as f:
            features = json.load(f)
        
        # Import MLP class from train_model.py
        from train_model import MLP
        
        # Initialize model
        input_size = len(features)
        model = MLP(input_size)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        print("✅ Model assets loaded successfully!")
        return {
            "scaler": scaler,
            "model": model,
            "features": features
        }
    except Exception as e:
        print(f"❌ Error loading model assets: {e}")
        return None

# Load the dataset
def load_data():
    try:
        df = pd.read_csv("dataset/offense_set_cleaned.csv", parse_dates=['DateTime'])
        print(f"✅ Data loaded successfully: {len(df)} records found")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# Get vehicle type, day of week, and time recommendations from location model
def get_location_recommendations(prediction_date, predicted_count):
    try:
        # Query the location model API to get risky hours, days, and vehicle types
        day_of_week = prediction_date.weekday()  # 0-6 where 0 is Monday
        
        # Try to connect to the API, but use fallback data if not available
        try:
            response = requests.get('http://127.0.0.1:5000/api/top-locations?limit=5', timeout=2)
            if response.status_code == 200:
                top_locations = response.json()
                api_available = True
            else:
                print(f"⚠️ API warning: {response.status_code}. Using fallback data.")
                api_available = False
        except requests.exceptions.RequestException:
            print("⚠️ API unavailable. Using fallback data.")
            api_available = False
        
        # Use fallback data if API is not available
        if not api_available:
            # Fallback data based on typical offense patterns
            # These are common patterns that can be used when the API is not available
            top_hours = [8, 12, 17]  # Morning, noon, afternoon rush hours
            vehicle_type = 2  # Most common vehicle type (assuming 2 is common)
            
            # Weekends (Sunday=6, Saturday=5) typically have different patterns
            is_weekend = day_of_week >= 5
            day_risk = "HIGH" if day_of_week in [1, 4] else ("MEDIUM" if is_weekend else "LOW")
            
            return {
                "predicted_count": predicted_count,
                "date": prediction_date.strftime('%Y-%m-%d'),
                "day_of_week": day_of_week,
                "day_name": prediction_date.strftime('%A'),
                "day_risk": day_risk,
                "top_hours": top_hours,
                "recommended_vehicle_type": vehicle_type,
                "risk_score": (predicted_count / 10) * (1 if day_risk == "HIGH" else (0.7 if day_risk == "MEDIUM" else 0.5)),
                "data_source": "fallback"
            }
        
        # If we reach here, the API is available and we have data from it
        # Analyze patterns in top locations
        vehicle_types = []
        weekend_counts = 0
        weekday_counts = 0
        hour_counts = {h: 0 for h in range(24)}
        
        # Extract patterns from top locations
        for location in top_locations:
            if 'isWeekend' in location and location['isWeekend'] == 1:
                weekend_counts += 1
            else:
                weekday_counts += 1
                
            if 'vehicleType' in location:
                vehicle_types.append(location['vehicleType'])
        
        # Try to get feature importance
        try:
            feature_response = requests.get('http://127.0.0.1:5000/api/features', timeout=2)
            if feature_response.status_code == 200:
                features = feature_response.json()
                # Extract hour information if available
                for feature in features:
                    if 'hour' in feature['feature']:
                        hour = int(feature['feature'].split('_')[-1]) if '_' in feature['feature'] else 0
                        hour_counts[hour] = feature['importance']
        except:
            # If feature API fails, use some default values for hours
            for h in [8, 12, 17]:
                hour_counts[h] = 0.1
        
        # Find the top hours, vehicle types based on the data
        top_hours = [h for h, _ in sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
        
        # Handle empty vehicle types
        if vehicle_types:
            top_vehicle = max(set(vehicle_types), key=vehicle_types.count)
        else:
            top_vehicle = 2  # Default to vehicle type 2 if no data
            
        is_weekend_risky = weekend_counts > weekday_counts
        
        # Build recommendations based on the prediction date
        if is_weekend_risky and day_of_week >= 5:  # Weekend (5=Sat, 6=Sun)
            day_risk = "HIGH"
        elif not is_weekend_risky and day_of_week < 5:  # Weekday
            day_risk = "HIGH"
        else:
            day_risk = "MEDIUM"
            
        return {
            "predicted_count": predicted_count,
            "date": prediction_date.strftime('%Y-%m-%d'),
            "day_of_week": day_of_week,
            "day_name": prediction_date.strftime('%A'),
            "day_risk": day_risk,
            "top_hours": top_hours,
            "recommended_vehicle_type": top_vehicle,
            "risk_score": (predicted_count / 10) * (1 if day_risk == "HIGH" else (0.7 if day_risk == "MEDIUM" else 0.5)),
            "data_source": "api"
        }
    except Exception as e:
        print(f"❌ Error getting location recommendations: {e}")
        # Provide default fallback data even in case of error
        return {
            "predicted_count": predicted_count,
            "date": prediction_date.strftime('%Y-%m-%d'),
            "day_of_week": prediction_date.weekday(),
            "day_name": prediction_date.strftime('%A'),
            "day_risk": "MEDIUM",
            "top_hours": [8, 12, 17],
            "recommended_vehicle_type": 2,
            "risk_score": predicted_count / 10,
            "data_source": "error_fallback"
        }

# Generate predictions for a date range including location-based recommendations
def predict_with_location_data(start_date, end_date):
    # Load model assets and data
    assets = load_model_assets()
    df_cleaned = load_data()
    
    if assets is None or df_cleaned is None:
        print("Cannot generate predictions: missing model assets or data")
        return None
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Aggregate historical data for lag features
    df_daily = df_cleaned.set_index('DateTime').resample('D')['Offence_ID'].count().reset_index()
    df_daily.columns = ['DateTime', 'offense_count']
    df_daily = df_daily.fillna(0)
    
    # Initialize predictions dictionary
    predictions = []
    
    # Get latest counts for lag features
    latest_counts = df_daily.set_index('DateTime')['offense_count']
    
    # Generate predictions for each date
    for current_date in date_range:
        # Create features for prediction
        features = {}
        
        # Calculate lags based on historical data
        lag_1_date = current_date - pd.Timedelta(days=1)
        lag_7_date = current_date - pd.Timedelta(days=7)
        lag_14_date = current_date - pd.Timedelta(days=14)
        
        # Find the closest past date for lag features
        features['count_lag_1'] = latest_counts.get(
            latest_counts.index[latest_counts.index <= lag_1_date].max(), 0)
        features['count_lag_7'] = latest_counts.get(
            latest_counts.index[latest_counts.index <= lag_7_date].max(), 0)
        features['count_lag_14'] = latest_counts.get(
            latest_counts.index[latest_counts.index <= lag_14_date].max(), 0)
        
        # Time-based features
        features['year'] = current_date.year
        features['month'] = current_date.month
        features['day'] = current_date.day
        features['day_of_week'] = current_date.dayofweek
        features['day_of_year'] = current_date.dayofyear
        features['week_of_year'] = current_date.isocalendar().week
        
        # Create feature vector in the right order
        feature_vector = pd.DataFrame([features])[assets["features"]]
        
        # Scale features
        scaled_features = assets["scaler"].transform(feature_vector)
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        
        # Generate prediction
        with torch.no_grad():
            prediction = assets["model"](features_tensor)
            predicted_count = max(0, prediction.item())
            predicted_count = round(predicted_count, 2)
        
        # Get location-based recommendations for this date
        location_data = get_location_recommendations(current_date, predicted_count)
        
        # If location data is available, add to predictions
        if location_data:
            predictions.append(location_data)
        else:
            # Fallback without location data
            predictions.append({
                "date": current_date.strftime('%Y-%m-%d'),
                "predicted_count": predicted_count,
                "day_of_week": current_date.weekday(),
                "day_name": current_date.strftime('%A')
            })
    
    return predictions

# Main execution
if __name__ == "__main__":
    # Set start and end dates for predictions
    today = datetime.datetime.now().date()
    start_date = today
    end_date = today + datetime.timedelta(days=10)
    
    print(f"Generating predictions from {start_date} to {end_date}...")
    predictions = predict_with_location_data(start_date, end_date)
    
    if predictions:
        print("\n===== INTEGRATED OFFENSE PREDICTIONS =====")
        print(f"{'Date':<12} {'Count':<8} {'Day':<10} {'Risk':<6} {'Vehicle Type':<13} {'Peak Hours'}")
        print("-" * 60)
        
        for pred in predictions:
            date = pred["date"]
            count = pred["predicted_count"]
            day = pred["day_name"]
            risk = pred.get("day_risk", "N/A")
            vehicle = pred.get("recommended_vehicle_type", "N/A")
            hours = ", ".join(map(str, pred.get("top_hours", [])))
            
            print(f"{date:<12} {count:<8.2f} {day:<10} {risk:<6} {vehicle:<13} {hours}")
        
        print("\nNote: These predictions integrate both temporal (when) and spatial (where) offense patterns.")
        print("Use this information for more effective resource allocation and preventive measures.")
    else:
        print("Failed to generate predictions.") 