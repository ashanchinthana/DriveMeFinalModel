from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
import joblib
import json
import os
import datetime
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Create FastAPI app
app = FastAPI(
    title="Comprehensive Offense Prediction API",
    description="API for predicting offense details including location, vehicle type, violation type, and peak hours",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model directory
MODEL_DIR = "comprehensive_model"

# Global variables to hold loaded models and data
models = {}
scaler = None
label_mappings = {}
feature_importance = {}
summary_statistics = {}


# Load models on startup
@app.on_event("startup")
async def load_models():
    global models, scaler, label_mappings, feature_importance, summary_statistics
    
    # Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"Warning: Model directory '{MODEL_DIR}' not found.")
        return
    
    # Load models
    for model_file in os.listdir(MODEL_DIR):
        if model_file.endswith("_model.joblib"):
            model_name = model_file.replace("_model.joblib", "")
            model_path = os.path.join(MODEL_DIR, model_file)
            try:
                models[model_name] = joblib.load(model_path)
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
    
    # Load scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print("Loaded scaler")
        except Exception as e:
            print(f"Error loading scaler: {e}")
    
    # Load label mappings
    mappings_path = os.path.join(MODEL_DIR, "label_mappings.json")
    if os.path.exists(mappings_path):
        try:
            with open(mappings_path, 'r') as f:
                label_mappings = json.load(f)
            print("Loaded label mappings")
        except Exception as e:
            print(f"Error loading label mappings: {e}")
    
    # Load feature importance
    importance_path = os.path.join(MODEL_DIR, "feature_importance.json")
    if os.path.exists(importance_path):
        try:
            with open(importance_path, 'r') as f:
                feature_importance = json.load(f)
            print("Loaded feature importance")
        except Exception as e:
            print(f"Error loading feature importance: {e}")
    
    # Load summary statistics
    stats_path = os.path.join(MODEL_DIR, "summary_statistics.json")
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r') as f:
                summary_statistics = json.load(f)
            print("Loaded summary statistics")
        except Exception as e:
            print(f"Error loading summary statistics: {e}")
    
    print(f"Loaded {len(models)} models")


# Input models
class PredictionRequest(BaseModel):
    date: Optional[str] = None  # YYYY-MM-DD format
    hour: Optional[int] = None  # 0-23
    month: Optional[int] = None  # 1-12
    day: Optional[int] = None  # 1-31
    day_of_week: Optional[int] = None  # 0-6 (0=Monday, 6=Sunday)
    is_weekend: Optional[int] = None  # 0 or 1


# Routes
@app.get("/")
async def root():
    return {
        "message": "Comprehensive Offense Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(models) > 0,
        "scaler_loaded": scaler is not None,
        "available_models": list(models.keys())
    }


@app.get("/stats")
async def get_statistics():
    if not summary_statistics:
        raise HTTPException(status_code=404, detail="Statistics not available")
    
    return summary_statistics


@app.post("/predict")
async def predict(request: PredictionRequest):
    # Check if models are loaded
    if not models or not scaler:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Process input
    try:
        # If date is provided, extract month, day, day_of_week
        if request.date:
            date_obj = datetime.datetime.strptime(request.date, "%Y-%m-%d").date()
            month = date_obj.month
            day = date_obj.day
            day_of_week = date_obj.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
        else:
            # Use provided values or defaults
            month = request.month if request.month is not None else datetime.datetime.now().month
            day = request.day if request.day is not None else datetime.datetime.now().day
            
            if request.day_of_week is not None:
                day_of_week = request.day_of_week
                is_weekend = 1 if day_of_week >= 5 else 0
            else:
                # Calculate day_of_week and is_weekend from current date
                today = datetime.datetime.now().date()
                day_of_week = today.weekday()
                is_weekend = 1 if day_of_week >= 5 else 0
        
        # Use provided hour or default to current hour
        hour = request.hour if request.hour is not None else datetime.datetime.now().hour
        
        # Override is_weekend if provided
        if request.is_weekend is not None:
            is_weekend = request.is_weekend
        
        # Create feature vector
        features = pd.DataFrame({
            'month': [month],
            'day': [day],
            'day_of_week': [day_of_week],
            'hour': [hour],
            'is_weekend': [is_weekend]
        })
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make predictions for each target
        predictions = {}
        
        # Add input parameters to response
        predictions["input"] = {
            "month": month,
            "day": day,
            "day_of_week": day_of_week,
            "hour": hour,
            "is_weekend": is_weekend,
            "date": request.date or datetime.date(2023, month, day).strftime("%Y-%m-%d")
        }
        
        for target_name, model in models.items():
            if target_name == 'high_violation_time':
                continue  # Skip this model for now
                
            # Get prediction
            pred_encoded = model.predict(features_scaled)[0]
            
            # Get label mappings for this target
            target_mappings = label_mappings.get(target_name, {})
            
            # Convert encoded prediction back to original label using str to handle integer keys in JSON
            pred_label = target_mappings.get(str(pred_encoded), pred_encoded)
            
            # Store prediction
            predictions[target_name] = pred_label
        
        # Get high violation times
        if 'high_violation_time' in models:
            hour_predictions = {}
            for h in range(24):
                hour_vector = np.array([[h]])
                violation_count = models['high_violation_time'].predict(hour_vector)[0]
                hour_predictions[h] = float(violation_count)
            
            # Get top 3 hours with highest violation counts
            top_hours = sorted(hour_predictions.items(), key=lambda x: x[1], reverse=True)[:3]
            predictions['high_violation_hours'] = [hour for hour, _ in top_hours]
            predictions['hour_predictions'] = hour_predictions
        
        # Add probabilities for location and vehicle type
        for target_name in ['location', 'vehicle_type', 'violation_type']:
            # Get prediction probabilities
            if target_name in models:
                pred_proba = models[target_name].predict_proba(features_scaled)[0]
                
                # Get top 3 classes with highest probabilities
                top_classes_idx = pred_proba.argsort()[-3:][::-1]
                
                # Convert indices to original labels
                target_mappings = label_mappings.get(target_name, {})
                top_classes = [target_mappings.get(str(idx), idx) for idx in top_classes_idx]
                top_probs = pred_proba[top_classes_idx]
                
                # Store top predictions with probabilities
                predictions[f"{target_name}_top3"] = [
                    {"label": label, "probability": float(prob)} 
                    for label, prob in zip(top_classes, top_probs)
                ]
        
        return predictions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/importance")
async def get_feature_importance():
    if not feature_importance:
        raise HTTPException(status_code=404, detail="Feature importance not available")
    
    return feature_importance


# Main function to run the API
if __name__ == "__main__":
    # Run the API with uvicorn
    uvicorn.run("comprehensive_api:app", host="0.0.0.0", port=8000, reload=True) 