import pandas as pd
import numpy as np
import joblib
import os
import json
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# Create the Flask app with improved settings
app = Flask(__name__, 
            static_folder='./',
            static_url_path='')
            
# Enable CORS with more permissive settings
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Load model and data
model = None
location_risk_data = None

def load_assets():
    global model, location_risk_data
    
    try:
        # Run location_analysis.py if model doesn't exist
        if not os.path.exists('location_violation_model.joblib') or not os.path.exists('location_risk_scores.csv'):
            print("Model or risk data not found. Running location analysis...")
            import subprocess
            subprocess.run(['python', 'location_analysis.py'])
        
        # Load model
        model = joblib.load('location_violation_model.joblib')
        
        # Load location risk data
        location_risk_data = pd.read_csv('location_risk_scores.csv')
        
        print("Assets loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading assets: {e}")
        return False

# Routes
@app.route('/')
def index():
    return send_from_directory('./', 'location_dashboard.html')

@app.route('/api/top-locations', methods=['GET'])
def get_top_locations():
    if location_risk_data is None:
        return jsonify({"error": "Data not loaded"}), 500
    
    # Get query parameters
    limit = request.args.get('limit', default=10, type=int)
    
    # Get top locations from risk data
    top_locations = location_risk_data.sort_values('normalized_risk', ascending=False).head(limit)
    
    # Convert to list of dicts for JSON response
    locations = []
    for _, row in top_locations.iterrows():
        locations.append({
            "location": int(row['Location']),
            "offenses": int(row['total_offenses']),
            "weekendRatio": float(row['weekend_ratio']),
            "avgDay": float(row['avg_day']),
            "risk": float(row['normalized_risk'])
        })
    
    return jsonify(locations)

@app.route('/api/predict', methods=['POST'])
def predict_location_risk():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Get request data
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    vehicle_type = data.get('vehicleType')
    day_of_week = data.get('dayOfWeek')
    hour = data.get('hour')
    
    if vehicle_type is None or day_of_week is None or hour is None:
        return jsonify({"error": "Missing required parameters"}), 400
    
    try:
        # Convert to appropriate types
        vehicle_type = int(vehicle_type)
        day_of_week = int(day_of_week)
        hour = int(hour)
        
        # Create feature vector
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Create DataFrame for prediction (with one-hot encoding)
        features = pd.DataFrame([{
            'Vehicle_type': vehicle_type,
            'year': location_risk_data['year'].iloc[0] if 'year' in location_risk_data.columns else 2023,
            'month': location_risk_data['month'].iloc[0] if 'month' in location_risk_data.columns else 6,
            'day_of_week': day_of_week,
            'hour': hour,
            'is_weekend': is_weekend
        }])
        
        # One-hot encode
        features_encoded = pd.get_dummies(features, drop_first=True)
        
        # Get the list of feature columns used by the model
        model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        
        # If model features are available, make sure our input features match
        if model_features is not None:
            for feature in model_features:
                if feature not in features_encoded.columns:
                    features_encoded[feature] = 0
            # Keep only the columns used by the model
            features_encoded = features_encoded[model_features]
        
        # Make prediction (probability of high-violation location)
        probability = float(model.predict_proba(features_encoded)[0, 1])
        
        # Return prediction
        return jsonify({
            "risk": probability,
            "input": {
                "vehicleType": vehicle_type,
                "dayOfWeek": day_of_week,
                "hour": hour,
                "isWeekend": is_weekend
            }
        })
    
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route('/api/features', methods=['GET'])
def get_feature_importance():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get feature importance from model
        if hasattr(model, 'feature_importances_'):
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"Feature {i}" for i in range(len(model.feature_importances_))]
            
            # Create list of feature importance objects
            features = []
            for name, importance in zip(feature_names, model.feature_importances_):
                features.append({
                    "feature": name,
                    "importance": float(importance)
                })
            
            # Sort by importance
            features.sort(key=lambda x: x["importance"], reverse=True)
            
            return jsonify(features)
        else:
            return jsonify({"error": "Model does not have feature importances"}), 500
    
    except Exception as e:
        return jsonify({"error": f"Error getting feature importance: {str(e)}"}), 500

# Static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('./', path)

if __name__ == '__main__':
    # Load assets before starting server
    if load_assets():
        # Start server with host='0.0.0.0' to allow external connections
        print("Server running at http://127.0.0.1:5000 and http://localhost:5000")
        print("If you encounter 403 errors, try accessing through http://127.0.0.1:5000 directly")
        app.run(debug=True, port=5000, host='0.0.0.0')
    else:
        print("Failed to load required assets. Exiting.") 