import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from comprehensive_prediction_model import ComprehensivePredictionModel

# Set page configuration
st.set_page_config(
    page_title="Comprehensive Offense Prediction Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Function to load models and data
@st.cache_resource
def load_models():
    model_dir = "comprehensive_model"
    
    # Check if models exist
    if not os.path.exists(model_dir):
        return None
    
    # Load models
    models = {}
    for model_file in os.listdir(model_dir):
        if model_file.endswith("_model.joblib"):
            model_name = model_file.replace("_model.joblib", "")
            model_path = os.path.join(model_dir, model_file)
            models[model_name] = joblib.load(model_path)
    
    # Load scaler
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = None
    
    # Load label mappings
    mappings_path = os.path.join(model_dir, "label_mappings.json")
    if os.path.exists(mappings_path):
        with open(mappings_path, 'r') as f:
            label_mappings = json.load(f)
    else:
        label_mappings = {}
    
    # Load feature importance
    importance_path = os.path.join(model_dir, "feature_importance.json")
    if os.path.exists(importance_path):
        with open(importance_path, 'r') as f:
            feature_importance = json.load(f)
    else:
        feature_importance = {}
    
    # Load summary statistics
    stats_path = os.path.join(model_dir, "summary_statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            summary_statistics = json.load(f)
    else:
        summary_statistics = {}
    
    return {
        "models": models,
        "scaler": scaler,
        "label_mappings": label_mappings,
        "feature_importance": feature_importance,
        "summary_statistics": summary_statistics
    }

# Function to make predictions
def make_prediction(models, scaler, label_mappings, month, day, day_of_week, hour, is_weekend=None):
    # Check if models are loaded
    if not models or not scaler:
        st.error("Models not loaded. Please train the models first.")
        return None
    
    # If is_weekend is not provided, derive it from day_of_week
    if is_weekend is None:
        is_weekend = 1 if day_of_week >= 5 else 0
    
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
            hour_predictions[h] = violation_count
        
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

# Main app
def main():
    st.title("Comprehensive Offense Prediction Dashboard")
    
    # Check if model directory exists
    if not os.path.exists("comprehensive_model"):
        st.warning("Model directory not found. Please run the comprehensive_prediction_model.py script first.")
        
        if st.button("Train Models Now"):
            with st.spinner("Training models... This may take a few minutes."):
                # Create and train the model
                model = ComprehensivePredictionModel()
                model.load_data()
                model.prepare_features()
                model.train_models()
                model.evaluate_models()
                st.success("Models trained successfully! Please refresh the page to use them.")
        
        return
    
    # Load models and data
    assets = load_models()
    
    if not assets or not assets["models"]:
        st.error("Failed to load models. Please run the model training script first.")
        return
    
    # Sidebar for inputs
    st.sidebar.header("Prediction Parameters")
    
    # Date selection
    prediction_date = st.sidebar.date_input(
        "Select Date",
        value=datetime.datetime.now().date()
    )
    
    # Hour selection
    hour = st.sidebar.slider(
        "Hour of Day (24-hour format)",
        min_value=0,
        max_value=23,
        value=datetime.datetime.now().hour
    )
    
    # Extract features from date
    month = prediction_date.month
    day = prediction_date.day
    day_of_week = prediction_date.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Button to make prediction
    predict_button = st.sidebar.button("Generate Prediction")
    
    # Display summary statistics
    st.header("Historical Data Summary")
    
    stats = assets["summary_statistics"]
    
    if stats:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Violation Locations")
            locations_data = pd.Series(stats.get("top_locations", {}))
            if not locations_data.empty:
                st.bar_chart(locations_data)
            
            st.subheader("Violations by Hour of Day")
            time_data = pd.Series(stats.get("time_distribution", {}))
            if not time_data.empty:
                st.line_chart(time_data)
        
        with col2:
            st.subheader("Vehicle Type Distribution")
            vehicle_data = pd.Series(stats.get("vehicle_distribution", {}))
            if not vehicle_data.empty:
                st.bar_chart(vehicle_data)
            
            st.subheader("Violations by Day of Week")
            day_data = pd.Series(stats.get("day_distribution", {}))
            if not day_data.empty:
                st.bar_chart(day_data)
    else:
        st.info("Summary statistics not available.")
    
    # Make predictions when button is clicked
    if predict_button:
        st.header(f"Prediction for {prediction_date.strftime('%A, %B %d, %Y')} at {hour}:00")
        
        with st.spinner("Generating predictions..."):
            prediction = make_prediction(
                assets["models"],
                assets["scaler"],
                assets["label_mappings"],
                month,
                day,
                day_of_week,
                hour,
                is_weekend
            )
        
        if prediction:
            # Display predictions in a dashboard
            col1, col2, col3 = st.columns(3)
            
            # Location prediction
            with col1:
                st.subheader("Location Prediction")
                st.markdown(f"**Most Likely Location:** {prediction.get('location', 'N/A')}")
                
                # Top locations with probabilities
                if "location_top3" in prediction:
                    st.write("Top 3 Probable Locations:")
                    
                    # Create bars to visualize probabilities
                    for loc_data in prediction["location_top3"]:
                        label = loc_data["label"]
                        prob = loc_data["probability"]
                        st.write(f"Location {label}: {prob:.2f}")
                        st.progress(float(prob))
            
            # Vehicle type prediction
            with col2:
                st.subheader("Vehicle Type Prediction")
                st.markdown(f"**Most Likely Vehicle Type:** {prediction.get('vehicle_type', 'N/A')}")
                
                # Top vehicle types with probabilities
                if "vehicle_type_top3" in prediction:
                    st.write("Top 3 Probable Vehicle Types:")
                    
                    # Create bars to visualize probabilities
                    for veh_data in prediction["vehicle_type_top3"]:
                        label = veh_data["label"]
                        prob = veh_data["probability"]
                        st.write(f"Vehicle Type {label}: {prob:.2f}")
                        st.progress(float(prob))
            
            # Violation type prediction
            with col3:
                st.subheader("Violation Type Prediction")
                st.markdown(f"**Most Likely Violation Type:** {prediction.get('violation_type', 'N/A')}")
                
                # Top violation types with probabilities
                if "violation_type_top3" in prediction:
                    st.write("Top 3 Probable Violation Types:")
                    
                    # Create bars to visualize probabilities
                    for vio_data in prediction["violation_type_top3"]:
                        label = vio_data["label"]
                        prob = vio_data["probability"]
                        st.write(f"Violation Type {label}: {prob:.2f}")
                        st.progress(float(prob))
            
            # High violation times
            st.subheader("Hours with Highest Violation Rates")
            
            if "high_violation_hours" in prediction:
                top_hours = prediction["high_violation_hours"]
                st.markdown(f"**Peak Hours:** {', '.join([f'{h}:00' for h in top_hours])}")
                
                # Visualize hourly distribution if available
                if "hour_predictions" in prediction:
                    hour_data = prediction["hour_predictions"]
                    hours = list(range(24))
                    counts = [hour_data.get(h, 0) for h in hours]
                    
                    # Create DataFrame for visualization
                    hour_df = pd.DataFrame({
                        'Hour': hours,
                        'Predicted Violation Count': counts
                    })
                    
                    # Create a bar chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(x='Hour', y='Predicted Violation Count', data=hour_df, ax=ax)
                    
                    # Highlight the peak hours
                    for h in top_hours:
                        ax.axvline(x=hours.index(h), color='red', linestyle='--', alpha=0.7)
                    
                    ax.set_title('Predicted Violation Count by Hour')
                    ax.set_xlabel('Hour of Day (24-hour format)')
                    ax.set_ylabel('Violation Count')
                    
                    st.pyplot(fig)
            
            # Summary card
            st.subheader("Prediction Summary")
            
            # Create a markdown table
            summary_md = f"""
            | Prediction | Value |
            | --- | --- |
            | Date | {prediction_date.strftime('%Y-%m-%d')} |
            | Time | {hour}:00 |
            | Day of Week | {prediction_date.strftime('%A')} |
            | Most Likely Location | {prediction.get('location', 'N/A')} |
            | Most Likely Vehicle Type | {prediction.get('vehicle_type', 'N/A')} |
            | Most Likely Violation Type | {prediction.get('violation_type', 'N/A')} |
            | Peak Hours | {', '.join([f'{h}:00' for h in prediction.get('high_violation_hours', [])])} |
            """
            
            st.markdown(summary_md)
        else:
            st.error("Failed to generate predictions.")

# Run the app
if __name__ == "__main__":
    main() 