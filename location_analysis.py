import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ---- 1. Load and Analyze Data ----
print("Loading and analyzing offense data by location...")
df = pd.read_csv("dataset/offense_set_cleaned.csv", parse_dates=['DateTime'])

# Location distribution analysis
location_counts = df.groupby('Location')['Offence_ID'].count().sort_values(ascending=False)
print(f"\nTotal unique locations: {len(location_counts)}")
print("\nTop 10 locations by violation count:")
print(location_counts.head(10))

# Plot top locations 
plt.figure(figsize=(12, 6))
top_n = 20
location_counts.head(top_n).plot(kind='bar')
plt.title(f'Top {top_n} Locations by Number of Offenses')
plt.xlabel('Location ID')
plt.ylabel('Number of Offenses')
plt.tight_layout()
plt.savefig('location_distribution.png')
plt.close()

# Time series analysis by location
print("\nAnalyzing weekly patterns by location...")
top_locations = location_counts.head(5).index.tolist()
df_top = df[df['Location'].isin(top_locations)]

# Weekly time series by location
location_weekly = df_top.set_index('DateTime').groupby(['Location', pd.Grouper(freq='W')])['Offence_ID'].count().reset_index()

# Plot weekly trends for top locations
plt.figure(figsize=(14, 7))
for location in top_locations:
    loc_data = location_weekly[location_weekly['Location'] == location]
    plt.plot(loc_data['DateTime'], loc_data['Offence_ID'], label=f'Location {location}')

plt.title('Weekly Offense Counts for Top 5 Locations')
plt.xlabel('Date')
plt.ylabel('Number of Offenses')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('location_weekly_trends.png')
plt.close()

# ---- 2. Feature Engineering for Location Prediction ----
print("\nPerforming feature engineering for location prediction model...")

# Define high-violation threshold (locations in the top 20% are considered "high-violation")
violation_threshold = np.percentile(location_counts, 80)
print(f"High-violation threshold: {violation_threshold} offenses")

# Create a map of locations to their violation level (1 for high, 0 for low)
high_violation_locations = set(location_counts[location_counts >= violation_threshold].index)
print(f"Number of high-violation locations: {len(high_violation_locations)}")

# Add time-based features
df['year'] = df['DateTime'].dt.year
df['month'] = df['DateTime'].dt.month
df['day_of_week'] = df['DateTime'].dt.dayofweek
df['hour'] = df['DateTime'].dt.hour
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Add violation level label
df['is_high_violation_location'] = df['Location'].isin(high_violation_locations).astype(int)

# ---- 3. Create a predictive model ----
print("\nBuilding a model to predict high-violation locations...")

# Select features
features = ['Vehicle_type', 'year', 'month', 'day_of_week', 'hour', 'is_weekend']
X = pd.get_dummies(df[features], drop_first=True)
y = df['is_high_violation_location']

# Split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a RandomForest model
print("\nTraining Random Forest classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nTop 10 features for predicting high-violation locations:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Feature Importance for Predicting High-Violation Locations')
plt.tight_layout()
plt.savefig('location_feature_importance.png')
plt.close()

# ---- 4. Deeper Analysis - Location Risk Scores ----
print("\nCalculating location risk scores...")

# Calculate a risk score for each location based on offense patterns
location_risk = df.groupby('Location').agg({
    'is_high_violation_location': 'first',  # This will be the same for all rows of a location
    'Offence_ID': 'count',
    'is_weekend': ['mean', 'sum'],  # How often offenses occur on weekends
    'day_of_week': ['mean', 'std']  # Average day of week and variability
})

location_risk.columns = ['_'.join(col).strip() for col in location_risk.columns.values]
location_risk = location_risk.rename(columns={
    'is_high_violation_location_first': 'high_violation_flag',
    'Offence_ID_count': 'total_offenses',
    'is_weekend_mean': 'weekend_ratio', 
    'is_weekend_sum': 'weekend_count',
    'day_of_week_mean': 'avg_day',
    'day_of_week_std': 'day_std'
})

# Calculate additional metrics
location_risk['offense_per_day'] = location_risk['total_offenses'] / df['DateTime'].dt.date.nunique()
location_risk['risk_score'] = (
    location_risk['total_offenses'] * 0.6 + 
    location_risk['weekend_count'] * 0.2 + 
    (7 - location_risk['avg_day']) * location_risk['total_offenses'] * 0.2  # Higher weight to weekday offenses
)

# Normalize risk score
location_risk['normalized_risk'] = (location_risk['risk_score'] - location_risk['risk_score'].min()) / \
                                  (location_risk['risk_score'].max() - location_risk['risk_score'].min())

# Sort by risk score
location_risk = location_risk.sort_values('normalized_risk', ascending=False)

# Save top high-risk locations
top_risk_locations = location_risk.head(20)
print("\nTop 20 locations by risk score:")
print(top_risk_locations[['total_offenses', 'weekend_ratio', 'avg_day', 'normalized_risk']])

# Save the results
location_risk.to_csv('location_risk_scores.csv')
print("\nLocation risk scores saved to 'location_risk_scores.csv'")

# ---- 5. Save the model ----
joblib.dump(rf_model, 'location_violation_model.joblib')
print("\nModel saved to 'location_violation_model.joblib'")

# Define a function to predict high-risk locations
def predict_high_risk_locations(vehicle_type, day_of_week, hour, is_weekend=None):
    """
    Predict if a location is likely to be high-violation given certain conditions
    
    Parameters:
    - vehicle_type: int, type of vehicle
    - day_of_week: int, day of week (0=Monday, 6=Sunday)
    - hour: int, hour of day (0-23)
    - is_weekend: int or None, 1 if weekend, 0 if not, None to infer from day_of_week
    
    Returns:
    - Probability of being a high-violation location
    """
    if is_weekend is None:
        is_weekend = 1 if day_of_week >= 5 else 0
        
    # Create a feature vector
    features = {'Vehicle_type': vehicle_type, 
                'year': df['year'].mode()[0],  # Use most common year from data
                'month': df['month'].mode()[0], # Use most common month from data
                'day_of_week': day_of_week,
                'hour': hour,
                'is_weekend': is_weekend}
    
    # Convert to DataFrame and get dummies to match training data
    features_df = pd.DataFrame([features])
    features_encoded = pd.get_dummies(features_df, drop_first=True)
    
    # Ensure all columns from training exist in features
    for col in X.columns:
        if col not in features_encoded.columns:
            features_encoded[col] = 0
    
    # Keep only columns used in training
    features_encoded = features_encoded[X.columns]
    
    # Make prediction
    probability = rf_model.predict_proba(features_encoded)[0, 1]
    
    return probability

# Example usage
print("\nExample predictions for different scenarios:")
scenarios = [
    (1, 2, 14, 0),  # Vehicle type 1, Wednesday (2), 2pm, not weekend
    (1, 5, 20, 1),  # Vehicle type 1, Saturday (5), 8pm, weekend
    (2, 1, 9, 0),   # Vehicle type 2, Tuesday (1), 9am, not weekend
    (3, 6, 23, 1)   # Vehicle type 3, Sunday (6), 11pm, weekend
]

for scenario in scenarios:
    veh_type, day, hour, weekend = scenario
    risk = predict_high_risk_locations(veh_type, day, hour, weekend)
    print(f"Vehicle type {veh_type}, {'Weekend' if weekend == 1 else 'Weekday'}, " 
          f"{'Day' if 6 <= hour < 18 else 'Night'} ({hour}:00): {risk:.2%} risk of high-violation location")

print("\nLocation analysis and modeling complete.") 