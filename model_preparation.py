import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib
import os
import json

# Load the cleaned data
cleaned_file_path = "dataset/offense_set_cleaned.csv"
df = pd.read_csv(cleaned_file_path, parse_dates=['DateTime'])
df.sort_values('DateTime', inplace=True)

print("Loaded cleaned data.")

# --- Feature Engineering ---
print("\nStarting Feature Engineering...")

# 1. Time-based features
df['year'] = df['DateTime'].dt.year
df['month'] = df['DateTime'].dt.month
df['day'] = df['DateTime'].dt.day
df['day_of_week'] = df['DateTime'].dt.dayofweek # Monday=0, Sunday=6
df['day_of_year'] = df['DateTime'].dt.dayofyear
df['week_of_year'] = df['DateTime'].dt.isocalendar().week.astype(int)
# Consider hour/minute if granularity was higher, but we have daily data now.

# 2. Lag Features (Example: Target variable = daily counts)
# First, aggregate to daily counts (this will be our target)
df_daily = df.set_index('DateTime').resample('D')['Offence_ID'].count().reset_index()
df_daily.columns = ['DateTime', 'offense_count']
df_daily = df_daily.fillna(0) # Fill potentially missing days

# Create lag features for the daily count
lag_periods = [1, 7, 14] # Lag by 1 day, 1 week, 2 weeks
for lag in lag_periods:
    df_daily[f'count_lag_{lag}'] = df_daily['offense_count'].shift(lag)

# Add time features to daily data
df_daily['year'] = df_daily['DateTime'].dt.year
df_daily['month'] = df_daily['DateTime'].dt.month
df_daily['day'] = df_daily['DateTime'].dt.day
df_daily['day_of_week'] = df_daily['DateTime'].dt.dayofweek
df_daily['day_of_year'] = df_daily['DateTime'].dt.dayofyear
df_daily['week_of_year'] = df_daily['DateTime'].dt.isocalendar().week.astype(int)

# Drop rows with NaN values created by shift operation
initial_rows = len(df_daily)
df_daily.dropna(inplace=True)
print(f"Dropped {initial_rows - len(df_daily)} rows due to lag feature NaNs.")

# Define features (X) and target (y)
# Note: We might refine features later. Including original categoricals
# like Location/Offence_ID would require merging them back based on DateTime
# or creating aggregated daily features from them.
# For now, focusing on time features and lags of the target count.
features = [
    'count_lag_1', 'count_lag_7', 'count_lag_14',
    'year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year'
]
target = 'offense_count'

X = df_daily[features]
y = df_daily[target]

print("\nEngineered Features:")
print(X.head())
print("\nTarget Variable:")
print(y.head())

# --- Step 10: Data Splitting (Time Series Aware) ---
print("\nSplitting data into train, validation, and test sets...")

data_len = len(df_daily)
train_size = int(data_len * 0.70)
val_size = int(data_len * 0.15)
# test_size = data_len - train_size - val_size # Remaining data

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size : train_size + val_size], y[train_size : train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

print(f"Train set size: {len(X_train)} samples")
print(f"Validation set size: {len(X_val)} samples")
print(f"Test set size: {len(X_test)} samples")

# --- Step 11: Scaling Features ---
print("\nScaling features...")

scaler = StandardScaler()

# Fit scaler ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform validation and test data
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
scaler_path = 'feature_scaler.joblib'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# --- Step 12: PyTorch Data Preparation ---
print("\nCreating PyTorch DataLoaders...")

# Convert to PyTorch Tensors
# Ensure target variable y is also a tensor and potentially reshape if needed by the model (e.g., add channel dim)
# For a simple MLP/regression target, a shape of (batch_size,) or (batch_size, 1) is typical.
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1) # Add feature dimension

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 32 # Define batch size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # No shuffle for time series usually
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Created DataLoaders with batch size {batch_size}")
print(f"Train loader: {len(train_loader)} batches")
print(f"Validation loader: {len(val_loader)} batches")
print(f"Test loader: {len(test_loader)} batches")

# Optional: Check a batch shape
x_batch_sample, y_batch_sample = next(iter(train_loader))
print(f"Sample batch shapes - X: {x_batch_sample.shape}, y: {y_batch_sample.shape}")

# --- Save Tensors and DataLoaders Info ---
print("\nSaving processed data tensors...")
data_path = "processed_data"
os.makedirs(data_path, exist_ok=True)

torch.save(X_train_tensor, os.path.join(data_path, 'X_train.pt'))
torch.save(y_train_tensor, os.path.join(data_path, 'y_train.pt'))
torch.save(X_val_tensor, os.path.join(data_path, 'X_val.pt'))
torch.save(y_val_tensor, os.path.join(data_path, 'y_val.pt'))
torch.save(X_test_tensor, os.path.join(data_path, 'X_test.pt'))
torch.save(y_test_tensor, os.path.join(data_path, 'y_test.pt'))

# Save feature list as well
features_path = os.path.join(data_path, 'features.json')
with open(features_path, 'w') as f:
    json.dump(features, f)

print(f"Tensors saved to '{data_path}' directory.")
print(f"Feature list saved to '{features_path}'")

print("\nModel preparation script complete.") 