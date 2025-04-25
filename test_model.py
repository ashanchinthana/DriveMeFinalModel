import torch
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load test data
print("Loading test data...")
X_test = torch.load(os.path.join("processed_data", "X_test.pt"))
y_test = torch.load(os.path.join("processed_data", "y_test.pt"))

# Load the model
print("Loading model...")
# First, define the model class (same as in train_model.py)
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

# Load saved model state
with open(os.path.join("processed_data", "features.json"), 'r') as f:
    import json
    features = json.load(f)

input_size = len(features)
model = MLP(input_size)
model.load_state_dict(torch.load("offense_predictor_mlp.pth"))
model.eval()

# Make predictions
print("Making predictions...")
with torch.no_grad():
    predictions = model(X_test)

# Convert predictions and actual values to numpy arrays
y_test_np = y_test.numpy().flatten()
predictions_np = predictions.numpy().flatten()

# Calculate metrics
mse = mean_squared_error(y_test_np, predictions_np)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_np, predictions_np)

print(f"\nTest Set Evaluation:")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")

# Visualize predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test_np)), y_test_np, label='Actual')
plt.plot(range(len(predictions_np)), predictions_np, label='Predicted')
plt.title('Model Predictions vs Actual Values')
plt.xlabel('Time Step')
plt.ylabel('Offense Count')
plt.legend()
plt.savefig('model_test_results.png')
print("Saved visualization to model_test_results.png")

# Calculate prediction accuracy by measuring how many predictions are within +/- 5 of actual
within_range = np.abs(predictions_np - y_test_np) <= 5
accuracy_within_range = np.mean(within_range) * 100
print(f"Percentage of predictions within +/- 5 of actual: {accuracy_within_range:.2f}%") 