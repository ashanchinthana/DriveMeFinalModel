import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import time

# --- Configuration ---
data_path = "processed_data"
MODEL_SAVE_PATH = "offense_predictor_mlp.pth"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# --- Load Data ---
print("Loading processed data...")
X_train_tensor = torch.load(os.path.join(data_path, 'X_train.pt'))
y_train_tensor = torch.load(os.path.join(data_path, 'y_train.pt'))
X_val_tensor = torch.load(os.path.join(data_path, 'X_val.pt'))
y_val_tensor = torch.load(os.path.join(data_path, 'y_val.pt'))
X_test_tensor = torch.load(os.path.join(data_path, 'X_test.pt'))
y_test_tensor = torch.load(os.path.join(data_path, 'y_test.pt'))

with open(os.path.join(data_path, 'features.json'), 'r') as f:
    features = json.load(f)
INPUT_FEATURES = len(features)

print(f"Data loaded. Input features: {INPUT_FEATURES}")

# --- Create DataLoaders ---
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Define Model (MLP) ---
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output layer for regression (single value)
        )

    def forward(self, x):
        return self.network(x)

model = MLP(INPUT_FEATURES)
print("\nModel defined:")
print(model)

# --- Loss and Optimizer ---
criterion = nn.MSELoss() # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
print("\nStarting training...")
best_val_loss = float('inf')
epochs_no_improve = 0
start_time = time.time()

for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch+1}/{EPOCHS} \t Training Loss: {train_loss:.4f} \t Validation Loss: {val_loss:.4f}')

    # Check for improvement and save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f'   -> Validation loss decreased. Saving model to {MODEL_SAVE_PATH}')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f'   -> Validation loss did not improve for {epochs_no_improve} epoch(s).')

    # Early stopping
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f'\nEarly stopping triggered after {epoch+1} epochs.')
        break

training_time = time.time() - start_time
print(f"\nTraining finished in {training_time:.2f} seconds.")

# --- Evaluation ---
print("\nLoading best model for evaluation...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval() # Set model to evaluation mode

all_targets = []
all_predictions = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        all_targets.extend(targets.numpy().flatten())
        all_predictions.extend(outputs.numpy().flatten())

# Calculate metrics
mse = mean_squared_error(all_targets, all_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(all_targets, all_predictions)

print("\nTest Set Evaluation:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")

print("\nModel training and evaluation complete.") 