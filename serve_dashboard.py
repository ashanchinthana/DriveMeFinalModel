import http.server
import socketserver
import os
import json
import subprocess
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

PORT = 8080

# Run the test_model.py script to get updated metrics if needed
def get_model_metrics():
    # If test_model.py hasn't been run yet or we want fresh metrics, run it
    if not os.path.exists('model_test_results.png'):
        print("Running test_model.py to generate metrics...")
        subprocess.run(['python3', 'test_model.py'])
    
    # Read the test data and model to calculate metrics
    try:
        X_test = torch.load(os.path.join("processed_data", "X_test.pt"))
        y_test = torch.load(os.path.join("processed_data", "y_test.pt"))
        
        # Define the model class (same as in train_model.py)
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
        
        # Load features and model
        with open(os.path.join("processed_data", "features.json"), 'r') as f:
            features = json.load(f)
        
        input_size = len(features)
        model = MLP(input_size)
        model.load_state_dict(torch.load("offense_predictor_mlp.pth"))
        model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = model(X_test)
        
        # Calculate metrics
        y_test_np = y_test.numpy().flatten()
        predictions_np = predictions.numpy().flatten()
        
        mse = mean_squared_error(y_test_np, predictions_np)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_np, predictions_np)
        
        # Calculate % within ±5
        within_range = np.abs(predictions_np - y_test_np) <= 5
        accuracy_within_range = np.mean(within_range) * 100
        
        return {
            "mse": f"{mse:.4f}",
            "rmse": f"{rmse:.4f}",
            "mae": f"{mae:.4f}",
            "accuracy": f"{accuracy_within_range:.2f}% of predictions within +/- 5 of actual value"
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            "mse": "Error",
            "rmse": "Error",
            "mae": "Error",
            "accuracy": "Error calculating accuracy"
        }

# Update the HTML file with actual metrics
def update_html_with_metrics():
    metrics = get_model_metrics()
    
    # Using the fixed dashboard file
    with open('model_dashboard_fixed.html', 'r') as file:
        html_content = file.read()
    
    # Replace placeholder values with actual metrics
    html_content = html_content.replace(
        'document.getElementById(\'mse-value\').textContent = "307.9146";',
        f'document.getElementById(\'mse-value\').textContent = "{metrics["mse"]}";'
    )
    html_content = html_content.replace(
        'document.getElementById(\'rmse-value\').textContent = "17.5476";',
        f'document.getElementById(\'rmse-value\').textContent = "{metrics["rmse"]}";'
    )
    html_content = html_content.replace(
        'document.getElementById(\'mae-value\').textContent = "12.1425";',
        f'document.getElementById(\'mae-value\').textContent = "{metrics["mae"]}";'
    )
    html_content = html_content.replace(
        'document.getElementById(\'accuracy-value\').textContent = \n                    "71.43% of predictions within +/- 5 of actual value";',
        f'document.getElementById(\'accuracy-value\').textContent = "{metrics["accuracy"]}";'
    )
    
    with open('model_dashboard_fixed.html', 'w') as file:
        file.write(html_content)
    
    print("Updated HTML with actual metrics.")

# Create a simple HTTP server to serve the HTML
class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        http.server.SimpleHTTPRequestHandler.end_headers(self)

if __name__ == "__main__":
    # Update the HTML with actual metrics
    update_html_with_metrics()
    
    # Start the HTTP server
    Handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving dashboard at http://localhost:{PORT}/model_dashboard_fixed.html")
        print("Make sure the FastAPI server is running with:")
        print("python -m uvicorn api:app --reload")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped. Goodbye!") 