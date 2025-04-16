import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.multioutput import MultiOutputClassifier
import joblib
import datetime
import os
import json

# Set random seed for reproducibility
np.random.seed(42)

class ComprehensivePredictionModel:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.feature_importance = {}
        self.output_dir = "comprehensive_model"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_data(self):
        """Load and preprocess the offense dataset"""
        print("Loading and preprocessing data...")
        
        # Load the cleaned dataset
        self.data = pd.read_csv("dataset/offense_set_cleaned.csv")
        
        # Convert datetime column
        self.data['DateTime'] = pd.to_datetime(self.data['DateTime'])
        
        # Extract time-based features
        self.data['year'] = self.data['DateTime'].dt.year
        self.data['month'] = self.data['DateTime'].dt.month
        self.data['day'] = self.data['DateTime'].dt.day
        self.data['day_of_week'] = self.data['DateTime'].dt.dayofweek
        self.data['hour'] = self.data['DateTime'].dt.hour
        self.data['is_weekend'] = self.data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Create time bins (morning, afternoon, evening, night)
        self.data['time_of_day'] = pd.cut(
            self.data['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        
        print(f"Data loaded successfully: {len(self.data)} records")
        
        # Generate summary statistics
        self._generate_summary_statistics()
        
        return self.data
    
    def _generate_summary_statistics(self):
        """Generate and save summary statistics about the dataset"""
        print("Generating summary statistics...")
        
        # 1. Top violation locations
        top_locations = self.data['Location'].value_counts().head(10)
        
        # 2. Vehicle type distribution
        vehicle_distribution = self.data['Vehicle_type'].value_counts()
        
        # 3. Violation type (Offence_ID) distribution
        offense_distribution = self.data['Offence_ID'].value_counts().head(10)
        
        # 4. Time of day analysis
        time_distribution = self.data['hour'].value_counts().sort_index()
        
        # 5. Day of week analysis
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_distribution = self.data['day_of_week'].value_counts().sort_index()
        day_distribution.index = day_names
        
        # Save summary statistics
        # Create figures
        plt.figure(figsize=(20, 15))
        
        # Plot top locations
        plt.subplot(2, 2, 1)
        sns.barplot(x=top_locations.index, y=top_locations.values)
        plt.title('Top 10 Violation Locations')
        plt.xlabel('Location ID')
        plt.ylabel('Number of Violations')
        
        # Plot vehicle types
        plt.subplot(2, 2, 2)
        sns.barplot(x=vehicle_distribution.index, y=vehicle_distribution.values)
        plt.title('Vehicle Type Distribution')
        plt.xlabel('Vehicle Type')
        plt.ylabel('Number of Violations')
        
        # Plot time distribution
        plt.subplot(2, 2, 3)
        sns.lineplot(x=time_distribution.index, y=time_distribution.values)
        plt.title('Violations by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Number of Violations')
        
        # Plot day of week distribution
        plt.subplot(2, 2, 4)
        sns.barplot(x=day_distribution.index, y=day_distribution.values)
        plt.xticks(rotation=45)
        plt.title('Violations by Day of Week')
        plt.xlabel('Day')
        plt.ylabel('Number of Violations')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/summary_statistics.png")
        plt.close()
        
        # Save as CSV
        summary_data = {
            "top_locations": top_locations.to_dict(),
            "vehicle_distribution": vehicle_distribution.to_dict(),
            "offense_distribution": offense_distribution.to_dict(),
            "time_distribution": time_distribution.to_dict(),
            "day_distribution": day_distribution.to_dict()
        }
        
        with open(f"{self.output_dir}/summary_statistics.json", 'w') as f:
            json.dump(summary_data, f, indent=4)
            
        print("Summary statistics saved to comprehensive_model/summary_statistics.json")
    
    def prepare_features(self):
        """Prepare features for model training"""
        print("Preparing features for model training...")
        
        # Define features and targets
        features = [
            'month', 'day', 'day_of_week', 'hour', 'is_weekend',
        ]
        
        targets = {
            'location': 'Location',
            'vehicle_type': 'Vehicle_type',
            'violation_type': 'Offence_ID'
        }
        
        # Create feature matrix X
        X = self.data[features].copy()
        
        # Create label encoders for categorical features
        self.encoders = {}
        
        # Create target vectors
        y = {}
        for target_name, column in targets.items():
            # Label encode the target column
            encoder = LabelEncoder()
            encoded_target = encoder.fit_transform(self.data[column])
            y[target_name] = encoded_target
            self.encoders[target_name] = encoder
        
        # Save original label mappings
        label_mappings = {}
        for target_name, encoder in self.encoders.items():
            mapping = {}
            for i, label in enumerate(encoder.classes_):
                # Handle both numeric and string labels
                try:
                    mapping[str(i)] = int(label)
                except (ValueError, TypeError):
                    mapping[str(i)] = str(label)
            
            label_mappings[target_name] = mapping
        
        # Save label mappings
        with open(f"{self.output_dir}/label_mappings.json", 'w') as f:
            json.dump(label_mappings, f, indent=4)
        
        # Split into train and test sets
        X_train, X_test, y_train_dict, y_test_dict = {}, {}, {}, {}
        
        for target_name, target_vector in y.items():
            X_train[target_name], X_test[target_name], y_train_dict[target_name], y_test_dict[target_name] = train_test_split(
                X, target_vector, test_size=0.2, random_state=42
            )
        
        # Scale the features
        self.scaler = StandardScaler()
        
        # Store the split data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train_dict
        self.y_test = y_test_dict
        self.feature_names = features
        
        print("Features prepared successfully")
        return X_train, X_test, y_train_dict, y_test_dict
    
    def train_models(self):
        """Train individual models for each prediction task"""
        print("Training prediction models...")
        
        if self.X_train is None:
            raise ValueError("Features not prepared. Call prepare_features() first.")
        
        # Train models for each target
        for target_name in self.y_train.keys():
            print(f"Training model for {target_name}...")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(self.X_train[target_name])
            
            # Train model based on target type
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Train the model
            model.fit(X_train_scaled, self.y_train[target_name])
            
            # Store the model
            self.models[target_name] = model
            
                    # Store feature importance
        self.feature_importance[target_name] = {
            feature: float(importance) for feature, importance in zip(
                self.feature_names, model.feature_importances_
            )
        }
        
        # Train a regression model for high violation time prediction
        print("Training model for high violation time prediction...")
        
        # Group by hour and count violations
        hourly_counts = self.data.groupby('hour')['Offence_ID'].count().reset_index()
        hourly_counts.columns = ['hour', 'violation_count']
        
        # Create features for time prediction
        time_X = pd.DataFrame({'hour': range(24)})
        time_y = pd.merge(time_X, hourly_counts, on='hour', how='left')['violation_count'].fillna(0)
        
        # Train regression model for time prediction
        time_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        time_model.fit(time_X, time_y)
        
        # Store the time model
        self.models['high_violation_time'] = time_model
        
        print("All models trained successfully")
        
        # Plot feature importance
        self._plot_feature_importance()
        
        # Save the models
        self._save_models()
        
        return self.models
    
    def _plot_feature_importance(self):
        """Plot feature importance for each model"""
        plt.figure(figsize=(20, 15))
        
        for i, (target_name, importance_dict) in enumerate(self.feature_importance.items(), 1):
            plt.subplot(2, 2, i)
            
            features = list(importance_dict.keys())
            importances = list(importance_dict.values())
            
            # Sort by importance
            sorted_idx = np.argsort(importances)
            plt.barh(np.array(features)[sorted_idx], np.array(importances)[sorted_idx])
            
            plt.title(f'Feature Importance for {target_name.replace("_", " ").title()} Prediction')
            plt.xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_importance.png")
        plt.close()
    
    def _save_models(self):
        """Save trained models and metadata"""
        print("Saving trained models...")
        
        # Save models
        for target_name, model in self.models.items():
            joblib.dump(model, f"{self.output_dir}/{target_name}_model.joblib")
        
        # Save scaler
        joblib.dump(self.scaler, f"{self.output_dir}/scaler.joblib")
        
        # Save feature importance
        with open(f"{self.output_dir}/feature_importance.json", 'w') as f:
            json.dump(self.feature_importance, f, indent=4)
        
        print("Models saved successfully to comprehensive_model/ directory")
    
    def evaluate_models(self):
        """Evaluate the performance of trained models"""
        print("Evaluating model performance...")
        
        evaluation_results = {}
        
        for target_name, model in self.models.items():
            if target_name == 'high_violation_time':
                continue  # Skip evaluation for time regression model
                
            # Scale features
            X_test_scaled = self.scaler.transform(self.X_test[target_name])
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Compute metrics
            report = classification_report(
                self.y_test[target_name], 
                y_pred, 
                output_dict=True,
                zero_division=0
            )
            
            # Store evaluation results
            evaluation_results[target_name] = report
            
            # Print classification report
            print(f"\nClassification Report for {target_name}:")
            print(classification_report(
                self.y_test[target_name], 
                y_pred,
                zero_division=0
            ))
            
            # Generate confusion matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(self.y_test[target_name], y_pred)
            
            # Plot only if the number of classes is manageable
            num_classes = len(np.unique(self.y_test[target_name]))
            if num_classes <= 20:
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=range(num_classes),
                    yticklabels=range(num_classes)
                )
                plt.title(f'Confusion Matrix for {target_name}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.savefig(f"{self.output_dir}/{target_name}_confusion_matrix.png")
                plt.close()
        
        # Save evaluation results
        with open(f"{self.output_dir}/evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        
        print("Model evaluation completed")
        return evaluation_results
    
    def predict(self, month, day, day_of_week, hour, is_weekend=None):
        """Make predictions with the trained models"""
        if not self.models:
            raise ValueError("Models not trained. Train models first.")
        
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
        features_scaled = self.scaler.transform(features)
        
        # Make predictions for each target
        predictions = {}
        
        for target_name, model in self.models.items():
            if target_name == 'high_violation_time':
                continue  # Skip this model for now
                
            # Get prediction
            pred_encoded = model.predict(features_scaled)[0]
            
            # Convert encoded prediction back to original label
            pred_label = self.encoders[target_name].inverse_transform([pred_encoded])[0]
            
            # Store prediction
            predictions[target_name] = int(pred_label)
        
        # Get high violation times
        hour_predictions = {}
        for h in range(24):
            hour_vector = np.array([[h]])
            violation_count = self.models['high_violation_time'].predict(hour_vector)[0]
            hour_predictions[h] = violation_count
        
        # Get top 3 hours with highest violation counts
        top_hours = sorted(hour_predictions.items(), key=lambda x: x[1], reverse=True)[:3]
        predictions['high_violation_hours'] = [hour for hour, _ in top_hours]
        
        # Add probabilities for location and vehicle type
        for target_name in ['location', 'vehicle_type', 'violation_type']:
            # Get prediction probabilities
            if target_name in self.models:
                pred_proba = self.models[target_name].predict_proba(features_scaled)[0]
                
                # Get top 3 classes with highest probabilities
                top_classes_idx = pred_proba.argsort()[-3:][::-1]
                top_classes = self.encoders[target_name].inverse_transform(top_classes_idx)
                top_probs = pred_proba[top_classes_idx]
                
                # Store top predictions with probabilities
                predictions[f"{target_name}_top3"] = [
                    {"label": int(label), "probability": float(prob)} 
                    for label, prob in zip(top_classes, top_probs)
                ]
        
        return predictions

# Main execution
if __name__ == "__main__":
    # Create instance
    model = ComprehensivePredictionModel()
    
    # Load data
    model.load_data()
    
    # Prepare features
    model.prepare_features()
    
    # Train models
    model.train_models()
    
    # Evaluate models
    model.evaluate_models()
    
    # Example prediction
    today = datetime.datetime.now()
    prediction = model.predict(
        month=today.month,
        day=today.day,
        day_of_week=today.weekday(),
        hour=today.hour
    )
    
    print("\nSample Prediction for current date/time:")
    print(json.dumps(prediction, indent=2))
    
    print("\nModel training and evaluation complete. All assets saved to comprehensive_model/ directory.") 