# Project Plan: Offense Data Analysis, Modeling, and Streamlit App

## Phase 1: Data Exploration and Preprocessing

1.  **Load and Inspect Data:** Read the `dataset/offense set.csv` file into a pandas DataFrame. Get initial insights like column names, data types, number of rows/columns, and view the first few rows (`.head()`, `.info()`).
2.  **Descriptive Statistics:** Generate summary statistics for numerical columns (`.describe()`) to understand central tendency and dispersion.
3.  **Missing Value Analysis:** Check for missing values (`.isnull().sum()`) and decide on a strategy (imputation or removal).
4.  **Data Cleaning:** Apply the chosen strategy for missing values. Correct any obvious data entry errors if found. Convert columns to appropriate data types (e.g., dates/times to datetime objects).
5.  **Exploratory Data Analysis (EDA):**
    *   Visualize distributions of key numerical features (histograms, box plots).
    *   Analyze categorical features (value counts, bar charts).
    *   Explore relationships between features and the potential target variable (scatter plots, correlation matrices).

## Phase 2: Time Series Analysis

6.  **Time Indexing:** Identify the time/date column, ensure it's in datetime format, and set it as the DataFrame index.
7.  **Resampling (Optional):** Resample the data to a suitable frequency (e.g., daily, weekly, monthly) if necessary.
8.  **Seasonality and Trend Analysis:**
    *   Visualize the time series data to observe overall trends and potential seasonality.
    *   Use statistical methods (e.g., `statsmodels.tsa.seasonal_decompose`) to decompose the time series into trend, seasonal, and residual components.
    *   Plot the decomposed components.
    *   Log findings about seasonality and repeating patterns.

## Phase 3: Feature Engineering and Model Preparation

9.  **Feature Engineering:** Create new features that might be useful for modeling (e.g., lag features, rolling statistics, time-based features like day of the week, month).
10. **Data Splitting:** Split the data into training, validation, and test sets, ensuring the temporal order is maintained for time series data.
11. **Scaling/Encoding:**
    *   Scale numerical features (e.g., using `MinMaxScaler` or `StandardScaler`).
    *   Encode categorical features (e.g., using One-Hot Encoding). Apply the same scaler/encoder fitted on the training data to the validation and test sets.
12. **PyTorch Data Preparation:** Convert the processed data into PyTorch Tensors and create `DataLoader` instances for batching during training. For time series models like LSTMs, structure the data into sequences.

## Phase 4: PyTorch Model Development

13. **Model Selection:** Choose an appropriate PyTorch model architecture based on the problem (e.g., LSTM, GRU for sequence prediction; MLP for regression/classification if time isn't the primary prediction target).
14. **Model Definition:** Define the model class in PyTorch.
15. **Loss Function and Optimizer:** Select a suitable loss function (e.g., MSE for regression, CrossEntropyLoss for classification) and optimizer (e.g., Adam).
16. **Training Loop:** Implement the training loop, iterating through epochs and batches, performing forward pass, calculating loss, backward pass, and updating weights. Include validation checks within the loop.
17. **Evaluation:** Evaluate the trained model on the test set using appropriate metrics (e.g., RMSE, MAE for regression; accuracy, F1-score for classification). Save the trained model.

## Phase 5: Streamlit Application

18. **Create App Structure:** Set up a new Python file (e.g., `app.py`) for the Streamlit application.
19. **Load Assets:** Load the preprocessed data (or parts of it), the trained PyTorch model, and any necessary scalers/encoders.
20. **UI Design:**
    *   Add titles and descriptions.
    *   Display key EDA findings and visualizations (using libraries like Matplotlib, Seaborn, Plotly integrated with Streamlit).
    *   Show the time series analysis results (decomposition plots).
    *   Create an interface for users to potentially input data or select parameters for prediction (if applicable).
21. **Prediction Interface:** Implement the logic to take user input (if any), preprocess it using the saved scalers/encoders, feed it to the loaded PyTorch model, and display the prediction.
22. **Deployment (Optional):** Prepare necessary files (like `requirements.txt`) for deploying the Streamlit app. 