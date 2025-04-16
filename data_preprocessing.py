import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the dataset
file_path = "dataset/offense set.csv"

# Load the dataset
# Use skipinitialspace=True to handle spaces after delimiters
df = pd.read_csv(file_path, skipinitialspace=True)

# Clean column names: remove leading/trailing spaces and replace spaces with underscores
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Print DataFrame info
print("DataFrame Info:")
df.info()

print("\n" + "="*50 + "\n", flush=True) # Separator

# Print the first few rows of the DataFrame
print("DataFrame Head:", flush=True)
print(df.head())

print("\n" + "="*50 + "\n", flush=True) # Separator

# --- Next steps will go here ---
# Step 2: Descriptive Statistics
print("Descriptive Statistics:", flush=True)
print(df.describe(include='all')) # Use include='all' to see stats for object columns too

print("\n" + "="*50 + "\n", flush=True) # Separator

# Step 3: Missing Value Analysis
print("Missing Values:", flush=True)
print(df.isnull().sum())

print("\n" + "="*50 + "\n", flush=True) # Separator

# Step 4: Data Cleaning (Type Conversion, Error Handling)

# --- Time and Date Processing ---

# Convert Date to string first to ensure consistency
df['Date'] = df['Date'].astype(str)

# Convert Time to string, format it to HH.MM
# Pad with leading zero if necessary (e.g., 8.5 -> 08.50)
df['Time_str'] = df['Time'].apply(lambda x: f"{int(x):02d}.{int((x % 1) * 100):02d}")

# Combine Date and formatted Time string
df['DateTime_str'] = df['Date'] + ' ' + df['Time_str']

# Define the correct format string (Year.Month.Day Hour.Minute)
format_string = '%Y.%m.%d %H.%M'

# Convert to datetime, coercing errors to NaT (Not a Time)
# This will handle invalid date/time combinations or formats gracefully
df['DateTime'] = pd.to_datetime(df['DateTime_str'], format=format_string, errors='coerce')

# Check for rows where conversion failed (resulted in NaT)
failed_conversions = df[df['DateTime'].isnull()]
print(f"Number of rows with potential datetime conversion issues: {len(failed_conversions)}")
if not failed_conversions.empty:
    print("Sample of rows with conversion issues:")
    print(failed_conversions[['Date', 'Time', 'DateTime_str']].head())

# Check if the original Time column had values >= 24, which would cause errors
problematic_times = df[df['Time'] >= 24]
print(f"\nNumber of rows with Time >= 24: {len(problematic_times)}")
if not problematic_times.empty:
    print("Sample of rows with Time >= 24:")
    print(problematic_times[['Date', 'Time', 'DateTime_str']].head())

# --- Decide on handling NaT values ---
# Drop rows where DateTime conversion failed (resulted in NaT)
original_rows = len(df)
df.dropna(subset=['DateTime'], inplace=True)
print(f"\nDropped {original_rows - len(df)} rows due to NaT DateTime values.")

# --- Clean up redundant columns ---
columns_to_drop = ['Time', 'Date', 'Time_str', 'DateTime_str']
df = df.drop(columns=columns_to_drop)
print(f"Dropped columns: {columns_to_drop}")

# --- Attempt to convert Offence_ID to numeric ---
print("\nAttempting to convert Offence_ID to numeric...")
try:
    df['Offence_ID'] = pd.to_numeric(df['Offence_ID'])
    print("Offence_ID converted to numeric successfully.")
except ValueError as e:
    print(f"Could not convert Offence_ID to numeric. Keeping as object. Error: {e}")
    # Find specific non-numeric values if needed for inspection
    non_numeric_offence_ids = df[pd.to_numeric(df['Offence_ID'], errors='coerce').isna()]['Offence_ID'].unique()
    print(f"Sample non-numeric Offence_IDs: {non_numeric_offence_ids[:10]}...") # Show first 10

# Display info again after cleaning
print("\nDataFrame Info after cleaning:", flush=True)
df.info()

# Display head after cleaning
print("\nDataFrame Head after cleaning:", flush=True)
print(df.head())

print("\n" + "="*50 + "\n", flush=True) # Separator

# Step 5: EDA
print("\nStarting Exploratory Data Analysis (EDA)...", flush=True)

# Set plot style
sns.set_style("whitegrid")

# --- Categorical Feature Analysis ---
print("\nValue Counts for Categorical Features:", flush=True)
categorical_cols = ['Offence_ID', 'Location', 'Vehicle_type', 'Officer_Detecting']
for col in categorical_cols:
    print(f"\n--- {col} ---")
    # Get top N value counts for brevity in console output
    top_n = 10
    counts = df[col].value_counts()
    print(f"Total Unique Values: {len(counts)}")
    print(f"Top {top_n} Values:")
    print(counts.head(top_n))
    print("-"*20)

# --- Visualizations ---

# Histograms for pseudo-numerical IDs (treated as categories but visualization helps)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data=df, x='Location', bins=max(df['Location']) + 1) # Bins match unique locations if possible
plt.title('Distribution of Location IDs')

plt.subplot(1, 2, 2)
sns.histplot(data=df, x='Vehicle_type', bins=max(df['Vehicle_type']) + 1)
plt.title('Distribution of Vehicle Type IDs')

plt.tight_layout()
plt.savefig('eda_histograms.png') # Save the plot
print("\nSaved histograms plot to eda_histograms.png", flush=True)
plt.close() # Close the plot to free memory

# Time Series Plot
print("\nGenerating Time Series Plot...", flush=True)
plt.figure(figsize=(15, 6))
df.set_index('DateTime').resample('D')['Offence_ID'].count().plot()
plt.title('Number of Offences Over Time (Daily)')
plt.ylabel('Number of Offences')
plt.xlabel('Date')
plt.tight_layout()
plt.savefig('eda_timeseries.png') # Save the plot
print("Saved time series plot to eda_timeseries.png", flush=True)
plt.close()

# --- Save Cleaned Data ---
cleaned_file_path = "dataset/offense_set_cleaned.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned data saved to {cleaned_file_path}", flush=True)

print("\n" + "="*50 + "\n") # Separator
print("Data preprocessing and initial EDA complete.", flush=True) 