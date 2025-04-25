import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

# Ignore specific warnings from statsmodels if they occur
warnings.filterwarnings("ignore", message="A date index has been provided, but it has no associated frequency information")
warnings.filterwarnings("ignore", message="Provided frequency is different from detected frequency")

# Load the cleaned data
cleaned_file_path = "dataset/offense_set_cleaned.csv"
df = pd.read_csv(cleaned_file_path, parse_dates=['DateTime'])

# Set DateTime as index
df = df.set_index('DateTime')
df.sort_index(inplace=True)

print("Cleaned data loaded and indexed by DateTime.")
print(df.info())
print(df.head())

# Resample to get daily counts
# We count 'Offence_ID' but any column would work as we just need row counts per day
daily_counts = df.resample('D')['Offence_ID'].count()

# Fill missing days (if any from resampling) with 0, as no offenses occurred
daily_counts = daily_counts.fillna(0)

print("\nDaily offense counts:")
print(daily_counts.head())

# --- Seasonal Decomposition ---
# We need to determine the period (seasonality). Common periods for daily data are 7 (weekly),
# 30 (monthly approximation), or 365 (yearly).
# Let's start by assuming weekly seasonality (period=7).
period = 7

print(f"\nPerforming seasonal decomposition with period={period}...")

# Check if we have enough data for the chosen period
if len(daily_counts) < 2 * period:
    print(f"Warning: Not enough data (length {len(daily_counts)}) for seasonal decomposition with period={period}. Need at least {2 * period} observations.")
    # Handle insufficient data case - maybe skip decomposition or choose a different period
    decomposition = None # Set decomposition to None or handle appropriately
else:
    # Use multiplicative model if counts are mostly positive and variance seems to scale with level
    # Use additive model if variance seems constant
    # Additive is often safer if unsure or if there are zeros.
    decomposition = seasonal_decompose(daily_counts, model='additive', period=period)

    # Plot the decomposition
    print("Plotting decomposition...")
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.suptitle(f'Seasonal Decomposition (Period={period}, Model=Additive)', y=1.02)
    plt.tight_layout()
    plt.savefig('tsa_decomposition_weekly.png')
    print("Saved decomposition plot to tsa_decomposition_weekly.png")
    plt.close()

    # Log findings (basic example)
    print("\n--- Time Series Analysis Findings (Weekly Seasonality) ---")
    if decomposition is not None:
        # Trend analysis (e.g., is it generally increasing/decreasing?)
        trend_slope = decomposition.trend.dropna().diff().mean()
        print(f"Average daily trend slope: {trend_slope:.4f}")

        # Seasonality analysis (e.g., which day of the week has highest/lowest offenses?)
        seasonal_mean = decomposition.seasonal.head(period).mean()
        seasonal_variation = decomposition.seasonal.head(period) - seasonal_mean
        print(f"Seasonal component (deviation from mean) for first {period} days:")
        print(seasonal_variation)
        print(f"Day with highest seasonal component: {seasonal_variation.idxmax().strftime('%A')}")
        print(f"Day with lowest seasonal component: {seasonal_variation.idxmin().strftime('%A')}")

        # Residual analysis (e.g., how much noise is left?)
        residual_std = decomposition.resid.std()
        print(f"Standard deviation of residuals: {residual_std:.4f}")
    else:
        print("Decomposition could not be performed due to insufficient data.")

print("\nTime series analysis complete.") 