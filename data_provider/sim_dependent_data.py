import pandas as pd
import numpy as np
import os

# Parameters
start_time = "2022-12-31 00:00:00"  # Start of the time series
end_time = "2023-12-31 23:00:00"    # End of the time series
num_features = 2                    # Number of features
autocorr_coeff = 0.5                # Base autocorrelation coefficient
window_size = 24                    # Number of past values influencing the current value
cross_coef = .8                     # Cross-correlation coefficient 
save_path = "../dataset/simulate/"


# Configurable periods with amplitudes
periodic_components_1 = {
    12: 0.1,     # Weekly cycle (amplitude 0.5)
    6: 0.05       # Half-daily cycle (amplitude 0.8)
}
periodic_components_2 = {
    8: 0.1,     # Weekly cycle (amplitude 0.5)
    4: 0.05       # Half-daily cycle (amplitude 0.8)
}
periodic_components_3 = {
    8: 0.02,      # Daily cycle (amplitude 1.0)
    6: 0.05,     # Weekly cycle (amplitude 0.5)
    3: 0.1       # Half-daily cycle (amplitude 0.8)
}
pcmps ={1:periodic_components_1,2:periodic_components_2,3:periodic_components_3}
# Generate a range of hourly timestamps
timestamps = pd.date_range(start=start_time, end=end_time, freq="H")
n = len(timestamps)

# Initialize the dataset
data = {"date": timestamps}

# Generate each feature with autocorrelation and periodic components
for feature_id in range(1, num_features + 1):
    # Initialize the time series with noise
    series = np.random.randn(n)
    # Configure the initial trend (increase or decrease)
    trend_type = np.random.choice(["increase", "decrease"])
    if trend_type == "increase":
        series[:window_size] = np.linspace(0, 1, window_size) + np.random.randn(window_size) * 0.1
    else:  # Decrease
        series[:window_size] = np.linspace(1, 0, window_size) + np.random.randn(window_size) * 0.1
        
    
    # Weighted sum of the last 'window_size' values
    weights = np.exp(-np.arange(window_size) / window_size)  # Exponential decay weights
    weights /= weights.sum()  # Normalize weights
    pows = np.random.uniform(size=window_size)
    weights = weights * pows
    
    # Apply the configurable autocorrelation formula
    for t in range(window_size, n):

        past_values = series[t - window_size:t]
        series[t] = autocorr_coeff * np.dot(weights, past_values) + np.random.randn() * (1 - autocorr_coeff)
    
    # Add periodic components
    periodic_components = pcmps[feature_id]
    for period, amplitude in periodic_components.items():
        series += amplitude * np.sin(2 * np.pi * np.arange(n) / period)
    
    # Add the feature to the dataset
    data[f"feature_{feature_id}"] = series

# Create a DataFrame
df = pd.DataFrame(data)    

weights_a = np.random.uniform(size=window_size)
weights_a /= weights_a.sum()  


series = df["feature_2"].values

# Apply the configurable autocorrelation formula
for t in range(window_size, n):

    past_values_a = df["feature_1"].values[t - window_size:t]

    series[t] = (1-cross_coef)*series[t] + cross_coef * np.dot(weights_a, past_values_a) 
            
# Add the feature to the dataset
df["feature_2"] = series

# Save to CSV, excluding the first 24 hours
df[24:].to_csv(os.path.join(save_path,"dependent_alpha"+str(int(autocorr_coeff*100))+"_gamma"+str(int(cross_coef*100))+".csv"), index=False)
