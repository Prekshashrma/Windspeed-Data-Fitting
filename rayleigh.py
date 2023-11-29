import pandas as pd
import numpy as np
from scipy.stats import rayleigh, kstest
import matplotlib.pyplot as plt

# Load the CSV file into a Pandas DataFrame
file_path = 'C:/college/asm_assg/Rajasthan1/windspeed.csv'
df = pd.read_csv(file_path)

# Assuming your wind speed data is in a column named 'windspeed'
wind_speed_data = df['windspeed'].values

# Fit Rayleigh distribution
loc_rayleigh, scale_rayleigh = rayleigh.fit(wind_speed_data)

# Perform KS test
ks_statistic_rayleigh, ks_p_value_rayleigh = kstest(wind_speed_data, 'rayleigh', args=(loc_rayleigh, scale_rayleigh))

# Print KS test results
print("KS Test Results for Rayleigh Distribution:")
print(f"Statistic: {ks_statistic_rayleigh}, P-value: {ks_p_value_rayleigh}")

# Plot the histogram of the data and the fitted Rayleigh distribution
plt.hist(wind_speed_data, bins=50, density=True, alpha=0.6, color='g', label='Histogram')

x = np.linspace(min(wind_speed_data), max(wind_speed_data), 1000)

# Plot Rayleigh fit
plt.plot(x, rayleigh.pdf(x, loc_rayleigh, scale_rayleigh), 'r-', lw=2, label='Rayleigh Fit')

plt.title('Wind Speed Distribution and Fitted Rayleigh Distribution')
plt.xlabel('Wind Speed')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

if ks_p_value_rayleigh < 0.05:
    print("The null hypothesis (data follows the Rayleigh distribution) is rejected.")
else:
    print("The null hypothesis cannot be rejected. The data is consistent with the Rayleigh distribution.")