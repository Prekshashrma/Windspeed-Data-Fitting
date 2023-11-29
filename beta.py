import pandas as pd
import numpy as np
from scipy.stats import beta, kstest
import matplotlib.pyplot as plt

# Load the CSV file into a Pandas DataFrame
file_path = 'windspeed.csv'
df = pd.read_csv(file_path)

# Assuming your wind speed data is in a column named 'windspeed'
wind_speed_data = df['windspeed'].values

# Fit Beta distribution
a_beta, b_beta, loc_beta, scale_beta = beta.fit(wind_speed_data)

# Perform KS test
ks_statistic_beta, ks_p_value_beta = kstest(wind_speed_data, 'beta', args=(a_beta, b_beta, loc_beta, scale_beta))

# Print KS test results
print("KS Test Results for Beta Distribution:")
print(f"Statistic: {ks_statistic_beta}, P-value: {ks_p_value_beta}")

# Plot the histogram of the data and the fitted Beta distribution
plt.hist(wind_speed_data, bins=50, density=True, alpha=0.6, color='g', label='Histogram')

x = np.linspace(min(wind_speed_data), max(wind_speed_data), 1000)

# Plot Beta fit
plt.plot(x, beta.pdf(x, a_beta, b_beta, loc_beta, scale_beta), 'r-', lw=2, label='Beta Fit')

plt.title('Wind Speed Distribution and Fitted Beta Distribution')
plt.xlabel('Wind Speed')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
if ks_p_value_beta < 0.05:
    print("The null hypothesis (data follows the Beta distribution) is rejected.")
else:
    print("The null hypothesis cannot be rejected. The data is consistent with the Beta distribution.")