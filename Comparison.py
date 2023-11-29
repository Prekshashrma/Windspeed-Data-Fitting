import pandas as pd
import numpy as np
from scipy.stats import weibull_min, gamma, norm, kstest
import matplotlib.pyplot as plt

# Load the CSV file into a Pandas DataFrame
file_path = 'windspeed.csv'
df = pd.read_csv(file_path)

# Assuming your wind speed data is in a column named 'windspeed'
wind_speed_data = df['windspeed'].values

# Fit Weibull distribution
shape_weibull, loc_weibull, scale_weibull = weibull_min.fit(wind_speed_data)

# Fit Gamma distribution
shape_gamma, loc_gamma, scale_gamma = gamma.fit(wind_speed_data)

# Fit Normal distribution
mean_normal, std_normal = norm.fit(wind_speed_data)

# Perform KS tests
ks_statistic_weibull, ks_p_value_weibull = kstest(wind_speed_data, 'weibull_min', args=(shape_weibull, loc_weibull, scale_weibull))
ks_statistic_gamma, ks_p_value_gamma = kstest(wind_speed_data, 'gamma', args=(shape_gamma, loc_gamma, scale_gamma))
ks_statistic_normal, ks_p_value_normal = kstest(wind_speed_data, 'norm', args=(mean_normal, std_normal))

# Print KS test results
print("KS Test Results:")
print(f"Weibull - Statistic: {ks_statistic_weibull}, P-value: {ks_p_value_weibull}")
print(f"Gamma - Statistic: {ks_statistic_gamma}, P-value: {ks_p_value_gamma}")
print(f"Normal - Statistic: {ks_statistic_normal}, P-value: {ks_p_value_normal}")

# Plot the histogram of the data and the fitted distributions
plt.hist(wind_speed_data, bins=50, density=True, alpha=0.6, color='g', label='Histogram')

x = np.linspace(min(wind_speed_data), max(wind_speed_data), 1000)

# Plot Weibull fit
plt.plot(x, weibull_min.pdf(x, shape_weibull, loc_weibull, scale_weibull), 'r-', lw=2, label='Weibull Fit')

# Plot Gamma fit
plt.plot(x, gamma.pdf(x, shape_gamma, loc_gamma, scale_gamma), 'b-', lw=2, label='Gamma Fit')

# Plot Normal fit
plt.plot(x, norm.pdf(x, mean_normal, std_normal), 'orange', lw=2, label='Normal Fit')

plt.title('Wind Speed Distribution and Fitted Distributions')
plt.xlabel('Wind Speed')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
