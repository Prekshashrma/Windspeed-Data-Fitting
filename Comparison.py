import pandas as pd
import numpy as np
from scipy.stats import weibull_min, gamma, norm, rayleigh, beta, kstest
import matplotlib.pyplot as plt

# Load the CSV file into a Pandas DataFrame
file_path = 'C:/college/asm_assg/Rajasthan1/windspeed.csv'
df = pd.read_csv(file_path)

# Assuming your wind speed data is in a column named 'windspeed'
wind_speed_data = df['windspeed'].values

# Fit Weibull distribution
shape_weibull, loc_weibull, scale_weibull = weibull_min.fit(wind_speed_data)

# Fit Gamma distribution
shape_gamma, loc_gamma, scale_gamma = gamma.fit(wind_speed_data)

# Fit Normal distribution
mean_normal, std_normal = norm.fit(wind_speed_data)

# Fit Rayleigh distribution
loc_rayleigh, scale_rayleigh = rayleigh.fit(wind_speed_data)

# Fit Beta distribution
a_beta, b_beta, loc_beta, scale_beta = beta.fit(wind_speed_data)

# Perform KS tests
ks_statistic_weibull, ks_p_value_weibull = kstest(wind_speed_data, 'weibull_min', args=(shape_weibull, loc_weibull, scale_weibull))
ks_statistic_gamma, ks_p_value_gamma = kstest(wind_speed_data, 'gamma', args=(shape_gamma, loc_gamma, scale_gamma))
ks_statistic_normal, ks_p_value_normal = kstest(wind_speed_data, 'norm', args=(mean_normal, std_normal))
ks_statistic_rayleigh, ks_p_value_rayleigh = kstest(wind_speed_data, 'rayleigh', args=(loc_rayleigh, scale_rayleigh))
ks_statistic_beta, ks_p_value_beta = kstest(wind_speed_data, 'beta', args=(a_beta, b_beta, loc_beta, scale_beta))

# Print KS test results
print("KS Test Results:")
print(f"Weibull - Statistic: {ks_statistic_weibull}, P-value: {ks_p_value_weibull}")
print(f"Gamma - Statistic: {ks_statistic_gamma}, P-value: {ks_p_value_gamma}")
print(f"Normal - Statistic: {ks_statistic_normal}, P-value: {ks_p_value_normal}")
print(f"Rayleigh - Statistic: {ks_statistic_rayleigh}, P-value: {ks_p_value_rayleigh}")
print(f"Beta - Statistic: {ks_statistic_beta}, P-value: {ks_p_value_beta}")

# Plot the histogram of the data and the fitted distributions
plt.hist(wind_speed_data, bins=50, density=True, alpha=0.6, color='g', label='Histogram')

x = np.linspace(min(wind_speed_data), max(wind_speed_data), 1000)

# Plot Weibull fit
plt.plot(x, weibull_min.pdf(x, shape_weibull, loc_weibull, scale_weibull), 'r-', lw=2, label='Weibull Fit')

# Plot Gamma fit
plt.plot(x, gamma.pdf(x, shape_gamma, loc_gamma, scale_gamma), 'b-', lw=2, label='Gamma Fit')

# Plot Normal fit
plt.plot(x, norm.pdf(x, mean_normal, std_normal), 'orange', lw=2, label='Normal Fit')

# Plot Rayleigh fit
plt.plot(x, rayleigh.pdf(x, loc_rayleigh, scale_rayleigh), 'purple', lw=2, label='Rayleigh Fit')

# Plot Beta fit
plt.plot(x, beta.pdf(x, a_beta, b_beta, loc_beta, scale_beta), 'cyan', lw=2, label='Beta Fit')

plt.title('Wind Speed Distribution and Fitted Distributions')
plt.xlabel('Wind Speed')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
