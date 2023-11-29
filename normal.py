import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.stats import chi2

# Load data from a CSV file
file_path = 'C:/college/asm_assg/Rajasthan1/windspeed.csv'
df = pd.read_csv(file_path)

# Assuming your data is in a column named 'data'
data = df['windspeed'].values

# Plot histogram of the data
hist, bin_edges = np.histogram(data, bins=50, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Overlay a fitted normal distribution
def normal_distribution(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

# Fit the normal distribution to the data
params, covariance = curve_fit(normal_distribution, bin_centers, hist)

# Plot the fitted normal distribution
x_range = np.linspace(min(data) - 3 * params[1], max(data) + 3 * params[1], 100)
plt.hist(data, bins=50, density=True, alpha=0.6, color='blue', label='Histogram')
plt.plot(x_range, normal_distribution(x_range, *params), 'r-', lw=2, label='Fitted Gaussian')

# # Calculate expected frequencies using the fitted Gaussian distribution
# expected_frequencies = normal_distribution(bin_centers, *params) * np.sum(hist)
#
# # Calculate chi-squared test statistic
# chi2_statistic = np.sum((hist - expected_frequencies)**2 / expected_frequencies)
#
# # Degrees of freedom
# degrees_of_freedom = len(bin_centers) - len(params)
#
# # Significance level
# alpha = 0.05
#
# # Critical value from chi-squared distribution
# critical_value = chi2.ppf(1 - alpha, degrees_of_freedom)
#
# # Interpretation
# print(f'Chi-squared Statistic: {chi2_statistic}')
# print(f'Degrees of Freedom: {degrees_of_freedom}')
# print(f'Critical Value: {critical_value}')
#
# if chi2_statistic > critical_value:
#     print("The null hypothesis (data follows the Gaussian distribution) is rejected.")
# else:
#     print("The null hypothesis cannot be rejected. The data is consistent with the Gaussian distribution.")

# Display the plot
plt.title('Gaussian Distribution and Fitted Gaussian')
plt.xlabel('Data Values')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

from scipy.stats import kstest, norm

# Perform the Kolmogorov-Smirnov test for normal distribution
ks_statistic_norm, ks_p_value_norm = kstest(data, 'norm', args=(np.mean(data), np.std(data)))

# Print the test results for normal distribution
print(f"KS Statistic (Normal): {ks_statistic_norm}")
print(f"P-value (Normal): {ks_p_value_norm}")

# Interpret the results for normal distribution
if ks_p_value_norm < 0.05:
    print("The null hypothesis (data follows a normal distribution) is rejected.")
else:
    print("The null hypothesis cannot be rejected. The data is consistent with a normal distribution.")

