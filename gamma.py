import numpy as np
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
file_path = 'windspeed.csv'
data = pd.read_csv(file_path)

# Assuming your windspeed data is in a column named 'windspeed'
windspeed_data = data['windspeed']

# Fit a gamma distribution to the data
shape, loc, scale = gamma.fit(windspeed_data)

# Plot the histogram of the windspeed data
plt.hist(windspeed_data, bins=50, density=True, alpha=0.6, color='g')

# Plot the fitted gamma distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = gamma.pdf(x, shape, loc, scale)
plt.plot(x, p, 'k', linewidth=2)

title = "Fit results: shape = %.2f, loc = %.2f, scale = %.2f" % (shape, loc, scale)
plt.title(title)

plt.show()

# Print the estimated parameters
print(f"Shape: {shape}")
print(f"Location: {loc}")
print(f"Scale: {scale}")

from scipy.stats import kstest

# Perform the Kolmogorov-Smirnov test
ks_statistic, ks_p_value = kstest(windspeed_data, 'gamma', args=(shape, loc, scale))

# Print the test results
print(f"KS Statistic: {ks_statistic}")
print(f"P-value: {ks_p_value}")

# Interpret the results
if ks_p_value < 0.05:
    print("The null hypothesis (that the data follows the gamma distribution) is rejected.")
else:
    print("The null hypothesis cannot be rejected. The data may follow the gamma distribution.")
