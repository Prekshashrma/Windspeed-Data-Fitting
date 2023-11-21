import pandas as pd
import numpy as np
from scipy.stats import lognorm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the CSV file into a Pandas DataFrame
file_path = 'C:/college/asm_assg/Rajasthan1/windspeed.csv'
df = pd.read_csv(file_path)

# Assuming your wind speed data is in a column named 'windspeed'
wind_speed_data = df['windspeed'].values

# Define negative log-likelihood function for Lognormal distribution
def neg_log_likelihood(params):
    s, loc, scale = params
    return -np.sum(lognorm.logpdf(wind_speed_data, s=s, loc=loc, scale=scale))

# Initial parameter guess
initial_guess = [1, 0, 1]  # s, loc, scale

# Minimize negative log-likelihood
result = minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B')

# Estimated parameters
s_est, loc_est, scale_est = result.x
print("Estimated s (shape):", s_est)
print("Estimated loc (location):", loc_est)
print("Estimated scale:", scale_est)

# Plot the histogram of the data and the fitted Lognormal distribution
plt.hist(wind_speed_data, bins=50, density=True, alpha=0.6, color='g', label='Histogram')
x = np.linspace(min(wind_speed_data), max(wind_speed_data), 100)
plt.plot(x, lognorm.pdf(x, s=s_est, loc=loc_est, scale=scale_est), 'r-', lw=2, label='Lognormal Fit')
plt.title('Wind Speed Distribution and Lognormal Fit')
plt.xlabel('Wind Speed')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
