import pandas as pd
import numpy as np
from scipy.stats import weibull_min
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the CSV file into a Pandas DataFrame
file_path = 'C:/college/asm_assg/Rajasthan1/windspeed.csv'
df = pd.read_csv(file_path)

# Assuming your wind speed data is in a column named 'windspeed'
wind_speed_data = df['windspeed'].values

# Define negative log-likelihood function for Weibull distribution
def neg_log_likelihood(params):
    lambda_, k = params
    return -np.sum(weibull_min.logpdf(wind_speed_data, c=k, scale=lambda_))

# Initial parameter guess
initial_guess = [1, 1]

# Minimize negative log-likelihood
result = minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B')

# Estimated parameters
lambda_est, k_est = result.x
print("Estimated lambda:", lambda_est)
print("Estimated k:", k_est)

# Plot the histogram of the data and the fitted Weibull distribution
plt.hist(wind_speed_data, bins=50, density=True, alpha=0.6, color='g', label='Histogram')
x = np.linspace(min(wind_speed_data), max(wind_speed_data), 100)
plt.plot(x, weibull_min.pdf(x, c=k_est, scale=lambda_est), 'r-', lw=2, label='Weibull Fit')
plt.title('Wind Speed Distribution and Weibull Fit')
plt.xlabel('Wind Speed')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
