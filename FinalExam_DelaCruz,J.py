import numpy as np
import matplotlib.pyplot as plt

# Problem 1 
# Bayesian Estimation of Daily Social Media Usage
# This code simulates a Bayesian model to estimate the average daily screen time (in hours) for students.

# Generate synthetic data
np.random.seed(1)
true_mu = 5.5  # Actual mean screen time in hours/day
true_sigma = 1.3
data = np.random.normal(true_mu, true_sigma, size=100)

# Prior hyperparameters
prior_mu_mean = 4
prior_mu_precision = 1
prior_sigma_alpha = 3
prior_sigma_beta = 2

# Posterior updates
posterior_mu_precision = prior_mu_precision + len(data) / true_sigma**2
posterior_mu_mean = (prior_mu_precision * prior_mu_mean + np.sum(data)) / posterior_mu_precision

posterior_sigma_alpha = prior_sigma_alpha + len(data) / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data))**2) / 2

# Posterior sampling
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=10000)
posterior_sigma = np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000)

# Plotting
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, density=True, color='pink', edgecolor='black')
plt.title('Posterior of μ (Screen Time)')
plt.xlabel('Social Media Usage (in hours/day)')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='lightblue', edgecolor='black')
plt.title('Posterior of σ (Screen Time)')
plt.xlabel('Std Dev')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Summary stats
print("Mean of μ:", np.mean(posterior_mu))
print("SD of μ:", np.std(posterior_mu))
print("Mean of σ:", np.mean(posterior_sigma))
print("SD of σ:", np.std(posterior_sigma))

# Problem 2
# Bayesian Estimation of Average Sleep Duration
# This code simulates a Bayesian model to estimate the average sleep duration per night (in hours) for students.

# Generate synthetic data
np.random.seed(2)
true_mu = 7
true_sigma = 1.1
data = np.random.normal(true_mu, true_sigma, size=100)

# Prior hyperparameters
prior_mu_mean = 7
prior_mu_precision = 1
prior_sigma_alpha = 3
prior_sigma_beta = 2

# Posterior updates
posterior_mu_precision = prior_mu_precision + len(data) / true_sigma**2
posterior_mu_mean = (prior_mu_precision * prior_mu_mean + np.sum(data)) / posterior_mu_precision

posterior_sigma_alpha = prior_sigma_alpha + len(data) / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data))**2) / 2

# Posterior sampling
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=10000)
posterior_sigma = np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000)

# Plotting
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, density=True, color='yellow', edgecolor='black')
plt.title('Posterior of μ (Sleep Duration)')
plt.xlabel('Average Sleep Duration (in hours/night)')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='pink', edgecolor='black')
plt.title('Posterior of σ (Sleep Duration)')
plt.xlabel('Std Dev')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Summary stats
print("Mean of μ:", np.mean(posterior_mu))
print("SD of μ:", np.std(posterior_mu))
print("Mean of σ:", np.mean(posterior_sigma))
print("SD of σ:", np.std(posterior_sigma))

# Problem 3
# Bayesian Estimation of Weekly Transportation Expenses
# This code simulates a Bayesian model to estimate the weekly transportation expenses of the students.

# Generate synthetic data
np.random.seed(3)
true_mu = 400  # ₱
true_sigma = 80
data = np.random.normal(true_mu, true_sigma, size=100)

# Prior hyperparameters
prior_mu_mean = 350
prior_mu_precision = 1 / 100**2  # SD = ₱100
prior_sigma_alpha = 3
prior_sigma_beta = 2

# Posterior updates
posterior_mu_precision = prior_mu_precision + len(data) / true_sigma**2
posterior_mu_mean = (prior_mu_precision * prior_mu_mean + np.sum(data)) / posterior_mu_precision

posterior_sigma_alpha = prior_sigma_alpha + len(data) / 2
posterior_sigma_beta = prior_sigma_beta + np.sum((data - np.mean(data))**2) / 2

# Posterior sampling
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=10000)
posterior_sigma = np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000)

# Plotting
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, density=True, color='lightblue', edgecolor='black')
plt.title('Posterior of μ (Transport Cost)')
plt.xlabel('Weekly Transportation Expenses')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='yellow', edgecolor='black')
plt.title('Posterior of σ (Transport Cost)')
plt.xlabel('Std Dev')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Summary stats
print("Mean of μ:", np.mean(posterior_mu))
print("SD of μ:", np.std(posterior_mu))
print("Mean of σ:", np.mean(posterior_sigma))
print("SD of σ:", np.std(posterior_sigma))
