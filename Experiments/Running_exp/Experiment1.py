import pickle
import numpy as np
from scipy.spatial.distance import cdist
from Methods.GaussianKernel import GaussianKernel
from Methods.KTNGrad import KTNGrad
import scipy.stats

# Set random seed for reproducibility
seed = 76
np.random.seed(seed)

# Data generation parameters
distribution = 'uniform'
n = 200
d = 10
s = 2
std_noise = 0.1
n_test = 100

# Model parameters
nu = 1e-6
tau = 1/(8*n)
mu = 1e-16
epsilon = 1e-8
delta = 1e-6
max_iter = 10

# Generate data
X = np.random.uniform(-1, 1, (n, d))
X_test = np.random.uniform(-1, 1, (n_test, d))
p = scipy.stats.ortho_group.rvs(d)
p = p[:, 0:s]
noise = np.random.normal(0, std_noise, n)
noise_test = np.random.normal(0, std_noise, n_test)
y = np.abs(np.sum(np.sin(np.dot(X, p)), axis=1)) + noise
y_test = np.abs(np.sum(np.sin(np.dot(X_test, p)), axis=1)) + noise_test

# Initialize Gaussian Kernel
sigma = np.median(cdist(X, X, 'euclidean'))
kernel = GaussianKernel(X, sigma)

# Initialize and fit the KTNGrad model
solver = KTNGrad(nu=nu, tau=tau, mu=mu, epsilon=epsilon)
solver.fit(X, y, kernel, delta=delta, max_iter=max_iter)

# Collect results
results = {
    'primal_values': [solver.primal(theta) for theta in solver.thetas_],
    'dual_values': [solver.dual(z) for z in solver.zs_],
    'dual_gaps': [solver.primal(theta) - solver.dual(z) for theta, z in zip(solver.thetas_, solver.zs_)],
    'train_scores': [1 - ((y - np.dot(theta, solver.kernel_.pred_vector(X))) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                     for theta in solver.thetas_],
    'test_scores': [1 - ((y_test - np.dot(theta, solver.kernel_.pred_vector(X_test))) ** 2).sum() / (
            (y_test - y_test.mean()) ** 2).sum() for theta in solver.thetas_]
}

# Save results to file
pickle.dump(results, open('../../Experiments_results/Results/Experiment1.pkl', 'wb'))

print('Experiment 1 over')
