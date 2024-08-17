import pickle
import numpy as np
import scipy.stats
from Methods.KTNGrad import KTNGrad
from Methods.KRR import KRR
from Methods.PyMave import PyMave
from Methods.MARS import MARS

# Parameters
seed = 0
np.random.seed(seed)
n_values = [10, 20, 50, 75, 100, 125, 150, 175]  # Different sample sizes
d_values = [3, 5, 10, 15, 20, 25, 30, 35]  # Different dimensions
fixed_d = 10
fixed_n = 175
ntest = 201
std_noise = 0.15
s = 3
repetitions = 10
max_iter = 5
# Statistical param
nu = 1e-5
mu = 1e-8
# Optimisation param
epsilon = 1e-8
delta = 1e-3


# Data generating mechanism
def generate_data(n_bis, d_bis, n_test_bis, s_bis, std_noise_bis, seed_bis):
    np.random.seed(seed_bis)
    p_bis = scipy.stats.ortho_group.rvs(d_bis)
    X_bis = np.random.rand(n_bis, d_bis) * 2 - 1
    y_bis = np.abs(np.sum(np.sin(np.dot(X_bis, p_bis)[:, 0:s_bis]), axis=1)) + std_noise_bis * np.random.randn(n_bis)
    X_test_bis = np.random.rand(n_test_bis, d_bis) * 2 - 1
    y_test_bis = np.abs(
        np.sum(np.sin(np.dot(X_test_bis, p_bis)[:, 0:s_bis]), axis=1)) + std_noise_bis * np.random.randn(n_test_bis)
    return X_bis, y_bis, X_test_bis, y_test_bis, p_bis[:, 0:s_bis]


# Initialize results storage
results = {
    method: {
        "scores": np.zeros((len(n_values) + len(d_values), repetitions)),
        "features": np.zeros((len(n_values) + len(d_values), repetitions)),
        "dimension": np.zeros((len(n_values) + len(d_values), repetitions))
    }
    for method in ['KTNGrad', 'KTNGrad, retrained', 'KRR', 'PyMave']
}

# Perform the experiment
for n_idx, n in enumerate(n_values):
    for rep in range(repetitions):
        print("n, rep:", n_idx, rep)
        # Generate data
        X, y, X_test, y_test, p = generate_data(n, fixed_d, ntest, s, std_noise, seed + rep)

        # KTNGrad
        tau = 1 / (2 * n)
        ktn_grad = KTNGrad(nu=nu, tau=tau, mu=mu, epsilon=epsilon)
        ktn_grad.fit(X, y, delta=delta, max_iter=max_iter)
        results['KTNGrad']['scores'][n_idx, rep] = ktn_grad.score(X_test, y_test)
        results['KTNGrad']['features'][n_idx, rep] = ktn_grad.feature_learning_score(p)
        results['KTNGrad']['dimension'][n_idx, rep] = ktn_grad.dimension_score(s)

        # KTNGrad, retrained
        mars = MARS()
        mars.fit(np.dot(X, ktn_grad.p_hat_), y)
        results['KTNGrad, retrained']['scores'][n_idx, rep] = mars.score(np.dot(X_test, ktn_grad.p_hat_), y_test)

        # KRR
        lambda_val = 1 / n
        krr = KRR(lambda_val=lambda_val)
        krr.fit(X, y)
        results['KRR']['scores'][n_idx, rep] = krr.score(X_test, y_test)

        # PyMave
        pymave = PyMave()
        pymave.fit(X, y)
        results['PyMave']['scores'][n_idx, rep] = pymave.score(X_test, y_test)
        results['PyMave']['features'][n_idx, rep] = pymave.feature_learning_score(p)
        results['PyMave']['dimension'][n_idx, rep] = pymave.dimension_score(s)

for d_idx, d in enumerate(d_values):
    for rep in range(repetitions):
        print("d, rep:", d_idx, rep)
        # Generate data
        X, y, X_test, y_test, p = generate_data(fixed_n, d, ntest, s, std_noise, seed + rep)

        # KTNGrad
        tau = 1 / (2 * fixed_n)
        ktn_grad = KTNGrad(nu=nu, tau=tau, mu=mu, epsilon=epsilon)
        ktn_grad.fit(X, y, delta=delta, max_iter=max_iter)
        results['KTNGrad']['scores'][len(n_values) + d_idx, rep] = ktn_grad.score(X_test, y_test)
        results['KTNGrad']['features'][len(n_values) + d_idx, rep] = ktn_grad.feature_learning_score(p)
        results['KTNGrad']['dimension'][len(n_values) + d_idx, rep] = ktn_grad.dimension_score(s)

        # KTNGrad, retrained
        mars = MARS()
        mars.fit(np.dot(X, ktn_grad.p_hat_), y)
        results['KTNGrad, retrained']['scores'][len(n_values) + d_idx, rep] = mars.score(
            np.dot(X_test, ktn_grad.p_hat_), y_test)

        # KRR
        lambda_val = 1 / fixed_n
        krr = KRR(lambda_val=lambda_val)
        krr.fit(X, y)
        results['KRR']['scores'][len(n_values) + d_idx, rep] = krr.score(X_test, y_test)

        # PyMave
        pymave = PyMave()
        pymave.fit(X, y)
        results['PyMave']['scores'][len(n_values) + d_idx, rep] = pymave.score(X_test, y_test)
        results['PyMave']['features'][len(n_values) + d_idx, rep] = pymave.feature_learning_score(p)
        results['PyMave']['dimension'][len(n_values) + d_idx, rep] = pymave.dimension_score(s)

# Save results
results_to_save = {
    'results': results,
    'n_values': n_values,
    'd_values': d_values,
    'repetitions': repetitions
}
pickle.dump(results_to_save, open('../../Experiments_results/Results/Experiment2.pkl', 'wb'))

print('Experiment2 over')
