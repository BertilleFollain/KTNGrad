# KTNGrad

## What is this project for?
This is the companion code to the second chapter of the PhD thesis of Follain, B, which will be available online in the fourth quarter of 2024.
It contains the estimator **KTNGrad** introduced in the previously referred chapter, the code to run the experiments from the chapter
and the results of said experiments. **KTNGrad** is a method for non-parametric regression with linear feature learning, 
which consists in regularised empirical risk minimisation in RKHS with the trace norm of the sample covariance matrix of gradients as the penalty. 
See the thesis for more details. The method is available through the class KTNGrad in '/Methods/KTNGrad.py'. It is easy to use thanks to compatibility with Scikit-learn. 
The code is maintained by Bertille Follain (https://bertillefollain.netlify.app/, email address available on website). Do not 
hesitate to reach out if you need help using it.

## Organisation of the code
The regressors used in the experiments are available in the folder 'Methods', while the code corresponding to each 
experiment are available in the folder 'Experiments'.
The results of the experiments (in .pkl format) are in the folder 'Experiments_results/Results', 
while the figures can be found in the folder 'Experiments_results/Plots'. The requirements for use of the code are in 'requirements.txt'.
Note that some packages are necessary for the experiments but not to use **KTNGrad**.

## Example
The class **KTNGrad** has many parameters, which are detailed in the definition of the class. Here is a (simple) example of 
usage.
```
from Methods.KTNGrad import KTNGrad
import numpy as np
import scipy.stats
from scipy.spatial.distance import cdist

# Data generation parameters
n = 200  # number of samples
n_test = 201  # number of test samples
d = 15  # original dimension
s = 2  # dimension of hidden linear subspace

# Generate training and test data
X = np.sqrt(3) * (2 * np.random.uniform(size=(n, d)) - 1)
X_test = np.sqrt(3) * (2 * np.random.uniform(size=(n_test, d)) - 1)
p = scipy.stats.ortho_group.rvs(d)[:, 0:s]
y = np.sum(np.dot(X, p) ** 2, axis=1)
y_test = np.sum(np.dot(X_test, p) ** 2, axis=1)

# Kernel parameters
sigma = np.median(cdist(X, X, 'euclidean'))

# Initialize the KTNGrad method
method = KTNGrad(nu=1e-6, tau=0.000125, mu=1e-12, epsilon=1e-12)

# Fit the model
method.fit(X, y, delta=1e-4, max_iter=10)

# Predict on test data
y_pred = method.predict(X_test)

# Compute the scores and print the results
score = method.score(X_test, y_test)
feature_learning_score = method.feature_learning_score(p)
print('R2 score:', score)
print('Feature learning score:', feature_learning_score)
```
