from Methods.GaussianKernel import GaussianKernel
import scipy.linalg
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class KTNGrad(RegressorMixin, BaseEstimator):
    """
    Implements a regressor based on kernel ridge regression where the kernel is the Gaussian kernel augmented
    with its derivative w.r.t. all the variables and a trace norm penalty on the sample matrix of
    gradients to improve feature learning.
    """

    def __init__(self, nu=1e-7, tau=1e-7, epsilon=1e-10, mu=1e-8):
        """
        Initialize the KTNGrad estimator with regularization parameters.
        :param nu: Regularization parameter for the RKHS norm.
        :param tau: Regularization parameter for the trace norm penalty.
        :param epsilon: Small constant for numerical stability.
        :param mu: Small constant for numerical stability.
        """
        self.nu = nu
        self.tau = tau
        self.epsilon = epsilon
        self.mu = mu

    def fit(self, X, y, kernel=None, delta=1e-3, max_iter=10):
        """
        Fit the model using the training data.

        :param X: Training data of shape (n_samples, n_features).
        :param y: Target values of shape (n_samples,).
        :param kernel: Precomputed kernel, if None, will be computed.
        :param delta: Convergence threshold for the optimization.
        :param max_iter: Maximal number of iterations for optimization.
        :return: self : Returns an instance of self.
        """
        if y is None:
            raise ValueError('Target variable y must be provided.')

        X, y = check_X_y(X, y, y_numeric=True)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = self.X_.shape[1]
        self.n_ = self.X_.shape[0]
        if self.n_ == 1:
            raise ValueError("Number of samples must be greater than 1. Got 1 sample.")

        if kernel is None:
            self.sigma_ = np.median(cdist(self.X_, self.X_, 'euclidean'))
            self.kernel_ = GaussianKernel(self.X_, self.sigma_)
        else:
            self.kernel_ = kernel
        self.delta_ = delta
        self.max_iter_ = max_iter

        # Computes relevant matrices once and for all
        self.a_ = np.dot(self.kernel_.U.T, self.kernel_.U) + self.kernel_.V * self.nu + self.mu * np.eye(
            np.shape(self.kernel_.V)[0])
        self.b_ = np.dot(self.kernel_.U.T, self.y_)
        self.c_ = np.tensordot(self.kernel_.W, self.kernel_.W, axes=([0, 1], [0, 1]))

        # Initialize primal variable theta using the solution of the problem without trace norm regularization
        cc, low = scipy.linalg.cho_factor(self.a_)
        self.theta_ = scipy.linalg.cho_solve((cc, low), self.b_)
        self.z_ = self._theta_to_z(self.theta_)

        # Initialize storage of the primal and dual variables across optimization
        self.thetas_ = []
        self.zs_ = []
        self.thetas_.append(self.theta_)
        self.zs_.append(self.z_)

        t = 0
        while (t < 1 or self.primal(self.theta_) - self.dual(self.z_) > self.delta_) and t < self.max_iter_:
            # Obtain closed-form solution for matrix Lambda
            nabla_n_f = np.dot(self.kernel_.W, self.theta_)
            big_lambda = scipy.linalg.sqrtm(np.dot(nabla_n_f.T, nabla_n_f) + self.epsilon * np.eye(self.n_features_in_))

            # Modify problem using Lambda
            temp = scipy.linalg.cholesky(big_lambda, lower=True)
            inv_big_lambda = scipy.linalg.cho_solve((temp, True), np.eye(big_lambda.shape[0]))
            trace_penalty_matrix = np.tensordot(self.kernel_.W,
                                                np.tensordot(inv_big_lambda, self.kernel_.W, axes=([1], [1])),
                                                axes=([1, 0], [0, 1]))
            a_t = self.a_ + self.tau * self.n_ * trace_penalty_matrix

            # Obtain new primal variable theta
            cc, low = scipy.linalg.cho_factor(a_t)
            self.theta_ = scipy.linalg.cho_solve((cc, low), self.b_)

            self.thetas_.append(self.theta_)
            self.z_ = self._theta_to_z(self.theta_)
            self.zs_.append(self.z_)
            t += 1
        if t == self.max_iter_:
            print("Stopped training after " + str(self.max_iter_) + " iterations")

        nabla_n_f = np.dot(self.kernel_.W, self.theta_)
        u, s_values, vh = np.linalg.svd(nabla_n_f, full_matrices=True)
        s_threshold = 1 / (2 * self.n_features_in_)
        self.s_hat_ = np.sum(s_values/np.sum(s_values) > s_threshold)
        if self.s_hat_ == 0:
            self.s_hat_ = self.n_features_in_
        self.p_hat_ = np.real(vh.T[:, 0:np.max([self.s_hat_, 2])]) # MARS needs at least two features

        self.is_fitted_ = True
        return self

    def primal(self, theta):
        """
        Computes the value of the primal problem given a fixed theta.

        :param theta: Primal variable.
        :return: Primal problem value for this primal variable.
        """
        nabla_n_f = np.dot(self.kernel_.W, theta)
        return (np.linalg.norm(self.y_ - np.dot(self.kernel_.U, theta)) ** 2) / self.n_ + \
               np.linalg.multi_dot(
                   [theta.T, self.nu * self.kernel_.V + self.mu * np.eye(np.shape(self.kernel_.V)[0]),
                    theta]) / self.n_ + \
               2 * self.tau * np.trace(
            np.real(scipy.linalg.sqrtm(np.dot(nabla_n_f.T, nabla_n_f)) + self.epsilon * np.eye(self.n_features_in_)))

    def dual(self, z):
        """
        Computes the value of the dual problem given a fixed z.

        :param z: Dual variable.
        :return: Dual problem value for this dual variable.
        """
        trace_vector = np.tensordot(z, self.kernel_.W, axes=([0, 1], [0, 1]))
        return (np.linalg.norm(self.y_) ** 2) / self.n_ - \
               np.linalg.multi_dot([(self.b_ + (self.n_ / 2) * trace_vector).T, np.linalg.inv(self.a_),
                                    self.b_ + (self.n_ / 2) * trace_vector]) / self.n_ - \
               np.sqrt(self.epsilon) * np.real(np.trace(
            scipy.linalg.sqrtm(4 * (self.tau ** 2) * np.eye(self.n_features_in_) - np.dot(z.T, z))))

    def _theta_to_z(self, theta):
        """
        Computes an admissible z for the dual from a theta.

        :param theta: Primal variable.
        :return: Admissible dual variable.
        """
        cc, low = scipy.linalg.cho_factor(self.c_ + self.mu * np.identity(self.n_ * (self.n_features_in_ + 1)))
        coefficients = scipy.linalg.cho_solve((cc, low), (2 / self.n_) * (np.dot(self.a_, theta) - self.b_))
        z_optimal = np.dot(self.kernel_.W, coefficients)
        return min(1, 2 * self.tau / np.linalg.norm(z_optimal, 2)) * z_optimal

    def predict(self, X):
        """
        Predict using the fitted KTNGrad model.

        :param X: Test data of shape (n_test, n_features).
        :return: Predicted values of shape (n_test,).
        """
        check_is_fitted(self)
        X = check_array(X)
        return np.dot(self.theta_, self.kernel_.pred_vector(X))

    def _more_tags(self):
        return {'poor_score': True}

    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction. The wort score can be -inf, the constant mean
        predictor score is 0 and the best possible score is 1.

        :param X: Test data of shape (n_test, n_features).
        :param y: True values for X of shape (n_test,).
        :param sample_weight: Sample weights (ignored).
        :return: R^2 score.
        """
        y_pred = self.predict(X)
        return 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    def feature_learning_score(self, p):
        """
        Compute the feature learning score. The best possible score is 1 and the worst score is 0.

        :param p: Ground truth feature matrix.
        :return: Feature learning score.
        """
        s = np.minimum(np.shape(p)[0], np.shape(p)[1])
        nabla_n_f = np.dot(self.kernel_.W, self.theta_)
        u, s_values, vh = np.linalg.svd(nabla_n_f, full_matrices=True)
        p_hat = vh.T[:, 0:s]
        pi_p_hat = np.dot(np.dot(p_hat, np.linalg.inv(np.dot(p_hat.T, p_hat))), p_hat.T)
        pi_p = np.dot(np.dot(p, np.linalg.inv(np.dot(p.T, p))), p.T)
        if s <= self.n_features_in_ / 2:
            error = (np.linalg.norm(pi_p - pi_p_hat)) ** 2 / (2 * s)
        elif s == self.n_features_in_:
            error = 0
        else:
            error = (np.linalg.norm(pi_p - pi_p_hat)) ** 2 / (2 * self.n_features_in_ - 2 * s)
        return 1 - error

    def dimension_score(self, s):
        """
        Compute the dimension learning score. The best possible score is 1 and the worst score is 0.

        :param s: Ground truth dimension.
        :return: Dimension learning score.
        """
        if s <= self.n_features_in_ / 2:
            error = np.abs(self.s_hat_ - s) / (self.n_features_in_ - s)
        else:
            error = np.abs(self.s_hat_ - s) / s
        return 1 - error
