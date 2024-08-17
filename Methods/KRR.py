from scipy.spatial.distance import cdist
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class KRR(RegressorMixin, BaseEstimator):
    """
    KRR Regressor: A basic kernel ridge regression method with the Gaussian kernel
    """

    def __init__(self, lambda_val=0.00125):
        """
        Initialize the KRR regressor.

        :param lambda_val: Regularization parameter. Must be non-negative.
        """
        if lambda_val < 0:
            raise ValueError("lambda_val must be non-negative.")
        self.lambda_val = lambda_val

    def fit(self, X, y):
        """
        Fit the KRR model according to the given training data.

        :param X: Training data of shape (n_samples, n_features).
        :param y: Target values of shape (n_samples,).
        :return: self : Returns an instance of self.
        """
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')
        X, y = check_X_y(X, y, y_numeric=True)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = self.X_.shape[1]
        self.n_ = self.X_.shape[0]
        if self.n_ == 1:
            raise ValueError("1 sample")

        # Compute the kernel matrix
        self.sigma_ = np.median(cdist(self.X_, self.X_, 'euclidean'))
        self.K_ = self.K_matrix(self.X_)

        # Solve for alpha
        self.alpha_ = np.linalg.solve(self.K_ + self.n_ * self.lambda_val * np.identity(self.n_), self.y_)
        self.is_fitted_ = True
        return self

    def K_matrix(self, other_X):
        """
        Compute the kernel matrix between the training data and other data.

        :param other_X: Other data of shape (n_test, self.n_features_in).
        :return: Kernel matrix of shape (n_test, n_samples).
        """
        n_test, _ = other_X.shape
        pairwise_sq_dists = cdist(other_X, self.X_, 'sqeuclidean')
        K = np.exp(-pairwise_sq_dists / (2 * self.sigma_ ** 2))
        return K

    def predict(self, X):
        """
        Predict using the KRR model.

        :param X: Test data of shape (n_test, n_features).
        :return: Predicted values of shape (n_test,).
        """
        check_is_fitted(self)
        X = check_array(X)
        K = self.K_matrix(X)
        y_pred = K @ self.alpha_
        return y_pred

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
