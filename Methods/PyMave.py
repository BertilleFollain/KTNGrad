from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from rpy2.robjects.packages import importr
from rpy2.robjects.pandas2ri import activate
import numpy as np
import pyearth
import warnings

importr("RcppArmadillo")
Mave = importr("MAVE")
activate()


class PyMave(RegressorMixin, BaseEstimator):
    """
    PyMave Regressor: A Python wrapper for the MAVE dimension reduction method from R, combined with MARS for regression.

    This estimator uses the MAVE algorithm from the R package MAVE to perform dimension reduction, followed by fitting a
    Multivariate Adaptive Regression Splines (MARS) model using the reduced
    dimensions. The method is fully integrated into the Scikit-learn framework.

    MAVE Reference:
    Xia, Y., Tong, H., Li, W.K., and Zhu, L.-X. (2002). An adaptive estimation of dimension reduction space.
    Journal of the Royal Statistical Society: Series B (Statistical Methodology), 64: 363-410.
    https://doi.org/10.1111/1467-9868.03411

    MARS Reference:
    Friedman, J. H. (1991). Multivariate Adaptive Regression Splines. The Annals of Statistics, 19(1), 1-67.

    """

    def __init__(self):
        """
        Initialize the PyMave regressor.
        """
        pass

    def fit(self, X, y):
        """
        Fit the PyMave model according to the given training data.

        :param X: Training data of shape (n_samples, n_features).
        :param y: Target values of shape (n_samples,).
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
        if self.n_features_in_ == 1:
            raise ValueError("Number of features must be greater than 1. Got 1 feature.")

        # Perform dimension reduction using MAVE
        self.dr_mave_ = Mave.mave_compute(self.X_, self.y_)
        self.dim_ = np.argmin(np.array(Mave.mave_dim(dr=self.dr_mave_)[-3])) + 1
        self.p_hat_ = np.array(Mave.coef_mave(self.dr_mave_, dim=self.dim_))

        # Fit MARS model on the reduced dimensions
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.mars_ = pyearth.Earth(max_degree=self.dim_)
            self.mars_.fit(np.dot(self.X_, self.p_hat_), self.y_)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict using the PyMave model.

        :param X: Test data of shape (n_samples, n_features).
        :return: Predicted values of shape (n_samples,).
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = self.mars_.predict(np.dot(X, self.p_hat_))
        return y_pred

    def _more_tags(self):
        return {'poor_score': True}

    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction.

        :param X: Test data of shape (n_samples, n_features).
        :param y: True values for X of shape (n_samples,).
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
        p_hat = np.array(Mave.coef_mave(self.dr_mave_, dim=s))
        pi_p_hat = np.dot(np.dot(p_hat, np.linalg.inv(np.dot(p_hat.T, p_hat))), p_hat.T)
        pi_p = np.dot(np.dot(p, np.linalg.inv(np.dot(p.T, p))), p.T)
        if s <= self.n_features_in_ / 2:
            error = (np.linalg.norm(pi_p - pi_p_hat)) ** 2 / (2 * s)
        else:
            error = (np.linalg.norm(pi_p - pi_p_hat)) ** 2 / (2 * (self.n_features_in_ - s))
        return 1 - error

    def dimension_score(self, s):
        """
        Compute the dimension learning score. The best possible score is 1 and the worst score is 0.

        :param s: Ground truth dimension.
        :return: Dimension learning score.
        """
        if s <= self.n_features_in_ / 2:
            error = np.abs(self.dim_ - s) / (self.n_features_in_ - s)
        else:
            error = np.abs(self.dim_ - s) / s
        return 1 - error
