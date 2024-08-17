from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from rpy2.robjects.packages import importr
from rpy2.robjects.pandas2ri import activate
import pyearth
import warnings

importr("RcppArmadillo")
activate()


class MARS(RegressorMixin, BaseEstimator):
    """
    MARS Regressor: A Python wrapper for the MARS regression method from R.

    Reference:
    Friedman, J. H. (1991). Multivariate Adaptive Regression Splines. The Annals of Statistics, 19(1), 1-67.

    """

    def __init__(self):
        """
        Initialize the MARS regressor.
        """
        pass

    def fit(self, X, y):
        """
        Fit the MARS model according to the given training data.

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

        # Fit MARS model on the reduced dimensions
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.mars_ = pyearth.Earth(max_degree=self.n_features_in_)
            self.mars_.fit(self.X_, self.y_)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict using the MARS model.

        :param X: Test data of shape (n_samples, n_features).
        :return: Predicted values of shape (n_samples,).
        """
        check_is_fitted(self)
        X = check_array(X)
        y_pred = self.mars_.predict(X)
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
