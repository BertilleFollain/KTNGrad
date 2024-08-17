import numpy as np
from scipy.spatial.distance import cdist


class GaussianKernel:
    def __init__(self, X, sigma):
        """
        Initialize the GaussianKernel with the provided data and parameters.
        Computes the matrices K,Z,L (small matrices) and then U,V,W (big matrices) for the provided input data.

        :param X: Input data matrix of shape (n_samples, n_features).
        :param sigma: Bandwidth parameter for the Gaussian kernel.
        """
        self.X = X
        self.sigma = sigma
        self.n = X.shape[0]
        self.d = X.shape[1]

        # Compute the small matrices and big matrices on initialization
        self._compute_little_matrices()
        self._compute_big_matrices()

    def _compute_little_matrices(self):
        """
        Compute the k, z, l matrices specific to the Gaussian kernel.
        """
        pairwise_sq_dists = cdist(self.X, self.X, 'sqeuclidean')
        self.K = np.exp(-pairwise_sq_dists / (2 * self.sigma ** 2)) / self.n

        Z = np.tile(-self.K / (self.sigma ** 2), (1, self.d))
        j_factor = np.repeat(self.X, self.n, axis=1)
        i_factor = np.tile(self.X.T.flatten(), (self.n, 1))
        self.Z = (i_factor - j_factor) * Z

        L = -np.tile(np.tile(self.K, (1, self.d)), (self.d, 1)) / (self.sigma ** 4)
        b_factor = np.tile(j_factor - i_factor, (self.d, 1))
        a_equal_b_factor = np.kron(np.eye(self.d, dtype=int), np.ones((self.n, self.n))) * self.sigma ** 2
        a_factor = np.tile((i_factor - j_factor).T, (1, self.d))
        self.L = L * (a_factor * b_factor - a_equal_b_factor)

    def _compute_big_matrices(self):
        """
        Computes the larger matrices U, V, W based on the smaller matrices K, Z, L.
        """
        self.U = np.hstack((self.K, self.Z))
        self.V = np.vstack((self.U, np.hstack((self.Z.T, self.L))))
        self.W = np.zeros((self.n, self.d, self.n * (self.d + 1)))

        # Efficiently populate the w matrix
        for a in range(self.d):
            Z_slice = self.Z[:, a * self.n:(a + 1) * self.n]
            L_slice = self.L[a * self.n:(a + 1) * self.n, :]
            self.W[:, a, :] = np.hstack((Z_slice.T, L_slice))

    def evaluate(self, x1, x2):
        """
        Evaluate the Gaussian kernel between two points.

        :param x1: First input vector.
        :param x2: Second input vector.
        :return: The Gaussian kernel value between z1 and z2.
        """
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * self.sigma ** 2))

    def pred_vector(self, other_X):
        """
        CCompute the prediction vector for a given input matrix.

        :param other_X: Input data of shape (n_test, n_features) representing the input data.
        :return: Vector of features for the input data.
        """
        pairwise_sq_dists = cdist(other_X, self.X, 'sqeuclidean')
        K_test = np.exp(-pairwise_sq_dists / (2 * self.sigma ** 2)) / self.n
        Z_test = np.zeros((other_X.shape[0], self.n * self.d))
        # Compute Z matrix
        for a in range(self.d):
            pairwise_diff_a = other_X[:, a][:, np.newaxis] - self.X[:, a][np.newaxis, :]
            Z_test_a = (pairwise_diff_a / self.sigma ** 2) * K_test
            Z_test[:, a * self.n:(a + 1) * self.n] = Z_test_a
        return np.hstack((K_test, Z_test)).T
