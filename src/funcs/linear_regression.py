#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from funcs.algorithms import mp, omp, ridge_regression, normalize_dictionary


class LinearRegression:
    """
    Generic class for linear regression

    Two attributes ``w`` and ``b`` are used for a linear regression of the form
    f(x) = w.x + b

    All linear regression method should inherit from :class:`LinearRegression`,
    and implement the ``fit`` method to learn ``w`` and ``b``. Method
    ``predict`` implemented in :class:`LinearRegression` will be inherited
    without needed to be reimplemented.
    """
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        This method should be used to learn w and b in any subclass. Here,
        it is just checking the parameters' properties (call this parent's
        method in any subclass).

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        assert X.ndim == 2
        n_samples, n_features = X.shape
        assert y.ndim == 1
        assert y.size == n_samples

        self.w = None
        self.b = None

    def predict(self, X):
        """
        Predict labels from feature vectors via a linear function.

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n feature vectors with size d

        Returns
        -------
        y : np.ndarray [n]
            Vector of n predicted labels for X
        """
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
        assert X.shape[1] == self.w.shape[0]

        return X @ self.w + self.b


class LinearRegressionLeastSquares(LinearRegression):
    """
    Linear regression using a least-squares method
    """
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, X, y):
        """
        Learn parameters using a least-square method

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        tilde_X = np.concatenate((X, np.ones(X.shape[0]).reshape(-1, 1)),
                                 axis=1)
        tilde_w = np.dot(np.linalg.pinv(tilde_X), y)
        
        self.w = tilde_w[:-1]
        self.b = tilde_w[-1]


class LinearRegressionMean(LinearRegression):
    """
    Constant-valued regression function equal to the mean of the training
    labels
    """
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, X, y):
        """
        Learn a constant function equal to the mean of the training labels

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        self.w = np.zeros(X.shape[1])
        self.b = np.mean(y)


class LinearRegressionMedian(LinearRegression):
    """
    Constant-valued regression function equal to the median of the training
    labels
    """
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, X, y):
        """
        Learn a constant function equal to the median of the training labels

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        self.w = np.zeros(X.shape[1])
        self.b = np.median(y)


class LinearRegressionMajority(LinearRegression):
    """
    Constant-valued regression function equal to the majority training label
    """
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, X, y):
        """
        Learn a constant function equal to the majority training label

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        h, bins = np.histogram(y, bins=np.arange(np.min(y), np.max(y) + 2))

        self.w = np.zeros(X.shape[1])
        self.b = bins[np.argmax(h)]


class LinearRegressionRidge(LinearRegression):
    """
    Linear regression based on ridge regression

    Parameters
    ----------
    lambda_ridge : float
        Non-negative penalty coefficient
    """
    def __init__(self, lambda_ridge):
        LinearRegression.__init__(self)
        self.lambda_ridge = lambda_ridge

    def fit(self, X, y):
        """
        Learn linear function using ridge regression

        The constant term ``b`` is first estimated and subtracted from
        labels ``y``; vector ``w`` is then estimated using a feature matrix
        normalized with l2-norm columns

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        #Esimation of b
        self.b = np.mean(y)
        #Shifting y so it has 0 mean 
        tilde_y = y - self.b
        #Normalization of X : normalized X and normalization coefficients
        tilde_X, alpha = normalize_dictionary(X)
        #Apply ridge regression algorithm 
        tilde_w = ridge_regression(tilde_X, tilde_y, self.lambda_ridge)
        #Rescaling of w
        w = tilde_w / alpha
        self.w = w
        


class LinearRegressionMp(LinearRegression):
    """
    Linear regression based on Matching Pursuit

    Parameters
    ----------
    n_iter : int
        Number of iterations
    """
    def __init__(self, n_iter):
        LinearRegression.__init__(self)
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Learn linear function using Matching Pursuit (MP)

        The constant term ``b`` is first estimated and subtracted from
        labels ``y``; vector ``w`` is then estimated using MP,
        after normalizing the dictionary ``X``

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d (may not be
            normalized)
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        #Estimation of b
        self.b = np.mean(y)
        #Shifting y so it has 0 mean 
        tilde_y = y - self.b
        #Normalization of X : normalized X and normalization coefficients
        tilde_X, alpha = normalize_dictionary(X)
        #Apply MP algorithm
        tilde_w, self.error_norm = mp(tilde_X, tilde_y, self.n_iter)
        #Rescaling of w
        w = tilde_w / alpha
        self.w = w
        
        
class LinearRegressionOmp(LinearRegression):
    """
    Linear regression based on Matching Pursuit

    Parameters
    ----------
    n_iter : int
        Number of iterations
    """
    def __init__(self, n_iter):
        LinearRegression.__init__(self)
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Learn linear function using Orthogonal Matching Pursuit (MP)

        The constant term ``b`` is first estimated and subtracted from
        labels ``y``; vector ``w`` is then estimated using MP,
        after normalizing the dictionary ``X``

        Parameters
        ----------
        X : np.ndarray [n, d]
            Array of n training feature vectors with size d (may not be
            normalized)
        y : np.ndarray [n]
            Vector of n training labels related to X
        """
        LinearRegression.fit(self, X, y)

        #Estimation of b
        self.b = np.mean(y)
        #Shifting y so it has 0 mean 
        tilde_y = y - self.b
        #Normalization of X : normalized X and normalization coefficients
        tilde_X, alpha = normalize_dictionary(X)
        #Apply OMP algorithm
        tilde_w, self.error_norm = omp(tilde_X, tilde_y, self.n_iter)
        #Rescaling of w
        w = tilde_w / alpha
        self.w = w
