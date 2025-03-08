#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def normalize_dictionary(X):
    """
    Normalize matrix to have unit l2-norm columns

    Parameters
    ----------
    X : np.ndarray [n, d]
        Matrix to be normalized

    Returns
    -------
    X_normalized : np.ndarray [n, d]
        Normalized matrix
    norm_coefs : np.ndarray [d]
        Normalization coefficients (i.e., l2-norm of each column of ``X``)
    """
    # Check arguments
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2

    alpha = np.linalg.norm(X, axis=0)
    return X / alpha, alpha


def ridge_regression(X, y, lambda_ridge):
    """
    Ridge regression estimation

    Minimize $\left\| X w - y\right\|_2^2 + \lambda \left\|w\right\|_2^2$
    with respect to vector $w$, for $\lambda > 0$ given a matrix $X$ and a
    vector $y$.

    Note that no constant term is added.

    Parameters
    ----------
    X : np.ndarray [n, d]
        Data matrix composed of ``n`` training examples in dimension ``d``
    y : np.ndarray [n]
        Labels of the ``n`` training examples
    lambda_ridge : float
        Non-negative penalty coefficient

    Returns
    -------
    w : np.ndarray [d]
        Estimated weight vector
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    w = np.linalg.multi_dot([
            np.linalg.inv(np.dot(X.T, X) \
                          + lambda_ridge * np.identity(n_features)),
            X.T,
            y])
    return w


def mp(X, y, n_iter):
    """
    Matching pursuit algorithm

    Parameters
    ----------
    X : np.ndarray [n, d]
        Dictionary, or data matrix composed of ``n`` training examples in
        dimension ``d``. It should be normalized to have unit l2-norm
        columns before calling the algorithm.
    y : np.ndarray [n]
        Observation vector, or labels of the ``n`` training examples
    n_iter : int
        Number of iterations

    Returns
    -------
    w : np.ndarray [d]
        Estimated sparse vector
    error_norm : np.ndarray [n_iter+1]
        Vector composed of the norm of the residual error at the beginning
        of the algorithm and at the end of each iteration
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    #added : check if the columns of X have unit l2-norm
    assert np.linalg.norm(X, axis=1).all()
    
    #Initialization of the residue r
    r = y 
    #Initialization of the parsimonious vector of decomposition w
    w = np.zeros(n_features)
    #Initialization of the vector of the successive norm of the residue 
    error_norm = np.array([np.linalg.norm(r)])
    
    for k in range(1, n_iter + 1) :
        #Compute the respective dot products between each column of X and r 
        #Stored in a vector of size d
        corr = np.matmul(X.T, r)
        #Selection of the column that shows the largest (relative) correlation
        hat_m = np.argmax(np.abs(corr))
        #Update w
        w[hat_m] += corr[hat_m]
        #Update residue
        r -= corr[hat_m] * X[:, hat_m]
        #Store the norm of the residue
        error_norm = np.append(error_norm, np.linalg.norm(r))
        
    return w, error_norm


def omp(X, y, n_iter):
    """
    Orthogonal matching pursuit algorithm

    Parameters
    ----------
    X : np.ndarray [n, d]
        Dictionary, or data matrix composed of ``n`` training examples in
        dimension ``d``. It should be normalized to have unit l2-norm
        columns before calling the algorithm.
    y : np.ndarray [n]
        Observation vector, or labels of the ``n`` training examples
    n_iter : int
        Number of iterations

    Returns
    -------
    w : np.ndarray [d]
        Estimated sparse vector
    error_norm : np.ndarray [n_iter+1]
        Vector composed of the norm of the residual error at the beginning
        of the algorithm and at the end of each iteration
    """
    # Check arguments
    assert X.ndim == 2
    n_samples, n_features = X.shape
    assert y.ndim == 1
    assert y.size == n_samples

    #added : check if the columns of X have unit l2-norm
    assert np.linalg.norm(X, axis=1).all()
    
    #Initialization of the residue r
    r = y 
    #Initialization of the parsimonious vector of decomposition w
    w = np.zeros(n_features)
    #Initialization of the decomposition support omega
    omega = np.empty(0, dtype='int')
    #Initialization of the vector of the successive norm of the residue 
    error_norm = np.array([np.linalg.norm(r)])
    
    for k in range(1, n_iter + 1) :
        #Compute the respective dot products between each column of X and r 
        #Stored in a vector of size d
        corr = np.matmul(X.T, r)
        #Selection of the column that shows the largest (relative) correlation
        hat_m = np.argmax(np.abs(corr))
        #Update support
        omega = np.union1d(omega, [int(hat_m)]) 
        #Update w
        w[omega] = np.dot(np.linalg.pinv(X[:, omega]), y)
        #Update residue
        r = y - np.dot(X, w)
        #Store the norm of the residue
        error_norm = np.append(error_norm, np.linalg.norm(r))

    return w, error_norm
    
