# -*- coding: utf-8 -*-

"""
@author: Nina 

Definition of the "learn_all_with_ALGO" functions used 
for training the model without specifiying the optimal parameter for the algorithm
"""

import numpy as np
from funcs.data_utils import split_data, mse
from funcs.linear_regression import (LinearRegressionRidge, 
                               LinearRegressionMp,
                               LinearRegressionOmp)



def learn_all_with_ridge(X, y) :
    """
    Ridge regression using optimal value for the hyperparameter on the training 
    set (X, y)
    
    Parameters
    ----------
    X : np.ndarray [n, d]
        Array of n training feature vectors with size d
    y : np.ndarray [n]
        Vector of n training labels related to X
    
    Returns
    -------
    reg : LinearRegressionRidge class object
          The ridge regression model trained on training set (X, y) using 
          optimal value of hyperparameter lambda
    """
    #Split X and y training data into subsets of train and validation sets 
    #with respective proportions 2/3 and 1/3
    X_train, y_train, X_valid, y_valid = split_data(X, y, 2/3)
    
    #Define a set of 20 lambda hyperparameter values to test
    hparam_values = np.linspace(10E-02, 2, 20)
    
    valid_errors = []
    #For every hyperparameter value in the set of 20 possible values
    #we compute (MSE) validation error of the regression achieved 
    #when using this value for the hyperparameter
    for hparam in hparam_values :
        reg = LinearRegressionRidge(hparam)
        reg.fit(X_train, y_train)
        y_pred_valid = reg.predict(X_valid)
        valid_errors.append(mse(y_valid, y_pred_valid))
    valid_errors = np.array(valid_errors)
    #Get the hyperparameter value that minimizes the validation error
    hparam_opti = hparam_values[np.argmin(valid_errors)]
    
    #Re-do the regression using the optimal value of hyperparameter and the 
    #whole labeled train set (X, y)
    reg = LinearRegressionRidge(hparam_opti)
    reg.fit(X, y)
    
    return reg 


def learn_all_with_mp(X, y) :
    """
    Regression based on Matching Pursuit using optimal value for the 
    hyperparameter on the training set (X, y)
    
    Parameters
    ----------
    X : np.ndarray [n, d]
        Array of n training feature vectors with size d
    y : np.ndarray [n]
        Vector of n training labels related to X
    
    Returns
    -------
    reg : LinearRegressionMp class object
          The MP regression model trained on training set (X, y) using 
          optimal value of the hyperparameter kmax
    """
    #Split X and y training data into subsets of train and validation sets 
    #with respective proportions 2/3 and 1/3
    X_train, y_train, X_valid, y_valid = split_data(X, y, 2/3)
    
    #Define a set of 20 hyperparameter values to test
    hparam_values = np.linspace(0, 150, 20, dtype='int')
    
    valid_errors = []
    #For every hyperparameter value in the set of 20 possible values
    #we compute (MSE) validation error of the regression achieved 
    #when using this value for the hyperparameter
    for hparam in hparam_values :
        reg = LinearRegressionMp(hparam)
        reg.fit(X_train, y_train)
        y_pred_valid = reg.predict(X_valid)
        valid_errors.append(mse(y_valid, y_pred_valid))
    valid_errors = np.array(valid_errors)
    #Get the hyperparameter value that minimizes the validation error
    hparam_opti = hparam_values[np.argmin(valid_errors)]
    
    #Re-do the regression using the optimal value of hyperparameter and the 
    #whole labeled train set (X, y)
    reg = LinearRegressionMp(hparam_opti)
    reg.fit(X, y)
    
    return reg 


def learn_all_with_omp(X, y) :
    """
    Regression based on Orthgonal Matching Pursuit using optimal value for the 
    hyperparameter on the training set (X, y)
    
    Parameters
    ----------
    X : np.ndarray [n, d]
        Array of n training feature vectors with size d
    y : np.ndarray [n]
        Vector of n training labels related to X
    
    Returns
    -------
    reg : LinearRegressionMp class object
          The OMP regression model trained on training set (X, y) using 
          optimal value of the hyperparameter kmax
    """
    #Split X and y training data into subsets of train and validation sets 
    #with respective proportions 2/3 and 1/3
    X_train, y_train, X_valid, y_valid = split_data(X, y, 2/3)
    
    #Define a set of 20 hyperparameter values to test
    hparam_values = np.linspace(0, 150, 20, dtype='int')
    
    valid_errors = []
    #For every hyperparameter value in the set of 20 possible values
    #we compute (MSE) validation error of the regression achieved when using 
    #this value for the hyperparameter
    for hparam in hparam_values :
        reg = LinearRegressionOmp(hparam)
        reg.fit(X_train, y_train)
        y_pred_valid = reg.predict(X_valid)
        valid_errors.append(mse(y_valid, y_pred_valid))
    valid_errors = np.array(valid_errors)
    #Get the hyperparameter value that minimizes the validation error
    hparam_opti = hparam_values[np.argmin(valid_errors)]
    
    #Re-do the regression using the optimal value of hyperparameter and the 
    #whole labeled train set (X, y)
    reg = LinearRegressionOmp(hparam_opti)
    reg.fit(X, y)
    
    return reg 

