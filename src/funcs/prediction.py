# -*- coding: utf-8 -*-

"""
@author: Nina 

Function that learns the best predictor using a learn_all method 
(cross-validation for the choice of the hyper-parameter and estimation of the
parameters using this value for the hyper-parameter) and uses it to predict
the labels of the test data.
"""

import numpy as np
from funcs.data_utils import load_data, split_data
from funcs.learn_all import (learn_all_with_ridge, 
                       learn_all_with_mp, 
                       learn_all_with_omp)


def learn_best_predictor_and_predict_test_data(file, out_file, method='ridge', 
                                               n_train=500) :
    """
    Save the predictions of unlabeled data obtained with the choosen regression
    method into a .npz file
    
    Parameters
    ----------
    file : str
        path of the file to load the data from
    out_file : str
        path of the file to save the predictions in
    method : str
        method (ridge, mp or omp) to use for the regression
    n_train : int
        cardinal of the training set
    """
    #Defining the function to use according to the method that was given
    #and raising an error if the method is not defined
    if method == "ridge" :
        reg_func = learn_all_with_ridge
    elif method == "mp" :
        reg_func = learn_all_with_mp
    elif method == "omp" :
        reg_func = learn_all_with_omp
    else :
        raise ValueError("Invalid method: must be 'ridge', 'mp' or 'omp'")
        
    #Load the labeled data and unlabeled data from the file
    X_labeled, y_labeled, X_unlabeled = load_data(file)
    
    #Raising an error if the number of training examples is inconsistent with
    #the data
    if n_train >= len(y_labeled) or n_train < 0 :
        raise ValueError("Not enough training examples")
    
    #Randomly split the labeled data into 2 subsets: 
    #- "test" subset (X_test, y_test) of size 500
    #- "validation 2" subset (X_valid2, y_valid2) with the remaining data
    ratio = 500 / len(y_labeled)
    X_train, y_train, X_valid2, y_valid2 = split_data(X_labeled, y_labeled, 
                                                      ratio)
    
    reg = reg_func(X_train, y_train)
    y_valid_pred = reg.predict(X_valid2)
    valid_error = np.mean((y_valid2 - y_valid_pred)**2)
    print("MSE on validation set (size ", len(y_valid2), ") =", valid_error)
    
    y_test = reg.predict(X_unlabeled)
    np.save(out_file, y_test)
    
    return y_test
