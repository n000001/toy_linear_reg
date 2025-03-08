# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 12:51:43 2021

@author: Nina Baldy

Script that experiments and saves the performances of learn_all methods with 
respect to the size of the training set.
"""

import numpy as np
from funcs.learn_all import (learn_all_with_ridge, 
                       learn_all_with_mp, 
                       learn_all_with_omp)
from funcs.data_utils import load_data, split_data, mse
from time import time

#Set seed to make results reproductible
np.random.seed(0)

def save_performance(file_path, N, out_file_path) :
    """
    For the J training sizes given in array N, saves performance 
    (training set sizes, training and validation accuracy, training time) 
    of the 3 (Ridge, Matching Pursuit and Orthogonal Matching Pursuit)
    methods in a .npz file.
    
    Parameters
    ----------
        file_path : str
            Path of the file containing the dataset
        N : np.ndarray[J]
            Values of training set sizes to try
        out_file_path : str
            Path of the file for saving the results
    """
    #Loading the data from the .npz file 
    X_labeled, y_labeled, X_unlabeled = load_data(file_path)

    #Random split of the labelled data into two subsets for train and validation
    #S0_train = (X0_train, y0_train) ; S_valid = (X_valid, y_valid)
    #1/3 of the labeled data is in the validation 2 set S_valid
    X0_train, y0_train, X_valid, y_valid = split_data(X_labeled, y_labeled, 2/3)
    print("Size of the validation 2 set = ", X_valid.shape[0])

    #The methods we will use : learn_all methods for ridge, mp and omp
    methods = [learn_all_with_ridge, 
               learn_all_with_mp, 
               learn_all_with_omp]
    
    #Initialize arrays of errors and learning execution time
    train_error = np.empty((len(N), len(methods)))
    valid_error = np.empty((len(N), len(methods)))
    learning_time = np.empty((len(N), len(methods)))
    
    for  j, reg_method in enumerate(methods) :
        for i, n in enumerate(N) :
            #S_train : first n examples of S0_train
            X_train, y_train = X0_train[:n], y0_train[:n]
            
            #learn_all method using the j-th method in the list method
            #and the current train set 
            #And computation of the train and validation MSE
            start = time() #start the timer
            reg = reg_method(X_train, y_train) #hyper-param. estimation + fit
            end = time() #stop the timer
            #We store execution time 
            learning_time[i, j] = end - start
            #Prediction and evaluation of the errors 
            y_pred_train = reg.predict(X_train)
            y_pred_valid = reg.predict(X_valid)
            train_error[i][j] = mse(y_train, y_pred_train)
            valid_error[i][j] = mse(y_valid, y_pred_valid)
    
    #Save the resulting arrays in a .npz file in current directory
    np.savez(out_file_path, 
             N=N, 
             valid_error=valid_error, 
             train_error=train_error,
             learning_time=learning_time)


if __name__ == '__main__':
    #Define set N of training set sizes to try
    N = np.array([2**j for j in range(5, 12)])
    #Saving perfromance results into a .npz file
    save_performance("../data/YearPredictionMSD_100.npz", N, 
                     "exp_training_size_learn_all.npz")
    
    #----Optional: comparison with the other data sets provided on Ametice:
    #YearPredictionMSD_2_100.npz and YearPredictionMSD_2_12_100.npz 
    #which contain data from the same songs but with a larger number of 
    #descriptive variables
    save_performance("../data/YearPredictionMSD_2_100.npz", N, 
                     "exp_training_size_learn_all_2.npz")
    save_performance("../data/YearPredictionMSD_2_12_100.npz", N, 
                     "exp_training_size_learn_all_2_12.npz")
