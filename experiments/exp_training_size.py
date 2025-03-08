# -*- coding: utf-8 -*-
"""
@author: Nina 

Script that experiments and saves the performances of simple regression methods 
with respect to the size of the training set.
"""


import numpy as np
from funcs.data_utils import load_data, split_data, mse
from funcs.linear_regression import (LinearRegressionMean, 
                               LinearRegressionMedian,
                               LinearRegressionMajority, 
                               LinearRegressionLeastSquares)
from time import time

#Initialize the seed : for reproductive splitting results
np.random.seed(0)

#Loading the data from .npz file of data repository ./data/
X_labeled, y_labeled, X_unlabeled = load_data(
        "../data/YearPredictionMSD_100.npz")

#Random split of the labelled data into two subsets for train and validation
#S0_train = X0_train, y0_train ; S_valid = X_valid, y_valid
X0_train, y0_train, X_valid, y_valid = split_data(X_labeled, y_labeled, 2/3)

#Define set N
N = np.array([2**j for j in range(5, 12)])

#Define the linear regression methods we will use
methods = [LinearRegressionMean(), 
           LinearRegressionMedian(), 
           LinearRegressionMajority(),
           LinearRegressionLeastSquares()]

#Initialize arrays of errors and learning execution time
train_error = np.empty((len(N), len(methods)))
valid_error = np.empty((len(N), len(methods)))
learning_time = np.empty((len(N), len(methods)))

for  j, reg_method in enumerate(methods) :
    for i, n in enumerate(N) :
        #S_train : first n examples of S0_train (X0_train, y0_train)
        X_train, y_train = X0_train[:n], y0_train[:n]
        
        #Regression using the j-th method in the list method
        #and the current train set
        linear_reg = reg_method
        
        #Estimate parameters
        start = time() #starting the timer
        linear_reg.fit(X_train, y_train) #regression fit
        end = time()  #stopping the timer
        #Store execution time 
        learning_time[i, j] = end - start
        
        #Prediction mean squared error on validation set
        #The cardinal of the validaion set remains unchanged
        #valid_error[i, j] is the mse of the prediction on the i-th validation
        #set, using the j-th regression method: 
        #Mean, Median, Majority or Least squares [0, 1, 2, 3].
        y_valid_pred = linear_reg.predict(X_valid)
        valid_error[i, j] = mse(y_valid,y_valid_pred)
        
        #Prediction mean squared error on training set
        #train_error[i, j] is the mse of the prediction on the i-th train set, 
        #set, using the j-th regression method: 
        #Mean, Median, Majority or Least squares [0, 1, 2, 3].
        y_train_pred = linear_reg.predict(X_train)
        train_error[i, j] = mse(y_train, y_train_pred)
        
#Save the resulting arrays in a .npz file in current directory
np.savez("exp_training_size.npz", 
         N=N, 
         valid_error=valid_error, 
         train_error=train_error, 
         learning_time=learning_time)
    
    
    
    
    
