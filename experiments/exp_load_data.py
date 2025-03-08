# -*- coding: utf-8 -*-
"""
Loads training data and print dimensions as well as a few coefficients
in the first and last places and at random locations.
"""

from funcs.data_utils import load_data

#Load data
X_labeled, y_labeled, X_unlabeled = load_data(
        "../data/YearPredictionMSD_100.npz")

#Print properties of the loaded data frames and some of the values
for array, name_array in zip([X_labeled, y_labeled, X_unlabeled],
                             ["X_labeled", "y_labeled", "X_unlabeled"]) :
    print(name_array + ": type=", type(array), 
          ", ndim=", array.ndim, 
          ", shape=", array.shape)

print('First 5 values of y_labeled', y_labeled[:5], 
      '\n Last 5 values of y_labeled', y_labeled[-5:])
 
for array, name_array in zip([X_labeled, X_unlabeled],
                             ["X_labeled", "X_unlabeled"]) :
    print('First 2 coefficients of the first and last lines of ' + name_array 
          + ':', array[0, :2], '\n', array[-1, :2])
    print('Last coefficient of the first and last lines of ' + name_array
          + ':', array[0, -1], '\n', array[-1, -1])
    
