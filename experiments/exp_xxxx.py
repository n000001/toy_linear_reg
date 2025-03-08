# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 17:02:15 2021

@author: Nina Baldy

Experimenting the choice of the hyper-parameter.
"""

import numpy as np
from funcs.data_utils import load_data, split_data
from funcs.linear_regression import (LinearRegressionRidge, 
                               LinearRegressionMp,
                               LinearRegressionOmp)
import matplotlib.pyplot as plt

#Loading the data from .npz file of data repository ./data/
X_labeled, y_labeled, X_unlabeled = load_data(
        "../data/YearPredictionMSD_100.npz")
#Splitting the labeled data : 500 training examples
ratio = 500 / len(y_labeled)
X_train, y_train, X_valid, y_valid = split_data(X_labeled, y_labeled, ratio)
print(X_train.shape)

kmax = 1000
mp_reg = LinearRegressionMp(kmax)
mp_reg.fit(X_train, y_train)
omp_reg = LinearRegressionOmp(kmax)
omp_reg.fit(X_train, y_train)
fig, ax = plt.subplots()
ax.plot(range(kmax + 1), mp_reg.error_norm, label='MP')
ax.scatter([0, kmax], [mp_reg.error_norm[0], mp_reg.error_norm[-1]])
ax.plot(range(kmax + 1), omp_reg.error_norm, label='OMP')
ax.scatter([0, kmax], [omp_reg.error_norm[0], omp_reg.error_norm[-1]])
ax.legend()

plt.close("all")
