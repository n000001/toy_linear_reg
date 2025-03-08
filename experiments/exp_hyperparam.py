# -*- coding: utf-8 -*-
"""
@author: Nina 

Script that illustrates the choice of the hyper-parameter for minimizing 
the validation error.
"""

import numpy as np
from funcs.data_utils import load_data, split_data, mse
from funcs.linear_regression import (LinearRegressionRidge, 
                               LinearRegressionMp,
                               LinearRegressionOmp)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
sns.set_context("paper")

#Set seed to make resulsts reproductible 
np.random.seed(0)

#Loading the data from .npz file of data repository ./data/
X_labeled, y_labeled, X_unlabeled = load_data(
        "../data/YearPredictionMSD_100.npz")
#Splitting the labeled data : 500 training examples
ratio = 500 / len(y_labeled)
X_train, y_train, X_valid, y_valid = split_data(X_labeled, y_labeled, ratio)
print(X_train.shape)

#Ridge regression
#100  evenly spaced values for the parameter lambda_ridge: [10^{-4}, ... , 10]
lambda_ridge_arr = np.linspace(10E-04, 10, 100)
ridge_errors = [[], []]
for lambda_ridge in lambda_ridge_arr :
    ridge_reg = LinearRegressionRidge(lambda_ridge)
    ridge_reg.fit(X_train, y_train)
    y_pred_train = ridge_reg.predict(X_train)
    y_pred_valid = ridge_reg.predict(X_valid)
    ridge_errors[0].append(mse(y_train, y_pred_train))
    ridge_errors[1].append(mse(y_valid, y_pred_valid))
ridge_errors = np.array(ridge_errors).T
lambda_opti_valid = lambda_ridge_arr[np.argmin(ridge_errors[:, 1])]
print("Optimal lambda ridge value for validation =", lambda_opti_valid)

#Matching pursuit
#100 values for the parameter kmax: [1, 10, ..., 800]
kmax_values = range(1, 800, 10)
mp_errors = [[], []]
for kmax in kmax_values :
    mp_reg = LinearRegressionMp(kmax)
    mp_reg.fit(X_train, y_train)
    y_pred_train = mp_reg.predict(X_train)
    y_pred_valid = mp_reg.predict(X_valid)
    mp_errors[0].append(mse(y_train, y_pred_train))
    mp_errors[1].append(mse(y_valid, y_pred_valid))
mp_errors = np.array(mp_errors).T
kmaxmp_opti_valid = kmax_values[np.argmin(mp_errors[:, 1])]
print("optimal kmax value for validation (MP) =", kmaxmp_opti_valid)

#Orthogonal matching pursuit
#100 values for the parameter kmax: [1, 10, ..., 800]
#Please not that the computation might take some time
omp_errors = [[], []]
for kmax in kmax_values :
    mp_reg = LinearRegressionOmp(kmax)
    mp_reg.fit(X_train, y_train)
    y_pred_train = mp_reg.predict(X_train)
    y_pred_valid = mp_reg.predict(X_valid)
    omp_errors[0].append(mse(y_train, y_pred_train))
    omp_errors[1].append(mse(y_valid, y_pred_valid))
omp_errors = np.array(omp_errors).T  
kmaxomp_opti_valid = kmax_values[np.argmin(omp_errors[:, 1])]
print("optimal kmax value for validation (OMP) =", kmaxomp_opti_valid)


#Plots
#Error plot for ridge regression
fig, ax = plt.subplots()
ax.plot(lambda_ridge_arr, ridge_errors[:, 1], '-r', label='Validation')
ax.plot(lambda_ridge_arr, ridge_errors[:, 0], '--r', label='Train')
ax.scatter(x=lambda_opti_valid, y=np.min(ridge_errors[:, 1]), color='grey',
           marker ='x')
ax.vlines(x=lambda_opti_valid, ymin=0, ymax=np.min(ridge_errors[:, 1]), 
          color='grey', linestyle=':')
ax.invert_xaxis()
ax.set_xlabel(r'$\lambda_{ridge}$')
ax.set_ylabel('Mean squared error')
ax.legend(frameon=True)
ax.set_ylim([150, 400])
ax.set_title(r'MSE according to the value of hyperparameter $\lambda_{ridge}$')
plt.semilogx()
plt.tight_layout()
plt.savefig(r'../figures/errors_lambda_ridge.png')
#Error plot for MP and OMP algorithms
fig, ax = plt.subplots()
ax.plot(kmax_values, mp_errors[:, 1], '-', color='seagreen', 
        label='MP - Validation')
ax.plot(kmax_values, mp_errors[:, 0], '--', color='seagreen', 
        label='MP - Train')
ax.plot(kmax_values, omp_errors[:, 1], '-', color='slateblue', 
        label='OMP - Validation')
ax.plot(kmax_values, omp_errors[:, 0], '--', color='slateblue',
        label='OMP - Train')
ax.scatter(x=kmaxmp_opti_valid, y=np.min(mp_errors[:, 1]), color='grey',
            marker='x')
ax.vlines(x=kmaxmp_opti_valid, ymin=0, ymax=np.min(mp_errors[:, 1]), 
          color='grey', linestyle=':')
ax.scatter(x=kmaxomp_opti_valid, y=np.min(omp_errors[:, 1]), color='grey',
            marker='x')
ax.vlines(x=kmaxomp_opti_valid, ymin=0, ymax=np.min(omp_errors[:, 1]), 
          color='grey', linestyle=':')
ax.set_xlabel(r'$k_{max}$')
ax.set_ylabel('Mean squared error')
ax.legend(frameon=True)
ax.set_ylim([150, 400])
ax.set_title(r'MSE according to the value of hyperparameter $k_{max}$')
plt.tight_layout()
plt.savefig(r'../figures/errors_kmax_MP_OMP.png')

plt.close("all")
