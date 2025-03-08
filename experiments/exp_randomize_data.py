#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:03:05 2017

@author: Valentin Emiya, AMU & CNRS LIF
"""
import numpy as np
from funcs.data_utils import randomize_data

""" Small programme to run randomize_data """
n_examples = 5
data_dim = 4

X_test = np.arange(n_examples*data_dim).reshape(-1, data_dim)
y_test = np.array([0, -4, -8, -12, -16])
print(X_test)
print(y_test)
rX_test, ry_test = randomize_data(X_test, y_test)
print(rX_test, '\n', ry_test)