# -*- coding: utf-8 -*-
"""
Script that performs the prediction (labels=years) of the test data using ridge 
regression with cross-validation for the choice of the hyper-parameter
(regularization strenght) and a training set of size 500.
"""

import numpy as np
from funcs.prediction import learn_best_predictor_and_predict_test_data

#Setting seed to make results reproductible
np.random.seed(42)

#Call to learn_best_predictor_and_predict_test_data that will save predictions
#for the unlabelled data from YearPredictionMSD_100.npz into the output file
#Prediction uses the default parameters : method="ridge" and n_train=500
test_data = learn_best_predictor_and_predict_test_data(
        file="../data/YearPredictionMSD_100.npz",
        out_file="test_prediction_results.npy")

