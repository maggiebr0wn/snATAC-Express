#!/usr/sbin/anaconda

import argparse
import h5py
from sklearn.model_selection import GridSearchCV
import multiprocessing
import numpy as np
import os
import pandas as pd
from scipy import sparse, io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
import sys

### 07-10-23 ###
# This script assesses parameters for RF regression
# using the optimal set of peaks previously ID'd
# which are noted in selected_peaks.csv files

# ============================================
def grid_search_init(final_pb_peak_df, pb_gex_df, gene):
    # determine values to test in RFR parameters:
    gs_dict = {}
    npeaks = len(final_pb_peak_df)
    nsamples = len(final_pb_peak_df.columns)
    # number of decision trees
    gs_dict["n_estimators"] = [round((i + 0.25) * 0.1 * nsamples) for i in range(10)]
    # max depth of tree
    gs_dict["max_depth"] = [2, 5, 10, 20, 50, 75, 100]
    # min number of samples required to split an internal node; default = 2
    gs_dict["min_samples_split"] = [2, 4, 6, 8, 10]
    # min number of samples required to be at a leaf node; default = 1
    gs_dict["min_samples_leaf"] = [1, 5, 10]
    ### run model with gridsearch ###
    # convert to numpy arrays
    peaks_array = final_pb_peak_df.values.T
    gex_array = pb_gex_df.values.T
    # Create new DataFrames with preserved index names
    func_peaks_df = pd.DataFrame(peaks_array, columns=final_pb_peak_df.index, index=final_pb_peak_df.columns.tolist())
    func_gex_df = pd.DataFrame(gex_array, columns=pb_gex_df.index, index=final_pb_peak_df.columns.tolist())
    # Full Model
    model = RandomForestRegressor(random_state=0)
    model.fit(func_peaks_df, func_gex_df.values.ravel())
    # gridsearch
    search = GridSearchCV(estimator = model, param_grid = gs_dict, n_jobs = 20)
    search.fit(func_peaks_df, func_gex_df.values.ravel())
    #>>> print("Best parameters:", search.best_params_)
    #Best parameters: {'max_depth': 10, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 316
    model_new = RandomForestRegressor(random_state=0, max_depth = 10, min_samples_leaf = 5, min_samples_split = 2, n_estimators = 316)
    model_new.fit(func_peaks_df, func_gex_df.values.ravel())


