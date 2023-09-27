#!/usr/sbin/anaconda

import fnmatch
import h5py
import multiprocessing
import numpy as np
from numpy import mean
from numpy import absolute
from numpy import sqrt
import os
import pandas as pd
from scipy import sparse, io
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import sys

from sklearn.ensemble import GradientBoostingRegressor

os.chdir("/storage/home/mfisher42/scProjects/Predict_GEX/Groups_Celltypes_Split_Pseudos_peakfilt10perc_paretofront_08302023")
from misc_helper_functions import load_files_with_match

### 07-14-2023 ###
# This script includes functions to perform
# leave-one-out cross-validation
# Addition: 08-16-2023
# performing cross validation using training and testing
# sets from splitting pseudobulks in half

# ============================================
def select_peaks(test, selected_peaks, gene, outdir): # get peaks from test results
    #test_name = test.split("_")[2] + "_" + test.split("_")[3]
    test_name = test
    npeaks = selected_peaks.loc[selected_peaks["Result"] == test, "nPeaks"].item()
    if "RFR" in test_name:
        if "rf_ranker" in test_name:
            test_dir = "rf_ranker"
        elif "perm_ranker" in test_name:
            test_dir = "rf_permranker"
        elif "dropcol_ranker" in test_name:
            test_dir = "rf_dropcolranker"
    elif "LinReg" in test_name:
        if "perm_ranker" in test_name:
            test_dir = "linreg_permranker"
        elif "dropcol_ranker" in test_name:
            test_dir = "linreg_dropcolranker"
    # load peak list
    peak_dir = outdir + "/" + test_dir
    match = "_" + str(npeaks) + "peaks_"
    peak_list_df = pd.read_csv(load_files_with_match(peak_dir, match)[0])
    final_peak_list = peak_list_df["Peak"].tolist()
    return final_peak_list, test_dir

# ============================================
def grid_search_init(sub_pb_peak_df, pb_gex_df, gene, test, outdir):
    # determine values to test in RFR parameters:
    gs_dict = {}
    npeaks = len(sub_pb_peak_df)
    nsamples = len(sub_pb_peak_df.columns)
    # number of decision trees
    gs_dict["n_estimators"] = [round((i + 0.25) * 0.1 * 150) for i in range(10)]
    # max depth of tree
    gs_dict["max_depth"] = [2, 5, 10, 20, 50, 75, 100]
    # min number of samples required to split an internal node; default = 2
    gs_dict["min_samples_split"] = [2, 4, 6, 8, 10]
    # min number of samples required to be at a leaf node; default = 1
    gs_dict["min_samples_leaf"] = [1, 5, 10]
    ### run model with gridsearch ###
    # convert to numpy arrays, create new DataFrames with preserved index names
    peaks_array = sub_pb_peak_df.values.T
    gex_array = pb_gex_df.values.T
    func_peaks_df = pd.DataFrame(peaks_array, columns=sub_pb_peak_df.index, index=sub_pb_peak_df.columns.tolist())
    func_gex_df = pd.DataFrame(gex_array, columns=pb_gex_df.index, index=sub_pb_peak_df.columns.tolist())
    # Full Model
    model = RandomForestRegressor(random_state=0)
    model.fit(func_peaks_df, func_gex_df.values.ravel())
    # gridsearch
    #search = GridSearchCV(estimator = model, param_grid = gs_dict, n_jobs = 20, scoring=make_scorer(r2_score))
    search = GridSearchCV(estimator = model, param_grid = gs_dict, n_jobs = 20, scoring="neg_mean_squared_error")
    search.fit(func_peaks_df, func_gex_df.values.ravel())
    # get best paramaets
    best_params = search.best_params_
    return best_params

# ============================================
def XGBoost_grid_search_init(sub_pb_peak_df, pb_gex_df, gene, test, outdir):
    # determine values to test in RFR parameters:
    gs_dict = {}
    npeaks = len(sub_pb_peak_df)
    nsamples = len(sub_pb_peak_df.columns)
    # number of decision trees
    gs_dict["n_estimators"] = [round((i + 0.25) * 0.1 * 150) for i in range(10)]
    # max depth of tree
    gs_dict["max_depth"] = [2, 5, 10, 20, 50, 75, 100]
    # min number of samples required to split an internal node; default = 2
    gs_dict["min_samples_split"] = [2, 4, 6, 8, 10]
    # min number of samples required to be at a leaf node; default = 1
    gs_dict["min_samples_leaf"] = [1, 5, 10]
    ### run model with gridsearch ###
    # convert to numpy arrays, create new DataFrames with preserved index names
    peaks_array = sub_pb_peak_df.values.T
    gex_array = pb_gex_df.values.T
    func_peaks_df = pd.DataFrame(peaks_array, columns=sub_pb_peak_df.index, index=sub_pb_peak_df.columns.tolist())
    func_gex_df = pd.DataFrame(gex_array, columns=pb_gex_df.index, index=sub_pb_peak_df.columns.tolist())
    # Full Model
    model = GradientBoostingRegressor(random_state=0)
    model.fit(func_peaks_df, func_gex_df.values.ravel())
    # gridsearch
    #search = GridSearchCV(estimator = model, param_grid = gs_dict, n_jobs = 20, scoring=make_scorer(r2_score))
    search = GridSearchCV(estimator = model, param_grid = gs_dict, n_jobs = 20, scoring="neg_mean_squared_error")
    search.fit(func_peaks_df, func_gex_df.values.ravel())
    # get best paramaets
    best_params = search.best_params_
    return best_params

# ============================================
def cross_validation_fun(test, training_sub_pb_peak_df, training_pb_gex_df_sub, testing_sub_pb_peak_df, testing_pb_gex_df_sub, gene, outdir): # cross validations: LOO and pseudo-split
    print("Running LOO cross validation")
    # select model type, LinReg or RFR:
    if "_RFR_" in test:
        # perform grid search
        print("Performing grid search")
        best_params = grid_search_init(training_sub_pb_peak_df, training_pb_gex_df_sub, gene, test, outdir)
        print("Grid search done")
        model = RandomForestRegressor(random_state = 1, **best_params)
    elif "_LinReg_" in test:
        model = LinearRegression()
    elif "_XGBoost_" in test:
        # perform GridSearch
        print("Performing grid search")
        best_params = XGBoost_grid_search_init(training_sub_pb_peak_df, training_pb_gex_df_sub, gene, test, outdir)
        print("Grid search done")
        model = GradientBoostingRegressor(random_state = 1, **best_params)
    # format input data; first convert to numpy arrays
    training_peaks_array = training_sub_pb_peak_df.values.T
    training_gex_array = training_pb_gex_df_sub.values.T
    testing_peaks_array = testing_sub_pb_peak_df.values.T
    testing_gex_array = testing_pb_gex_df_sub.values.T
    # Create new DataFrames with preserved index names
    training_func_peaks_df = pd.DataFrame(training_peaks_array, columns=training_sub_pb_peak_df.index, index=training_sub_pb_peak_df.columns.tolist())
    training_func_gex_df = pd.DataFrame(training_gex_array, columns=training_pb_gex_df_sub.index, index=training_pb_gex_df_sub.columns.tolist())
    testing_func_peaks_df = pd.DataFrame(testing_peaks_array, columns=testing_sub_pb_peak_df.index, index=testing_sub_pb_peak_df.columns.tolist())
    testing_func_gex_df = pd.DataFrame(testing_gex_array, columns=testing_pb_gex_df_sub.index, index=testing_pb_gex_df_sub.columns.tolist())
    # 1.0) Inititate LOO and get predicted values
    loo = LeaveOneOut()
    LOO_y_pred_list = []
    LOO_actual_values = []
    for train_index, test_index in loo.split(training_func_peaks_df.index):
        X_train, X_test = training_func_peaks_df.iloc[train_index], training_func_peaks_df.iloc[test_index]
        y_train, y_test = training_func_gex_df.iloc[train_index], training_func_gex_df.iloc[test_index]
        # fit model
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        LOO_y_pred_list.append(y_pred[0])
        LOO_actual_values.append(y_test.values[0][0])
    # 2.0) Initiate pseudo-split cross validation and get predicted values
    X_train = training_func_peaks_df
    X_test = testing_func_peaks_df
    y_train = training_func_gex_df
    y_test = testing_func_gex_df
    model.fit(X_train, y_train.values.ravel())
    SPLIT_y_pred = model.predict(X_test).flatten().tolist()
    SPLIT_actual_values = y_test.values.flatten().tolist()
    return LOO_y_pred_list, LOO_actual_values, SPLIT_y_pred, SPLIT_actual_values

# ============================================
def make_cv_plot(predicted, actual, filename):
    fig, ax = plt.subplots()    
    plt.scatter(actual, predicted)
    plt.xlabel("Actual GEX")
    plt.ylabel("Predicted GEX")
    plt.title("Predicted vs Actual")
    # add line of best fit
    slope, intercept = np.polyfit(actual, predicted, deg=1)
    line_of_best_fit = slope * np.array(actual) + intercept
    plt.plot(actual, line_of_best_fit, color='red', linestyle='--')
    # add r2 value
    r2 = r2_score(actual, predicted)
    r2_text = f'R2 = {r2:.2f}'
    ax.text(0.05, 0.95, r2_text, transform=ax.transAxes, ha='left', va='top')
    plt.savefig(filename, format='pdf')
    return r2

# ============================================
#def split_cross_val(test, training_pb_peak_df, training_gex_peak_df, testing_pb_peak_df, testing_gex_peak_df, gene, outdir):
#    print("Running pseudobulk split cross validation")
    
