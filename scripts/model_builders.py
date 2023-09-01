#!/usr/sbin/anaconda

import argparse
import h5py
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

os.chdir("/storage/home/mfisher42/scProjects/Predict_GEX/Groups_Celltypes_Split_Pseudos_peakfilt10perc_paretofront_08302023")
from feature_selection import rf_ranker, perm_ranker, RF_dropcolumn_importance, LinReg_dropcolumn_importance, feature_selector


### 07-10-2023 ###
# This script contains two model building functions:
# 1.) Random Forest Regression models
# 2.) Lienar Regression models
# These functions are used in testing_predict_gex.py

# ============================================
def build_RFR_model(pb_peak_df, gex_peak_df, gene, outdir, test): # BSW based on ranked importance; rerank after each model is built
    # convert to numpy arrays
    peaks_array = pb_peak_df.values.T
    gex_array = gex_peak_df.values.T
    # Create new DataFrames with preserved index names
    func_peaks_df = pd.DataFrame(peaks_array, columns=pb_peak_df.index, index=pb_peak_df.columns.tolist())
    func_gex_df = pd.DataFrame(gex_array, columns=gex_peak_df.index, index=pb_peak_df.columns.tolist())
    # Full Model
    model = RandomForestRegressor()
    model.fit(func_peaks_df, func_gex_df.values.ravel())
    # Rank peaks:
    if test == "rf_ranker":
        print("RF rf_ranker: " + gene)
        test_outdir = outdir + "/" + "rf_ranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        sorted_features_df = rf_ranker(model, gene, func_peaks_df, test_outdir)
    elif test == "perm_ranker":
        print("RF perm_ranker: " + gene)
        test_outdir = outdir + "/" + "rf_permranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        baseline = permutation_importance(model, func_peaks_df, func_gex_df)
        sorted_features_df = perm_ranker(baseline, gene, func_peaks_df, test_outdir)
    elif test == "dropcol_ranker":
        print("RF dropcol_ranker: " + gene)
        test_outdir = outdir + "/" + "rf_dropcolranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        sorted_features_df = RF_dropcolumn_importance(func_peaks_df, func_gex_df, gene, test_outdir)
    npeaks = len(sorted_features_df)
    results_dict = {npeaks: model.score(func_peaks_df, func_gex_df)}
    # Iteratively remove peaks in order of importance (least to most)
    peak_list = sorted_features_df["Peak"][::-1].tolist()
    while len(peak_list) > 1:
        # remove peak, rerun model
        peak = peak_list[0]
        #print("removing " + peak)
        peak_list.remove(peak)
        func_peaks_sub = func_peaks_df[peak_list]
        model.fit(func_peaks_sub, func_gex_df.values.ravel())
        npeaks = len(peak_list)
        # add R2 value to dictionary
        results_dict[npeaks] = model.score(func_peaks_sub, func_gex_df)
        # rerank peaks
        if test == "rf_ranker":
            sorted_features_df = rf_ranker(model, gene, func_peaks_sub, test_outdir)
        elif test == "perm_ranker":
            baseline = permutation_importance(model, func_peaks_sub, func_gex_df)
            sorted_features_df = perm_ranker(baseline, gene, func_peaks_sub, test_outdir)
        elif test == "dropcol_ranker":
            sorted_features_df = RF_dropcolumn_importance(func_peaks_sub, func_gex_df, gene, test_outdir)
        # elif drop column
        peak_list = sorted_features_df["Peak"][::-1].tolist()
    # save results for each model
    final_df = pd.DataFrame(results_dict.items(), columns=["nPeaks", "R2"])
    filename = outdir + "/" +  gene + "_RFR_" + test + "_results.txt"
    final_df.to_csv(filename, index=False)

# ============================================
def build_LinReg_model(pb_peak_df, gex_peak_df, gene, outdir, test): # BSW based on ranked importance
    # convert to numpy arrays
    peaks_array = pb_peak_df.values.T
    gex_array = gex_peak_df.values.T
    # Create new DataFrames with preserved index names
    func_peaks_df = pd.DataFrame(peaks_array, columns=pb_peak_df.index, index=pb_peak_df.columns.tolist())
    func_gex_df = pd.DataFrame(gex_array, columns=gex_peak_df.index, index=pb_peak_df.columns.tolist())
    # Full Model
    model = LinearRegression()
    model.fit(func_peaks_df, func_gex_df)
    # Rank peaks:
    if test == "perm_ranker":
        print("LinReg perm_ranker: " + gene)
        test_outdir = outdir + "/" + "linreg_permranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        baseline = permutation_importance(model, func_peaks_df, func_gex_df)
        sorted_features_df = perm_ranker(baseline, gene, func_peaks_df, test_outdir)
    elif test == "dropcol_ranker":
        print("LinReg dropcol_ranker: " + gene)
        test_outdir = outdir + "/" + "linreg_dropcolranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        sorted_features_df = RF_dropcolumn_importance(func_peaks_df, func_gex_df, gene, test_outdir)
    npeaks = len(sorted_features_df)
    results_dict = {npeaks: model.score(func_peaks_df, func_gex_df)}
    # Iteratively remove peaks in order of importance (least to most)
    peak_list = sorted_features_df["Peak"][::-1].tolist()
    while len(peak_list) > 1:
        # remove peak, rerun model
        peak = peak_list[0]
        #print("removing " + peak)
        peak_list.remove(peak)
        func_peaks_sub = func_peaks_df[peak_list]
        model.fit(func_peaks_sub, func_gex_df.values.ravel())
        npeaks = len(peak_list)
        # add R2 value to dictionary
        results_dict[npeaks] = model.score(func_peaks_sub, func_gex_df)
        # rerank peaks
        if test == "perm_ranker":
            baseline = permutation_importance(model, func_peaks_sub, func_gex_df)
            sorted_features_df = perm_ranker(baseline, gene, func_peaks_sub, test_outdir)
        elif test == "dropcol_ranker":
            sorted_features_df = RF_dropcolumn_importance(func_peaks_sub, func_gex_df, gene, test_outdir)
        # elif drop column
        peak_list = sorted_features_df["Peak"][::-1].tolist()
    # save results for each model
    final_df = pd.DataFrame(results_dict.items(), columns=["nPeaks", "R2"])
    filename = outdir + "/" + gene + "_LinReg_" + test + "_results.txt"
    final_df.to_csv(filename, index=False)


