#!/usr/sbin/anaconda

import argparse
import h5py
import numpy as np
import os
import pandas as pd
from scipy import sparse, io
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
import sys

# 6-29-2023
# This script contains feature ranking functions to be imported to the main predict_gex.py script.
# Each ranker function returns a sorted features dataframe.
# The feature_selector() function checks the final output of models
# and computes the pareto front to identify the optimal
# set of peaks to include in the model.

# ============================================
def rf_ranker(model, gene, func_peaks_df, outdir): # rank features using random forest regression
    # assess features
    feature_importances = model.feature_importances_
    sorted_indices = feature_importances.argsort()[::-1]
    sorted_feature_importances = feature_importances[sorted_indices]
    sorted_features = func_peaks_df.columns[sorted_indices]
    sorted_features_df = pd.DataFrame({'Importance': sorted_feature_importances, 'Peak': sorted_features})
    # write peaks and ranks to output
    npeaks = len(feature_importances)
    filename = outdir + "/" + gene + "_" + str(npeaks) + "peaks_rfranker_importance.csv"
    sorted_features_df.to_csv(filename, index=False)
    # return sorted features df
    return sorted_features_df

# ============================================
def perm_ranker(baseline, gene, func_peaks_df, outdir): # rank features with permutation importance
    # use permutation importance to rank features and assess features
    sorted_indices = baseline["importances_mean"].argsort()[::-1]
    sorted_feature_importances = baseline["importances_mean"][sorted_indices] 
    sorted_features = func_peaks_df.columns[sorted_indices]
    sorted_features_df = pd.DataFrame({'Importance': sorted_feature_importances, 'Peak': sorted_features})
    # write peaks and ranks to output
    npeaks = len(sorted_features_df)
    filename = outdir + "/" + gene + "_" + str(npeaks) + "peaks_permranker_importance.csv"
    sorted_features_df.to_csv(filename, index=False)
    # return sorted features df
    return sorted_features_df

# ============================================
def RF_dropcolumn_importance(func_peaks_df, func_gex_df, gene, outdir):
    # Train the baseline model
    baseline_model = RandomForestRegressor()
    baseline_model.fit(func_peaks_df, func_gex_df.values.ravel())
    baseline_pred = baseline_model.predict(func_peaks_df)
    baseline_score = mean_squared_error(func_gex_df, baseline_pred)
    # store feature importances
    feature_importance = {}
    # iterate over feature
    if len(func_peaks_df.columns) > 1:
        for feature in func_peaks_df.columns:
            peaks_modified = func_peaks_df.drop(columns=[feature])
            # Train the modified model
            modified_model = RandomForestRegressor()
            modified_model.fit(peaks_modified, func_gex_df.values.ravel())
            modified_pred = modified_model.predict(peaks_modified)
            modified_score = mean_squared_error(func_gex_df.values.ravel(), modified_pred)
            # check performance drop
            drop = modified_score - baseline_score
            # Store the drop in performance as feature importance
            feature_importance[feature] = drop
    elif len(func_peaks_df.columns) == 1:
        feature = func_peaks_df.columns[0]
        feature_importance[feature] = 0
    # sort features based on performance drop
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_importance_df = pd.DataFrame(sorted_importance, columns=['Peak', 'Drop in Performance'])
    # write peaks and ranks to output
    npeaks = len(sorted_importance_df)
    filename = outdir + "/" + gene + "_" + str(npeaks) + "peaks_dropcolumn_importance.csv"
    sorted_importance_df.to_csv(filename, index=False)
    return sorted_importance_df

# ============================================
def LinReg_dropcolumn_importance(func_peaks_df, func_gex_df, gene, outdir):
    # Train the baseline model
    baseline_model = LinearRegression()
    baseline_model.fit(func_peaks_df, func_gex_df.values.ravel())
    baseline_pred = baseline_model.predict(func_peaks_df)
    baseline_score = mean_squared_error(func_gex_df, baseline_pred)
    # store feature importances
    feature_importance = {}
    # iterate over features
    if len(func_peaks_df.columns) > 1:
        for feature in func_peaks_df.columns:
            peaks_modified = func_peaks_df.drop(columns=[feature])
            # Train the modified model
            modified_model = LinearRegression()
            modified_model.fit(peaks_modified, func_gex_df.values.ravel())
            modified_pred = modified_model.predict(peaks_modified)
            modified_score = mean_squared_error(func_gex_df.values.ravel(), modified_pred)
            # check performance drop
            drop = modified_score - baseline_score
            # Store the drop in performance as feature importance
            feature_importance[feature] = drop
    elif len(func_peaks_df.columns) == 1:
        feature = func_peaks_df.columns[0]
        feature_importance[feature] = 0
    # sort features based on performance drop
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_importance_df = pd.DataFrame(sorted_importance, columns=['Peak', 'Drop in Performance'])
    # write peaks and ranks to output
    npeaks = len(sorted_importance_df)
    filename = outdir + "/" + gene + "_" + str(npeaks) + "peaks_linreg_dropcolumn_importance.csv"
    sorted_importance_df.to_csv(filename, index=False)
    return sorted_importance_df

# ============================================
#def feature_selector(gene, outdir):
#    # get all ranker result files
#    gene_outdir = outdir
#    feat_dict = {}
#    for filename in os.listdir(gene_outdir):
#        if "_ranker_results.txt" in filename:
#            print(filename)
#            file = os.path.join(gene_outdir, filename)
#            # read in each ranker_results.txt file:
#            ranker_file = pd.read_csv(file, sep = ",")
#            # select peaks
#            model_all = ranker_file["R2"].iloc[0]
#            model_one = ranker_file["R2"].iloc[-1]
#            deviation_threshold =  (model_all - model_one) * 0.1
#            deviations_from_start = np.abs(ranker_file['R2'] - ranker_file.iloc[0]['R2'])
#            # pick peak
#            condition = deviations_from_start > deviation_threshold
#            selected_index = condition.idxmax()
#            # get nPeaks and R2 value 
#            npeaks = int(ranker_file["nPeaks"].iloc[[selected_index]])
#            r2 = float(ranker_file["R2"].iloc[selected_index])
#            feat_dict[filename] = [npeaks, r2]
#    # save elbow points in one file
#    best_feats = pd.DataFrame.from_dict(feat_dict, orient = "index")
#    best_feats = best_feats.rename(columns={0:"nPeaks", 1: "R2"})
#    best_feats = best_feats.reset_index().rename(columns={"index": "Result"})
#    outfilename = gene_outdir + "/selected_peaks.csv"
#    best_feats.to_csv(outfilename, index=False)

# ============================================
def pareto_frontier(Xs, Ys, maxX=True, maxY=True):
    paretoPoints = []
    if maxX:
        maxX = float('-inf')
        for x, y in zip(Xs, Ys):
            if x >= maxX:
                maxX = x
                maxY = y
                paretoPoints.append((x, y))
    else:
        maxY = float('-inf')
        for x, y in zip(Xs, Ys):
            if y >= maxY:
                maxY = y
                maxX = x
                paretoPoints.append((x, y))
    return paretoPoints

# ============================================
def feature_selector(gene, outdir):
    # get all ranker result files
    gene_outdir = outdir
    feat_dict = {}
    for filename in os.listdir(gene_outdir):
        if "_ranker_results.txt" in filename:
            print(filename)
            file = os.path.join(gene_outdir, filename)
            # read in each ranker_results.txt file:
            data = pd.read_csv(file, sep = ",")
            # sort data
            data_sorted = data.sort_values(by=["nPeaks", "R2"])
            # Extract nPeaks and R2 columns
            nPeaks_values = data_sorted["nPeaks"].tolist()
            R2_values = data_sorted["R2"].tolist()
            # Find the Pareto front
            pareto_points = pareto_frontier(nPeaks_values, R2_values, maxX=False, maxY=False)
            filtered_data = [(x, y) for x, y in pareto_points if y != 1.0]
            # best peakset:
            best_peaks = filtered_data[-1]
            # get nPeaks and R2 value for elbow point
            npeaks = int(best_peaks[0])
            r2 = float(best_peaks[1])
            feat_dict[filename] = [npeaks, r2]
    # save elbow points in one file
    best_feats = pd.DataFrame.from_dict(feat_dict, orient = "index")
    best_feats = best_feats.rename(columns={0:"nPeaks", 1: "R2"})
    best_feats = best_feats.reset_index().rename(columns={"index": "Result"})
    outfilename = gene_outdir + "/selected_peaks.csv"
    best_feats.to_csv(outfilename, index=False)





