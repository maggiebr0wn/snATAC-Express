#!/usr/sbin/anaconda

import lightgbm as lgbm
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# 10-16-2023
# This script contains feature ranking functions to be imported to the main predict_gex.py script.

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
def xgb_ranker(model, gene, func_peaks_df, outdir): # rank features using XGBoost regression
    # assess features
    feature_importances = model.feature_importances_
    sorted_indices = feature_importances.argsort()[::-1]
    sorted_feature_importances = feature_importances[sorted_indices]
    sorted_features = func_peaks_df.columns[sorted_indices]
    sorted_features_df = pd.DataFrame({'Importance': sorted_feature_importances, 'Peak': sorted_features})
    # write peaks and ranks to output
    npeaks = len(feature_importances)
    filename = outdir + "/" + gene + "_" + str(npeaks) + "peaks_xgbranker_importance.csv"
    sorted_features_df.to_csv(filename, index=False)
    # return sorted features df
    return sorted_features_df
    
# ============================================
def lgbm_ranker(model, gene, func_peaks_df, outdir): # rank features using LightLGBM regression
    # assess features
    feature_importances = model.feature_importances_
    sorted_indices = feature_importances.argsort()[::-1]
    sorted_feature_importances = feature_importances[sorted_indices]
    sorted_features = func_peaks_df.columns[sorted_indices]
    sorted_features_df = pd.DataFrame({'Importance': sorted_feature_importances, 'Peak': sorted_features})
    # write peaks and ranks to output
    npeaks = len(feature_importances)
    filename = outdir + "/" + gene + "_" + str(npeaks) + "peaks_lgbm_ranker_importance.csv"
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
def RF_dropcolumn_importance(best_params, func_peaks_df, func_gex_df, gene, outdir):
    # Train the baseline model
    baseline_model = RandomForestRegressor(n_estimators = best_params['n_estimators'], max_depth = best_params['max_depth'], min_samples_split = best_params['min_samples_split'], min_samples_leaf = best_params['min_samples_leaf'], max_features = best_params['max_features'])
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
            modified_model = RandomForestRegressor(n_estimators = best_params['n_estimators'], max_depth = best_params['max_depth'], min_samples_split = best_params['min_samples_split'], min_samples_leaf = best_params['min_samples_leaf'], max_features = best_params['max_features'])
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
    sorted_importance_df = pd.DataFrame(sorted_importance, columns=['Peak', 'Importance'])
    # write peaks and ranks to output
    npeaks = len(sorted_importance_df)
    filename = outdir + "/" + gene + "_" + str(npeaks) + "peaks_dropcolumn_importance.csv"
    sorted_importance_df.to_csv(filename, index=False)
    return sorted_importance_df

# ============================================
def LR_dropcolumn_importance(func_peaks_df, func_gex_df, gene, outdir):
    # Train the baseline model
    baseline_model = LinearRegression()
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
    sorted_importance_df = pd.DataFrame(sorted_importance, columns=['Peak', 'Importance'])
    # write peaks and ranks to output
    npeaks = len(sorted_importance_df)
    filename = outdir + "/" + gene + "_" + str(npeaks) + "peaks_dropcolumn_importance.csv"
    sorted_importance_df.to_csv(filename, index=False)
    return sorted_importance_df

# ============================================
def XGB_dropcolumn_importance(best_params, func_peaks_df, func_gex_df, gene, outdir):
    # Train the baseline model
    baseline_model = xgb.XGBRegressor(alpha = best_params["alpha"], importance_type = best_params["importance_type"], learning_rate = best_params["learning_rate"], max_depth = best_params["max_depth"], min_child_weight = best_params["min_child_weight"], n_estimators = best_params["n_estimators"], subsample = best_params["subsample"])
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
            modified_model = xgb.XGBRegressor(alpha = best_params["alpha"], importance_type = best_params["importance_type"], learning_rate = best_params["learning_rate"], max_depth = best_params["max_depth"], min_child_weight = best_params["min_child_weight"], n_estimators = best_params["n_estimators"], subsample = best_params["subsample"])
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
    sorted_importance_df = pd.DataFrame(sorted_importance, columns=['Peak', 'Importance'])
    # write peaks and ranks to output
    npeaks = len(sorted_importance_df)
    filename = outdir + "/" + gene + "_" + str(npeaks) + "peaks_dropcolumn_importance.csv"
    sorted_importance_df.to_csv(filename, index=False)
    return sorted_importance_df

# ============================================
def LGBM_dropcolumn_importance(best_params, func_peaks_df, func_gex_df, gene, outdir):
    # Train the baseline model
    baseline_model = lgbm.LGBMRegressor(**best_params)
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
            modified_model = lgbm.LGBMRegressor(**best_params)
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
    sorted_importance_df = pd.DataFrame(sorted_importance, columns=['Peak', 'Importance'])
    # write peaks and ranks to output
    npeaks = len(sorted_importance_df)
    filename = outdir + "/" + gene + "_" + str(npeaks) + "peaks_dropcolumn_importance.csv"
    sorted_importance_df.to_csv(filename, index=False)
    return sorted_importance_df


# ============================================
def feature_selector(gene, gene_outdir):
    # get all ranker result files
    columnnames = ["gene", "celltype", "method", "peak_filter", "npeaks_kept", "cv_R2"]
    summary = pd.DataFrame(columns = columnnames)
    feat_dict = {}
    for filename in os.listdir(gene_outdir):
        if "_ranker_results.txt" in filename:
            print(filename)
            file = os.path.join(gene_outdir, filename)
            # read in each ranker_results.txt file:
            ranker_file = pd.read_csv(file, sep = ",")
            # select peaks with max CV R2
            max_r2_row = ranker_file[ranker_file["R2"] == ranker_file["R2"].max()]
            max_nPeaks = max_r2_row["nPeaks"].values[0]
            max_r2 = ranker_file["R2"].max()
            # get info for using all peaks
            all_nPeaks = ranker_file["nPeaks"][0]
            all_r2 = ranker_file["R2"][0]
            # get method
            method = "_".join(filename.split("_")[1:4])
            # add info to summary: selected peaks
            new_row_selected = [gene, "all", method, "Y", max_nPeaks, max_r2]
            summary.loc[len(summary)] = new_row_selected
            # add info to summary: all peaks
            new_row_all = [gene, "all", method, "N", all_nPeaks, all_r2]
            summary.loc[len(summary)] = new_row_all
    return summary






