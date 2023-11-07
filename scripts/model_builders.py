#!/usr/sbin/anaconda

import joblib
import lightgbm as lgbm
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, GroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb

import warnings
from sklearn.exceptions import DataConversionWarning

os.chdir("/storage/home/mfisher42/scProjects/Predict_GEX/Multitest_kfoldcv_95featselect_hyperparam_10312023")
from feature_selection import rf_ranker, xgb_ranker, lgbm_ranker, perm_ranker, RF_dropcolumn_importance, LR_dropcolumn_importance, XGB_dropcolumn_importance, LGBM_dropcolumn_importance

# 10-12-2023
# This script contains functions which builds models and ranks features.
# The general workflow is to use nested k-fold cross validation for hyperparameter tuning and model evaluation.

# ============================================
def RF_init_peakranker(model, best_params, func_peaks_df, func_gex_df,  gene_outdir, test, gene):
    if test == "rf_ranker":
        print("RF rf_ranker: " + gene)
        test_outdir = gene_outdir + "/" + "rf_ranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        sorted_features_df = rf_ranker(model, gene, func_peaks_df, test_outdir)
    elif test == "perm_ranker":
        print("RF perm_ranker: " + gene)
        test_outdir = gene_outdir + "/" + "rf_permranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        baseline = permutation_importance(model, func_peaks_df, func_gex_df)
        sorted_features_df = perm_ranker(baseline, gene, func_peaks_df, test_outdir)
    elif test == "dropcol_ranker":
        print("RF dropcol_ranker: " + gene)
        test_outdir = gene_outdir + "/" + "rf_dropcolranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        sorted_features_df = RF_dropcolumn_importance(best_params, func_peaks_df, func_gex_df, gene, test_outdir)
        sorted_features_df.columns = ["Peak", "Importance"]
    npeaks = len(sorted_features_df)
    return npeaks, sorted_features_df, test_outdir

# ============================================
def LR_init_peakranker(model, func_peaks_df, func_gex_df,  gene_outdir, test, gene):
    if test == "perm_ranker":
        print("LR perm_ranker: " + gene)
        test_outdir = gene_outdir + "/" + "lr_permranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        baseline = permutation_importance(model, func_peaks_df, func_gex_df)
        sorted_features_df = perm_ranker(baseline, gene, func_peaks_df, test_outdir)
    elif test == "dropcol_ranker":
        print("LR dropcol_ranker: " + gene)
        test_outdir = gene_outdir + "/" + "lr_dropcolranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        sorted_features_df = LR_dropcolumn_importance(func_peaks_df, func_gex_df, gene, test_outdir)
        sorted_features_df.columns = ["Peak", "Importance"]
    npeaks = len(sorted_features_df)
    return npeaks, sorted_features_df, test_outdir

# ============================================
def XGB_init_peakranker(model, best_params, func_peaks_df, func_gex_df,  gene_outdir, test, gene):
    if test == "xgb_ranker":
        print("XGB xgb_ranker: " + gene)
        test_outdir = gene_outdir + "/" + "xgb_ranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        sorted_features_df = xgb_ranker(model, gene, func_peaks_df, test_outdir)
    elif test == "perm_ranker":
        print("XGB perm_ranker: " + gene)
        test_outdir = gene_outdir + "/" + "xgb_permranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        baseline = permutation_importance(model, func_peaks_df, func_gex_df)
        sorted_features_df = perm_ranker(baseline, gene, func_peaks_df, test_outdir)
    elif test == "dropcol_ranker":
        print("XGB dropcol_ranker: " + gene)
        test_outdir = gene_outdir + "/" + "xgb_dropcolranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        sorted_features_df = XGB_dropcolumn_importance(best_params, func_peaks_df, func_gex_df, gene, test_outdir)
        sorted_features_df.columns = ["Peak", "Importance"]
    npeaks = len(sorted_features_df)
    return npeaks, sorted_features_df, test_outdir

# ============================================
def LGBM_init_peakranker(model, best_params, func_peaks_df, func_gex_df,  gene_outdir, test, gene):
    if test == "lgbm_ranker":
        print("LGBM lgbm_ranker: " + gene)
        test_outdir = gene_outdir + "/" + "lgbm_ranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        sorted_features_df = lgbm_ranker(model, gene, func_peaks_df, test_outdir)
    elif test == "perm_ranker":
        print("LGBM perm_ranker: " + gene)
        test_outdir = gene_outdir + "/" + "lgbm_permranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        baseline = permutation_importance(model, func_peaks_df, func_gex_df)
        sorted_features_df = perm_ranker(baseline, gene, func_peaks_df, test_outdir)
    elif test == "dropcol_ranker":
        print("LGBM dropcol_ranker: " + gene)
        test_outdir = gene_outdir + "/" + "lgbm_dropcolranker"
        if not os.path.exists(test_outdir):
            os.makedirs(test_outdir)
        sorted_features_df = LGBM_dropcolumn_importance(best_params, func_peaks_df, func_gex_df, gene, test_outdir)
        sorted_features_df.columns = ["Peak", "Importance"]
    npeaks = len(sorted_features_df)
    return npeaks, sorted_features_df, test_outdir

# ============================================
def RFR_gridsearch(func_peaks_df, func_gex_df):
    ### define parameter grid:
    gs_dict = {}
    # number of decision trees
    gs_dict["n_estimators"] = [5, 15, 30, 50, 100]
    # max depth of tree
    gs_dict["max_depth"] = [2, 5, 10, 20, None]
    # min number of samples required to split an internal node; default = 2
    gs_dict["min_samples_split"] = [2, 4, 8, 20]
    # min number of samples required to be at a leaf node; default = 1
    gs_dict["min_samples_leaf"] = [1, 5, 20, 40]
    # max features provided to each tree in a forest
    gs_dict["max_features"] = [round(math.sqrt(len(func_peaks_df.columns)))]
    # define model
    rf = RandomForestRegressor()
    # define grid search cross validation loops
    inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 0)
    # define inner CV for parameter search
    gs_model = GridSearchCV(estimator = rf, param_grid = gs_dict, cv = inner_cv, n_jobs = -1)
    gs_model.fit(func_peaks_df, func_gex_df.values.ravel())
    best_params = gs_model.best_params_
    return best_params

# ============================================
def XGB_gridsearch(func_peaks_df, func_gex_df):
    ### define parameter grid:
    gs_dict = {}
    # number of decision trees
    gs_dict["n_estimators"] = [5, 15, 30, 50, 100]
    # max depth of tree; default = 6
    gs_dict["max_depth"] = [2, 3, 5, 10, 20, None]
    # min sum of instance weight needed in a child; min # of instances needed to be in each node; default = 1
    gs_dict["min_child_weight"] = [1, 2, 4, 8, 20]
    # L1 regularization term on weights. Increasing this value will make model more conservative.
    gs_dict["alpha"] = [0]
    # learning rate; Step size shrinkage used in update to prevents overfitting.
    gs_dict["learning_rate"] = [0.01, 0.1, 0.2, 0.3]
    # importance type
    gs_dict["importance_type"] = ["total_gain"]
    # fractino of observations to be randomly sampled for each tree
    gs_dict["subsample"] = [0.5, 1]
    # define model
    xgb_mod = xgb.XGBRegressor()
    # define grid search cross validation loops
    inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 0)
    # define inner CV for parameter search
    gs_model = GridSearchCV(estimator = xgb_mod, param_grid = gs_dict, cv = inner_cv, n_jobs = -1)
    gs_model.fit(func_peaks_df, func_gex_df.values.ravel())
    best_params = gs_model.best_params_
    return best_params

# ============================================
def LGBM_gridsearch(func_peaks_df, func_gex_df):
    ### define parameter grid:
    gs_dict = {}
    # number of decision trees
    gs_dict["n_estimators"] = [30, 50, 100, 200]
    # max depth of tree; default = 6
    gs_dict["max_depth"] = [2, 3, 5, 10]
    # min sum of instance weight needed in a child; min # of instances needed to be in each node; default = 1
    gs_dict["min_child_weight"] = [1, 2, 4, 8]
    # L1 regularization term on weights. Increasing this value will make model more conservative.
    gs_dict["reg_alpha"] = [0.0, 0.1]
    # learning rate; Step size shrinkage used in update to prevents overfitting.
    gs_dict["learning_rate"] = [0.01, 0.05, 0.1]
    # subsample_for_bin: # samples for constructing bins
    gs_dict["subsample_for_bin"] = [200, 300, 400]
    # fractino of observations to be randomly sampled for each tree
    gs_dict["subsample"] = [0.5, 1]
    # num_leaves: controls model complexity
    gs_dict["num_leaves"] = [4, 9, 25, 50]
    # define model
    lgbm_mod = lgbm.LGBMRegressor()
    # define grid search cross validation loops
    inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 0)
    # define inner CV for parameter search
    gs_model = RandomizedSearchCV(estimator = lgbm_mod, param_distributions = gs_dict, cv = inner_cv, n_jobs = 15)
    gs_model.fit(func_peaks_df, func_gex_df.values.ravel())
    best_params = gs_model.best_params_
    return best_params

# ============================================
def init_RFR_kfold_crossval(model, best_params, func_peaks_df, func_gex_df, gene_outdir, test, gene):
    num_kfold_columns = 3
    folds_per_column = 5
    skf_columns = [StratifiedKFold(n_splits = folds_per_column, shuffle = True, random_state = 0) for seed in range(num_kfold_columns)]
    # Lists to store the fold scores
    r2_fold_scores = []
    peak_importance_dict = {}
    # Perform cross-validation and store trained models
    for column_idx, skf in enumerate(skf_columns):
        for fold, (train_idx, test_idx) in enumerate(skf.split(func_peaks_df, func_gex_df[gene])):
            X_train, y_train = func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
            X_test, y_test = func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
            # run model
            model.fit(X_train, y_train)
            # Predict on the test set
            y_pred = model.predict(X_test)
            # Evaluate the model
            score = model.score(X_test, y_test)
            r2_fold_scores.append(score)
            # cross validation feature ranking
            npeaks, sorted_features_df, test_outdir = RF_init_peakranker(model, best_params, X_train, y_train, gene_outdir, test, gene)
            # save predicted vs actual for k-fold
            pred_act_dir = test_outdir + "/cross_validations_all_peaks"
            if not os.path.exists(pred_act_dir):
                os.makedirs(pred_act_dir)
            y_test = y_test.copy()
            y_test["Predicted"] = y_pred.tolist()
            outname = pred_act_dir + "/Column_" + str(column_idx) + "_Fold_" + str(fold) + ".csv"
            y_test.to_csv(outname)
            # sort features
            sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
            for peak, importance_values in sorted_feats_dict.items():
                if peak in peak_importance_dict:
                    peak_importance_dict[peak].extend(importance_values)
                else:
                    peak_importance_dict[peak] = importance_values
    return r2_fold_scores, peak_importance_dict, test_outdir

# ============================================
def init_XGB_kfold_crossval(model, best_params, func_peaks_df, func_gex_df, gene_outdir, test, gene):
    num_kfold_columns = 3
    folds_per_column = 5
    skf_columns = [StratifiedKFold(n_splits = folds_per_column, shuffle = True, random_state = 0) for seed in range(num_kfold_columns)]
    # Lists to store the fold scores
    r2_fold_scores = []
    peak_importance_dict = {}
    # Perform cross-validation and store trained models
    for column_idx, skf in enumerate(skf_columns):
        for fold, (train_idx, test_idx) in enumerate(skf.split(func_peaks_df, func_gex_df[gene])):
            X_train, y_train = func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
            X_test, y_test = func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
            # run model
            model.fit(X_train, y_train)
            # Predict on the test set
            y_pred = model.predict(X_test)
            # Evaluate the model
            score = model.score(X_test, y_test)
            r2_fold_scores.append(score)
            # cross validation feature ranking
            npeaks, sorted_features_df, test_outdir = XGB_init_peakranker(model, best_params, X_train, y_train, gene_outdir, test, gene)
            # save predicted vs actual for k-fold
            pred_act_dir = test_outdir + "/cross_validations_all_peaks"
            if not os.path.exists(pred_act_dir):
                os.makedirs(pred_act_dir)
            y_test = y_test.copy()
            y_test["Predicted"] = y_pred.tolist()
            outname = pred_act_dir + "/Column_" + str(column_idx) + "_Fold_" + str(fold) + ".csv"
            y_test.to_csv(outname)
            # sort features
            sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
            for peak, importance_values in sorted_feats_dict.items():
                if peak in peak_importance_dict:
                    peak_importance_dict[peak].extend(importance_values)
                else:
                    peak_importance_dict[peak] = importance_values
    return r2_fold_scores, peak_importance_dict, test_outdir

# ============================================
def init_LR_kfold_crossval(model, func_peaks_df, func_gex_df, gene_outdir, test, gene):
    num_kfold_columns = 3
    folds_per_column = 5
    skf_columns = [StratifiedKFold(n_splits = folds_per_column, shuffle = True, random_state = 0) for seed in range(num_kfold_columns)]
    # Lists to store the fold scores
    r2_fold_scores = []
    peak_importance_dict = {}
    # Perform cross-validation and store trained models
    for column_idx, skf in enumerate(skf_columns):
        for fold, (train_idx, test_idx) in enumerate(skf.split(func_peaks_df, func_gex_df[gene])):
            X_train, y_train = func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
            X_test, y_test = func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
            # run model
            model.fit(X_train, y_train)
            # Predict on the test set
            y_pred = model.predict(X_test)
            # Evaluate the model
            score = model.score(X_test, y_test)
            r2_fold_scores.append(score)
            # cross validation feature ranking
            npeaks, sorted_features_df, test_outdir = LR_init_peakranker(model, func_peaks_df, func_gex_df,  gene_outdir, test, gene)
            # save predicted vs actual for k-fold
            pred_act_dir = test_outdir + "/cross_validations_all_peaks"
            if not os.path.exists(pred_act_dir):
                os.makedirs(pred_act_dir)
            y_test = y_test.copy()
            y_test["Predicted"] = y_pred.tolist()
            outname = pred_act_dir + "/Column_" + str(column_idx) + "_Fold_" + str(fold) + ".csv"
            y_test.to_csv(outname)
            # sort features
            sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
            for peak, importance_values in sorted_feats_dict.items():
                if peak in peak_importance_dict:
                    peak_importance_dict[peak].extend(importance_values)
                else:
                    peak_importance_dict[peak] = importance_values
    return r2_fold_scores, peak_importance_dict, test_outdir

# ============================================
def init_LGBM_kfold_crossval(model, best_params, func_peaks_df, func_gex_df, gene_outdir, test, gene):
    num_kfold_columns = 3
    folds_per_column = 5
    skf_columns = [StratifiedKFold(n_splits = folds_per_column, shuffle = True, random_state = 0) for seed in range(num_kfold_columns)]
    # Lists to store the fold scores
    r2_fold_scores = []
    peak_importance_dict = {}
    # Perform cross-validation and store trained models
    for column_idx, skf in enumerate(skf_columns):
        for fold, (train_idx, test_idx) in enumerate(skf.split(func_peaks_df, func_gex_df[gene])):
            X_train, y_train = func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
            X_test, y_test = func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
            # run model
            model.fit(X_train, y_train)
            # Predict on the test set
            y_pred = model.predict(X_test)
            # Evaluate the model
            score = model.score(X_test, y_test)
            r2_fold_scores.append(score)
            # cross validation feature ranking
            npeaks, sorted_features_df, test_outdir = LGBM_init_peakranker(model, best_params, X_train, y_train, gene_outdir, test, gene)
            # save predicted vs actual for k-fold
            pred_act_dir = test_outdir + "/cross_validations_all_peaks"
            if not os.path.exists(pred_act_dir):
                os.makedirs(pred_act_dir)
            y_test = y_test.copy()
            y_test["Predicted"] = y_pred.tolist()
            outname = pred_act_dir + "/Column_" + str(column_idx) + "_Fold_" + str(fold) + ".csv"
            y_test.to_csv(outname)
            # sort features
            sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
            for peak, importance_values in sorted_feats_dict.items():
                if peak in peak_importance_dict:
                    peak_importance_dict[peak].extend(importance_values)
                else:
                    peak_importance_dict[peak] = importance_values
    return r2_fold_scores, peak_importance_dict, test_outdir

# ============================================
def avg_feature_importances(peak_importance_dict):
    # Calculate the average of the values for each key in the dictionary
    average_importance_dict = {}
    for peak, importance_values in peak_importance_dict.items():
        average_importance = sum(importance_values) / len(importance_values)
        average_importance_dict[peak] = average_importance
    # Create a DataFrame from the dictionary
    average_importance_df = pd.DataFrame(list(average_importance_dict.items()), columns = ["Peak", "Average Importance"])
    average_importance_df = average_importance_df.sort_values(by = "Average Importance", ascending = False)
    # Reset the index to have the peak as one column and average importance as another
    average_importance_df = average_importance_df.reset_index(drop = True)
    return average_importance_df

# ============================================
def build_RFR_model(pb_peak_df, gex_peak_df, gene, gene_outdir, test):
    warnings.filterwarnings(action = "ignore", category = DataConversionWarning)
    warnings.filterwarnings("ignore", category = UserWarning)
    # convert to numpy arrays
    peaks_array = pb_peak_df.values.T
    gex_array = gex_peak_df.values.T
    # Create new DataFrames with preserved index names
    func_peaks_df = pd.DataFrame(peaks_array, columns=pb_peak_df.index, index=pb_peak_df.columns.tolist())
    func_gex_df = pd.DataFrame(gex_array, columns=gex_peak_df.index, index=pb_peak_df.columns.tolist())
    ### optimize parameters with gridsearch
    best_params = RFR_gridsearch(func_peaks_df, func_gex_df)
    model = RandomForestRegressor(n_estimators = best_params['n_estimators'], max_depth = best_params['max_depth'], min_samples_split = best_params['min_samples_split'], min_samples_leaf = best_params['min_samples_leaf'], max_features = best_params['max_features'])
    ### perform k-fold cross validation for optimized model
    r2_fold_scores, peak_importance_dict, test_outdir = init_RFR_kfold_crossval(model, best_params, func_peaks_df, func_gex_df, gene_outdir, test, gene)
    # Calculate the average R2 cross validation score
    average_score = np.mean(r2_fold_scores)
    print(f"Average Score: {average_score}")
    npeaks = len(func_peaks_df.columns)
    results_dict = {npeaks: average_score}
    # Calculate the average of the values for each key in the dictionary
    average_importance_df = avg_feature_importances(peak_importance_dict)
    # Run/Test model on top 95% cumulative important peaks
    total = average_importance_df["Average Importance"].sum()
    thresh = total*.95
    current_sum = 0
    rows_to_keep = []
    # Extract peaks
    for index, row in average_importance_df.iterrows():
        current_sum += row["Average Importance"]
        rows_to_keep.append(index)
        if current_sum > thresh:
            break
    # Slice the DataFrame to keep the top rows
    extracted_average_importance_df = average_importance_df.loc[rows_to_keep]
    # rerun model and k-fold cross validation with extracted peaks
    sub_func_peaks_df = func_peaks_df[extracted_average_importance_df["Peak"].tolist()]
    ### optimize parameters with gridsearch
    best_params = RFR_gridsearch(sub_func_peaks_df, func_gex_df)
    model = RandomForestRegressor(n_estimators = best_params['n_estimators'], max_depth = best_params['max_depth'], min_samples_split = best_params['min_samples_split'], min_samples_leaf = best_params['min_samples_leaf'], max_features = best_params['max_features'])
    ### perform k-fold cross validation for optimized model
    num_kfold_columns = 3
    folds_per_column = 5
    skf_columns = [StratifiedKFold(n_splits = folds_per_column, shuffle = True, random_state = 0) for seed in range(num_kfold_columns)]
    r2_fold_scores = []
    peak_importance_dict = {}
    # Perform cross-validation and store trained models
    for column_idx, skf in enumerate(skf_columns):
        for fold, (train_idx, test_idx) in enumerate(skf.split(sub_func_peaks_df, func_gex_df[gene])):
            X_train, y_train = sub_func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
            X_test, y_test = sub_func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
            # run model
            model.fit(X_train, y_train)
            # Predict on the test set
            y_pred = model.predict(X_test)
            # Evaluate the model
            score = model.score(X_test, y_test)
            r2_fold_scores.append(score)
            # save predicted vs actual for k-fold
            pred_act_dir = test_outdir + "/cross_validations_top95_peaks"
            if not os.path.exists(pred_act_dir):
                os.makedirs(pred_act_dir)
            y_test = y_test.copy()
            y_test["Predicted"] = y_pred.tolist()
            outname = pred_act_dir + "/Column_" + str(column_idx) + "_Fold_" + str(fold) + ".csv"
            y_test.to_csv(outname)
            # cross validation model feature ranking
            if test == "rf_ranker":
                sorted_features_df = rf_ranker(model, gene, sub_func_peaks_df, test_outdir)
            elif test == "perm_ranker":
                baseline = permutation_importance(model, X_train, y_train)
                sorted_features_df = perm_ranker(baseline, gene, sub_func_peaks_df, test_outdir)
            elif test == "dropcol_ranker":
                sorted_features_df = RF_dropcolumn_importance(best_params, X_train, y_train, gene, test_outdir)
            # sort atac peaks
            sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
            for peak, importance_values in sorted_feats_dict.items():
                if peak in peak_importance_dict:
                    peak_importance_dict[peak].extend(importance_values)
                else:
                    peak_importance_dict[peak] = importance_values
    # Calculate the average R2 score
    average_score = np.mean(r2_fold_scores)
    print(f"Average Score: {average_score}")
    npeaks = len(sub_func_peaks_df.columns)
    # add R2 value to dictionary
    results_dict[npeaks] = average_score
    # Calculate the average of the values for each key in the dictionary
    average_importance_df = avg_feature_importances(peak_importance_dict)
    # save results for each model
    final_df = pd.DataFrame(results_dict.items(), columns = ["nPeaks", "R2"])
    filename = gene_outdir + "/" + gene + "_RFR_" + test + "_results.txt"
    final_df.to_csv(filename, index = False)

# ============================================
def build_LR_model(pb_peak_df, gex_peak_df, gene, gene_outdir, test):
    # convert to numpy arrays
    peaks_array = pb_peak_df.values.T
    gex_array = gex_peak_df.values.T
    # Create new DataFrames with preserved index names
    func_peaks_df = pd.DataFrame(peaks_array, columns=pb_peak_df.index, index=pb_peak_df.columns.tolist())
    func_gex_df = pd.DataFrame(gex_array, columns=gex_peak_df.index, index=pb_peak_df.columns.tolist())
    ### perform k-fold cross validation for optimized model
    model = LinearRegression()
    r2_fold_scores, peak_importance_dict, test_outdir = init_LR_kfold_crossval(model, func_peaks_df, func_gex_df, gene_outdir, test, gene)
    # Calculate the average R2 cross validation score
    average_score = np.mean(r2_fold_scores)
    print(f"Average Score: {average_score}")
    npeaks = len(func_peaks_df.columns)
    results_dict = {npeaks: average_score}
    # Calculate the average of the values for each key in the dictionary
    average_importance_df = avg_feature_importances(peak_importance_dict)
    # Run/Test model on top 95% cumulative important peaks
    total = average_importance_df["Average Importance"].sum()
    thresh = total*.95
    current_sum = 0
    rows_to_keep = []
    # Extract peaks
    for index, row in average_importance_df.iterrows():
        current_sum += row["Average Importance"]
        rows_to_keep.append(index)
        if current_sum > thresh:
            break
    # Slice the DataFrame to keep the top rows
    extracted_average_importance_df = average_importance_df.loc[rows_to_keep]
    # rerun model and k-fold cross validation with extracted peaks
    sub_func_peaks_df = func_peaks_df[extracted_average_importance_df["Peak"].tolist()]
    # Lists to store the fold scores
    r2_fold_scores = []
    peak_importance_dict = {}
    num_kfold_columns = 3
    folds_per_column = 5
    skf_columns = [StratifiedKFold(n_splits = folds_per_column, shuffle = True, random_state = 0) for seed in range(num_kfold_columns)]
    # Perform cross-validation and store trained models
    for column_idx, skf in enumerate(skf_columns):
        for fold, (train_idx, test_idx) in enumerate(skf.split(sub_func_peaks_df, func_gex_df[gene])):
            X_train, y_train = sub_func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
            X_test, y_test = sub_func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
            # run model
            model.fit(X_train, y_train)
            # Predict on the test set
            y_pred = model.predict(X_test)
            # Evaluate the model
            score = model.score(X_test, y_test)
            r2_fold_scores.append(score)
            # cross validation model feature ranking
            if test == "perm_ranker":
                    baseline = permutation_importance(model, X_train, y_train)
                    sorted_features_df = perm_ranker(baseline, gene, sub_func_peaks_df, test_outdir)
            elif test == "dropcol_ranker":
                    sorted_features_df = LR_dropcolumn_importance(X_train, y_train, gene, test_outdir)
            # save predicted vs actual for k-fold
            pred_act_dir = test_outdir + "/cross_validations_top95_peaks"
            if not os.path.exists(pred_act_dir):
                os.makedirs(pred_act_dir)
            y_test = y_test.copy()
            y_test["Predicted"] = y_pred.tolist()
            outname = pred_act_dir + "/Column_" + str(column_idx) + "_Fold_" + str(fold) + ".csv"
            y_test.to_csv(outname)
            # sort atac peaks
            sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
            for peak, importance_values in sorted_feats_dict.items():
                if peak in peak_importance_dict:
                    peak_importance_dict[peak].extend(importance_values)
                else:
                    peak_importance_dict[peak] = importance_values
    # Calculate the average R2 score
    average_score = np.mean(r2_fold_scores)
    print(f"Average Score: {average_score}")
    npeaks = len(peak_importance_dict)
    # add R2 value to dictionary
    results_dict[npeaks] = average_score
    # Calculate the average of the values for each key in the dictionary
    average_importance_df = avg_feature_importances(peak_importance_dict)
    # save results for each model
    final_df = pd.DataFrame(results_dict.items(), columns = ["nPeaks", "R2"])
    filename = gene_outdir + "/" + gene + "_LR_" + test + "_results.txt"
    final_df.to_csv(filename, index = False)

# ============================================
def build_XGB_model(pb_peak_df, gex_peak_df, gene, gene_outdir, test):
    # convert to numpy arrays
    peaks_array = pb_peak_df.values.T
    gex_array = gex_peak_df.values.T
    # Create new DataFrames with preserved index names
    func_peaks_df = pd.DataFrame(peaks_array, columns=pb_peak_df.index, index=pb_peak_df.columns.tolist())
    func_gex_df = pd.DataFrame(gex_array, columns=gex_peak_df.index, index=pb_peak_df.columns.tolist())
    ### optimize parameters with gridsearch
    best_params = XGB_gridsearch(func_peaks_df, func_gex_df)
    model = xgb.XGBRegressor(alpha = best_params["alpha"], importance_type = best_params["importance_type"], learning_rate = best_params["learning_rate"], max_depth = best_params["max_depth"], min_child_weight = best_params["min_child_weight"], n_estimators = best_params["n_estimators"], subsample = best_params["subsample"])
    ### Perform k-fold CV for optimized model
    r2_fold_scores, peak_importance_dict, test_outdir = init_XGB_kfold_crossval(model, best_params, func_peaks_df, func_gex_df, gene_outdir, test, gene)
    # Calculate the average R2 cross validation score
    average_score = np.mean(r2_fold_scores)
    print(f"Average Score: {average_score}")
    npeaks = len(func_peaks_df.columns)
    results_dict = {npeaks: average_score}
    # Calculate the average of the values for each key in the dictionary
    average_importance_df = avg_feature_importances(peak_importance_dict)
    # Run/Test model on top 95% cumulative important peaks
    total = average_importance_df["Average Importance"].sum()
    thresh = total*.95
    current_sum = 0
    rows_to_keep = []
    # Extract peaks
    for index, row in average_importance_df.iterrows():
        current_sum += row["Average Importance"]
        rows_to_keep.append(index)
        if current_sum > thresh:
            break
    # Slice the DataFrame to keep the top rows
    extracted_average_importance_df = average_importance_df.loc[rows_to_keep]
    # rerun model and k-fold cross validation with extracted peaks
    sub_func_peaks_df = func_peaks_df[extracted_average_importance_df["Peak"].tolist()]
    ### optimize parameters with gridsearch
    best_params = XGB_gridsearch(sub_func_peaks_df, func_gex_df)
    model = xgb.XGBRegressor(alpha = best_params["alpha"], importance_type = best_params["importance_type"], learning_rate = best_params["learning_rate"], max_depth = best_params["max_depth"], min_child_weight = best_params["min_child_weight"], n_estimators = best_params["n_estimators"], subsample = best_params["subsample"])
    ### perform k-fold cross validation for optimized model
    num_kfold_columns = 3
    folds_per_column = 5
    skf_columns = [StratifiedKFold(n_splits = folds_per_column, shuffle = True, random_state = 0) for seed in range(num_kfold_columns)]
    r2_fold_scores = []
    peak_importance_dict = {}
    # Perform cross-validation and store trained models
    for column_idx, skf in enumerate(skf_columns):
        for fold, (train_idx, test_idx) in enumerate(skf.split(sub_func_peaks_df, func_gex_df[gene])):
            X_train, y_train = sub_func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
            X_test, y_test = sub_func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
            # run model
            model.fit(X_train, y_train)
            # Predict on the test set
            y_pred = model.predict(X_test)
            # Evaluate the model
            score = model.score(X_test, y_test)
            r2_fold_scores.append(score)
            # cross validation model feature ranking
            if test == "xgb_ranker":
                sorted_features_df = xgb_ranker(model, gene, sub_func_peaks_df, test_outdir)
            elif test == "perm_ranker":
                baseline = permutation_importance(model, X_train, y_train)
                sorted_features_df = perm_ranker(baseline, gene, sub_func_peaks_df, test_outdir)
            elif test == "dropcol_ranker":
                sorted_features_df = XGB_dropcolumn_importance(best_params, X_train, y_train, gene, test_outdir)
            # save predicted vs actual for k-fold
            pred_act_dir = test_outdir + "/cross_validations_top95_peaks"
            if not os.path.exists(pred_act_dir):
                os.makedirs(pred_act_dir)
            y_test = y_test.copy()
            y_test["Predicted"] = y_pred.tolist()
            outname = pred_act_dir + "/Column_" + str(column_idx) + "_Fold_" + str(fold) + ".csv"
            y_test.to_csv(outname)
            # sort atac peaks
            sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
            for peak, importance_values in sorted_feats_dict.items():
                if peak in peak_importance_dict:
                    peak_importance_dict[peak].extend(importance_values)
                else:
                    peak_importance_dict[peak] = importance_values
    # Calculate the average R2 score
    average_score = np.mean(r2_fold_scores)
    print(f"Average Score: {average_score}")
    npeaks = len(sub_func_peaks_df.columns)
    # add R2 value to dictionary
    results_dict[npeaks] = average_score
    # Calculate the average of the values for each key in the dictionary
    extracted_average_importance_df = average_importance_df.loc[rows_to_keep]
    # save results for each model
    final_df = pd.DataFrame(results_dict.items(), columns = ["nPeaks", "R2"])
    filename = gene_outdir + "/" + gene + "_XGB_" + test + "_results.txt"
    final_df.to_csv(filename, index = False)

# ============================================
def build_LGBM_model(pb_peak_df, gex_peak_df, gene, gene_outdir, test):
    # convert to numpy arrays
    peaks_array = pb_peak_df.values.T
    gex_array = gex_peak_df.values.T
    # Create new DataFrames with preserved index names
    func_peaks_df = pd.DataFrame(peaks_array, columns=pb_peak_df.index, index=pb_peak_df.columns.tolist())
    func_gex_df = pd.DataFrame(gex_array, columns=gex_peak_df.index, index=pb_peak_df.columns.tolist())
    func_peaks_df.columns = func_peaks_df.columns.str.replace(':', '_').str.replace('-', '_') # [LightGBM] [Fatal] Do not support special JSON characters in feature name.
    ### optimize parameters with gridsearch
    best_params = LGBM_gridsearch(func_peaks_df, func_gex_df)
    model = lgbm.LGBMRegressor(**best_params)
    ### Perform k-fold CV for optimized model
    r2_fold_scores, peak_importance_dict, test_outdir = init_LGBM_kfold_crossval(model, best_params, func_peaks_df, func_gex_df, gene_outdir, test, gene)
    # fit and save optimized/trained model:
    model.fit(func_peaks_df, func_gex_df)
    model_name = test_outdir + "/trained_model_all_peaks.pkl"
    joblib.dump(model, model_name)
    # Calculate the average R2 cross validation score
    average_score = np.mean(r2_fold_scores)
    print(f"Average Score: {average_score}")
    npeaks = len(func_peaks_df.columns)
    results_dict = {npeaks: average_score}
    # Calculate the average of the values for each key in the dictionary
    average_importance_df = avg_feature_importances(peak_importance_dict)
    # Run/Test model on top 95% cumulative important peaks
    total = average_importance_df["Average Importance"].sum()
    thresh = total*.95
    current_sum = 0
    rows_to_keep = []
    # Extract peaks
    for index, row in average_importance_df.iterrows():
        current_sum += row["Average Importance"]
        rows_to_keep.append(index)
        if current_sum > thresh:
            break
    # Slice the DataFrame to keep the top rows
    extracted_average_importance_df = average_importance_df.loc[rows_to_keep]
    # rerun model and k-fold cross validation with extracted peaks
    sub_func_peaks_df = func_peaks_df[extracted_average_importance_df["Peak"].tolist()]
    ### optimize parameters with gridsearch
    best_params = LGBM_gridsearch(sub_func_peaks_df, func_gex_df)
    model = lgbm.LGBMRegressor(**best_params)
    # fit and save optimized/trained model:
    model.fit(sub_func_peaks_df, func_gex_df)
    model_name = test_outdir + "/trained_model_top95_peaks.pkl"
    joblib.dump(model, model_name)
    ### perform k-fold cross validation for optimized model
    num_kfold_columns = 3
    folds_per_column = 5
    skf_columns = [StratifiedKFold(n_splits = folds_per_column, shuffle = True, random_state = 0) for seed in range(num_kfold_columns)]
    r2_fold_scores = []
    peak_importance_dict = {}
    # Perform cross-validation and store trained models
    for column_idx, skf in enumerate(skf_columns):
        for fold, (train_idx, test_idx) in enumerate(skf.split(sub_func_peaks_df, func_gex_df[gene])):
            X_train, y_train = sub_func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
            X_test, y_test = sub_func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
            # run model
            model.fit(X_train, y_train)
            # Predict on the test set
            y_pred = model.predict(X_test)
            # Evaluate the model
            score = model.score(X_test, y_test)
            r2_fold_scores.append(score)
            # cross validation model feature ranking
            if test == "lgbm_ranker":
                sorted_features_df = lgbm_ranker(model, gene, sub_func_peaks_df, test_outdir)
            elif test == "perm_ranker":
                baseline = permutation_importance(model, X_train, y_train)
                sorted_features_df = perm_ranker(baseline, gene, sub_func_peaks_df, test_outdir)
            elif test == "dropcol_ranker":
                sorted_features_df = LGBM_dropcolumn_importance(best_params, X_train, y_train, gene, test_outdir)
            # save predicted vs actual for k-fold
            pred_act_dir = test_outdir + "/cross_validations_top95_peaks"
            if not os.path.exists(pred_act_dir):
                os.makedirs(pred_act_dir)
            y_test = y_test.copy()
            y_test["Predicted"] = y_pred.tolist()
            outname = pred_act_dir + "/Column_" + str(column_idx) + "_Fold_" + str(fold) + ".csv"
            y_test.to_csv(outname)
            # sort atac peaks
            sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
            for peak, importance_values in sorted_feats_dict.items():
                if peak in peak_importance_dict:
                    peak_importance_dict[peak].extend(importance_values)
                else:
                    peak_importance_dict[peak] = importance_values
    # Calculate the average R2 score
    average_score = np.mean(r2_fold_scores)
    print(f"Average Score: {average_score}")
    npeaks = len(sub_func_peaks_df.columns)
    # add R2 value to dictionary
    results_dict[npeaks] = average_score
    # Calculate the average of the values for each key in the dictionary
    extracted_average_importance_df = average_importance_df.loc[rows_to_keep]
    # save results for each model
    final_df = pd.DataFrame(results_dict.items(), columns = ["nPeaks", "R2"])
    filename = gene_outdir + "/" + gene + "_LGBM_" + test + "_results.txt"
    final_df.to_csv(filename, index = False)



