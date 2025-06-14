#!/usr/bin/env python3
"""
Modified model builder for snATAC-Express to match original implementation
Each model function now builds TWO models:
1. Model with all peaks
2. Model with top 95% peaks (selected based on that model's importance)
"""

import os
import joblib
import lightgbm as lgbm
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import warnings
from sklearn.exceptions import DataConversionWarning

# Import feature selection utilities
from .feature_selection import (
    rf_ranker, xgb_ranker, lgbm_ranker, perm_ranker,
    RF_dropcolumn_importance, LR_dropcolumn_importance,
    XGB_dropcolumn_importance, LGBM_dropcolumn_importance
)

warnings.filterwarnings(action="ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ModelBuilder:
    """Model builder that matches original implementation structure"""
    
    def __init__(self, config):
        """Initialize model builder with configuration"""
        self.config = config
        self.random_seed = config.get('random_seed', 12345)
        
    def build_and_evaluate_model(self, model_name, X, y, gene, gene_outdir, test_method):
        """
        Build and evaluate a model with both all peaks and top 95% peaks
        This matches the original implementation structure
        """
        if model_name == 'linear_regression':
            return self._build_LR_model(X, y, gene, gene_outdir, test_method)
        elif model_name == 'random_forest':
            return self._build_RFR_model(X, y, gene, gene_outdir, test_method)
        elif model_name == 'xgboost':
            return self._build_XGB_model(X, y, gene, gene_outdir, test_method)
        elif model_name == 'lightgbm':
            return self._build_LGBM_model(X, y, gene, gene_outdir, test_method)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def _build_RFR_model(self, func_peaks_df, func_gex_df, gene, gene_outdir, test):
        """Build Random Forest model with all peaks and top 95% peaks"""
        # Grid search for best parameters
        best_params = self._RFR_gridsearch(func_peaks_df, func_gex_df)
        model = RandomForestRegressor(**best_params)
        
        # Perform k-fold cross validation for all peaks
        r2_fold_scores, peak_importance_dict, test_outdir = self._init_RFR_kfold_crossval(
            model, best_params, func_peaks_df, func_gex_df, gene_outdir, test, gene
        )
        
        # Fit and save model with all peaks
        model.fit(func_peaks_df, func_gex_df.values.ravel())
        model_name = os.path.join(test_outdir, "trained_model_all_peaks.pkl")
        joblib.dump(model, model_name)
        
        # Calculate average R2 for all peaks
        average_score_all = np.mean(r2_fold_scores)
        print(f"Average Score (all peaks): {average_score_all}")
        npeaks_all = len(func_peaks_df.columns)
        results_dict = {npeaks_all: average_score_all}
        
        # Calculate average feature importances
        average_importance_df = self._avg_feature_importances(peak_importance_dict)
        
        # Select top 95% cumulative important peaks
        total = average_importance_df["Average Importance"].sum()
        thresh = total * 0.95
        current_sum = 0
        rows_to_keep = []
        
        for index, row in average_importance_df.iterrows():
            current_sum += row["Average Importance"]
            rows_to_keep.append(index)
            if current_sum > thresh:
                break
        
        # Extract top 95% peaks
        extracted_average_importance_df = average_importance_df.loc[rows_to_keep]
        sub_func_peaks_df = func_peaks_df[extracted_average_importance_df["Peak"].tolist()]
        
        # Rebuild model with selected peaks
        best_params_95 = self._RFR_gridsearch(sub_func_peaks_df, func_gex_df)
        model_95 = RandomForestRegressor(**best_params_95)
        
        # Fit and save model with 95% peaks
        model_95.fit(sub_func_peaks_df, func_gex_df.values.ravel())
        model_name_95 = os.path.join(test_outdir, "trained_model_top95_peaks.pkl")
        joblib.dump(model_95, model_name_95)
        
        # Cross-validation for 95% peaks
        num_kfold_columns = 3
        folds_per_column = 5
        skf_columns = [StratifiedKFold(n_splits=folds_per_column, shuffle=True, random_state=0) 
                       for _ in range(num_kfold_columns)]
        
        r2_fold_scores_95 = []
        peak_importance_dict_95 = {}
        
        for column_idx, skf in enumerate(skf_columns):
            for fold, (train_idx, test_idx) in enumerate(skf.split(sub_func_peaks_df, func_gex_df[gene])):
                X_train, y_train = sub_func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
                X_test, y_test = sub_func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
                
                # Train model
                model_95.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model_95.predict(X_test)
                score = model_95.score(X_test, y_test)
                r2_fold_scores_95.append(score)
                
                # Save predictions
                pred_act_dir = os.path.join(test_outdir, "cross_validations_top95_peaks")
                os.makedirs(pred_act_dir, exist_ok=True)
                y_test_copy = y_test.copy()
                y_test_copy["Predicted"] = y_pred.tolist()
                outname = os.path.join(pred_act_dir, f"Column_{column_idx}_Fold_{fold}.csv")
                y_test_copy.to_csv(outname)
                
                # Feature ranking for 95% peaks
                if test == "rf_ranker":
                    sorted_features_df = rf_ranker(model_95, gene, sub_func_peaks_df, test_outdir)
                elif test == "perm_ranker":
                    baseline = permutation_importance(model_95, X_train, y_train)
                    sorted_features_df = perm_ranker(baseline, gene, sub_func_peaks_df, test_outdir)
                elif test == "dropcol_ranker":
                    sorted_features_df = RF_dropcolumn_importance(best_params_95, X_train, y_train, gene, test_outdir)
                
                # Aggregate importance
                sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
                for peak, importance_values in sorted_feats_dict.items():
                    if peak in peak_importance_dict_95:
                        peak_importance_dict_95[peak].extend(importance_values)
                    else:
                        peak_importance_dict_95[peak] = importance_values
        
        # Calculate average R2 for 95% peaks
        average_score_95 = np.mean(r2_fold_scores_95)
        print(f"Average Score (95% peaks): {average_score_95}")
        npeaks_95 = len(sub_func_peaks_df.columns)
        results_dict[npeaks_95] = average_score_95
        
        # Save results in organized directory structure
        model_results_dir = os.path.join(gene_outdir, "model_results")
        os.makedirs(model_results_dir, exist_ok=True)
        final_df = pd.DataFrame(results_dict.items(), columns=["nPeaks", "R2"])
        filename = os.path.join(model_results_dir, f"{gene}_RFR_{test}_results.txt")
        final_df.to_csv(filename, index=False)
        
        return {
            'all_peaks': {'n_peaks': npeaks_all, 'r2': average_score_all},
            '95_peaks': {'n_peaks': npeaks_95, 'r2': average_score_95}
        }
    
    def _build_LR_model(self, func_peaks_df, func_gex_df, gene, gene_outdir, test):
        """Build Linear Regression model with all peaks and top 95% peaks"""
        # Linear Regression doesn't need grid search
        model = LinearRegression()
        
        # Perform k-fold cross validation for all peaks
        r2_fold_scores, peak_importance_dict, test_outdir = self._init_LR_kfold_crossval(
            model, func_peaks_df, func_gex_df, gene_outdir, test, gene
        )
        
        # Fit and save model with all peaks
        model.fit(func_peaks_df, func_gex_df.values.ravel())
        model_name = os.path.join(test_outdir, "trained_model_all_peaks.pkl")
        joblib.dump(model, model_name)
        
        # Calculate average R2 for all peaks
        average_score_all = np.mean(r2_fold_scores)
        print(f"Average Score (all peaks): {average_score_all}")
        npeaks_all = len(func_peaks_df.columns)
        results_dict = {npeaks_all: average_score_all}
        
        # Calculate average feature importances
        average_importance_df = self._avg_feature_importances(peak_importance_dict)
        
        # Select top 95% cumulative important peaks
        total = average_importance_df["Average Importance"].sum()
        thresh = total * 0.95
        current_sum = 0
        rows_to_keep = []
        
        for index, row in average_importance_df.iterrows():
            current_sum += row["Average Importance"]
            rows_to_keep.append(index)
            if current_sum > thresh:
                break
        
        # Extract top 95% peaks
        extracted_average_importance_df = average_importance_df.loc[rows_to_keep]
        sub_func_peaks_df = func_peaks_df[extracted_average_importance_df["Peak"].tolist()]
        
        # Rebuild model with selected peaks
        model_95 = LinearRegression()
        model_95.fit(sub_func_peaks_df, func_gex_df.values.ravel())
        model_name_95 = os.path.join(test_outdir, "trained_model_top95_peaks.pkl")
        joblib.dump(model_95, model_name_95)
        
        # Cross-validation for 95% peaks
        num_kfold_columns = 3
        folds_per_column = 5
        skf_columns = [StratifiedKFold(n_splits=folds_per_column, shuffle=True, random_state=0) 
                       for _ in range(num_kfold_columns)]
        
        r2_fold_scores_95 = []
        peak_importance_dict_95 = {}
        
        for column_idx, skf in enumerate(skf_columns):
            for fold, (train_idx, test_idx) in enumerate(skf.split(sub_func_peaks_df, func_gex_df[gene])):
                X_train, y_train = sub_func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
                X_test, y_test = sub_func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
                
                # Train model
                model_95.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model_95.predict(X_test)
                score = model_95.score(X_test, y_test)
                r2_fold_scores_95.append(score)
                
                # Save predictions
                pred_act_dir = os.path.join(test_outdir, "cross_validations_top95_peaks")
                os.makedirs(pred_act_dir, exist_ok=True)
                y_test_copy = y_test.copy()
                y_test_copy["Predicted"] = y_pred.tolist()
                outname = os.path.join(pred_act_dir, f"Column_{column_idx}_Fold_{fold}.csv")
                y_test_copy.to_csv(outname)
                
                # Feature ranking for 95% peaks
                if test == "perm_ranker":
                    baseline = permutation_importance(model_95, X_train, y_train)
                    sorted_features_df = perm_ranker(baseline, gene, sub_func_peaks_df, test_outdir)
                elif test == "dropcol_ranker":
                    sorted_features_df = LR_dropcolumn_importance(X_train, y_train, gene, test_outdir)
                
                # Aggregate importance
                sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
                for peak, importance_values in sorted_feats_dict.items():
                    if peak in peak_importance_dict_95:
                        peak_importance_dict_95[peak].extend(importance_values)
                    else:
                        peak_importance_dict_95[peak] = importance_values
        
        # Calculate average R2 for 95% peaks
        average_score_95 = np.mean(r2_fold_scores_95)
        print(f"Average Score (95% peaks): {average_score_95}")
        npeaks_95 = len(peak_importance_dict_95)  # Note: original uses len(peak_importance_dict)
        results_dict[npeaks_95] = average_score_95
        
        # Save results in organized directory structure
        model_results_dir = os.path.join(gene_outdir, "model_results")
        os.makedirs(model_results_dir, exist_ok=True)
        final_df = pd.DataFrame(results_dict.items(), columns=["nPeaks", "R2"])
        filename = os.path.join(model_results_dir, f"{gene}_LR_{test}_results.txt")
        final_df.to_csv(filename, index=False)
        
        return {
            'all_peaks': {'n_peaks': npeaks_all, 'r2': average_score_all},
            '95_peaks': {'n_peaks': npeaks_95, 'r2': average_score_95}
        }
    
    def _build_XGB_model(self, func_peaks_df, func_gex_df, gene, gene_outdir, test):
        """Build XGBoost model with all peaks and top 95% peaks"""
        # Grid search for best parameters
        best_params = self._XGB_gridsearch(func_peaks_df, func_gex_df)
        model = xgb.XGBRegressor(**best_params)
        
        # Perform k-fold cross validation for all peaks
        r2_fold_scores, peak_importance_dict, test_outdir = self._init_XGB_kfold_crossval(
            model, best_params, func_peaks_df, func_gex_df, gene_outdir, test, gene
        )
        
        # Fit and save model with all peaks
        model.fit(func_peaks_df, func_gex_df.values.ravel())
        model_name = os.path.join(test_outdir, "trained_model_all_peaks.pkl")
        joblib.dump(model, model_name)
        
        # Calculate average R2 for all peaks
        average_score_all = np.mean(r2_fold_scores)
        print(f"Average Score (all peaks): {average_score_all}")
        npeaks_all = len(func_peaks_df.columns)
        results_dict = {npeaks_all: average_score_all}
        
        # Calculate average feature importances
        average_importance_df = self._avg_feature_importances(peak_importance_dict)
        
        # Select top 95% cumulative important peaks
        total = average_importance_df["Average Importance"].sum()
        thresh = total * 0.95
        current_sum = 0
        rows_to_keep = []
        
        for index, row in average_importance_df.iterrows():
            current_sum += row["Average Importance"]
            rows_to_keep.append(index)
            if current_sum > thresh:
                break
        
        # Extract top 95% peaks
        extracted_average_importance_df = average_importance_df.loc[rows_to_keep]
        sub_func_peaks_df = func_peaks_df[extracted_average_importance_df["Peak"].tolist()]
        
        # Rebuild model with selected peaks
        best_params_95 = self._XGB_gridsearch(sub_func_peaks_df, func_gex_df)
        model_95 = xgb.XGBRegressor(**best_params_95)
        
        # Fit and save model with 95% peaks
        model_95.fit(sub_func_peaks_df, func_gex_df.values.ravel())
        model_name_95 = os.path.join(test_outdir, "trained_model_top95_peaks.pkl")
        joblib.dump(model_95, model_name_95)
        
        # Cross-validation for 95% peaks
        num_kfold_columns = 3
        folds_per_column = 5
        skf_columns = [StratifiedKFold(n_splits=folds_per_column, shuffle=True, random_state=0) 
                       for _ in range(num_kfold_columns)]
        
        r2_fold_scores_95 = []
        peak_importance_dict_95 = {}
        
        for column_idx, skf in enumerate(skf_columns):
            for fold, (train_idx, test_idx) in enumerate(skf.split(sub_func_peaks_df, func_gex_df[gene])):
                X_train, y_train = sub_func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
                X_test, y_test = sub_func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
                
                # Train model
                model_95.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model_95.predict(X_test)
                score = model_95.score(X_test, y_test)
                r2_fold_scores_95.append(score)
                
                # Save predictions
                pred_act_dir = os.path.join(test_outdir, "cross_validations_top95_peaks")
                os.makedirs(pred_act_dir, exist_ok=True)
                y_test_copy = y_test.copy()
                y_test_copy["Predicted"] = y_pred.tolist()
                outname = os.path.join(pred_act_dir, f"Column_{column_idx}_Fold_{fold}.csv")
                y_test_copy.to_csv(outname)
                
                # Feature ranking for 95% peaks
                if test == "xgb_ranker":
                    sorted_features_df = xgb_ranker(model_95, gene, sub_func_peaks_df, test_outdir)
                elif test == "perm_ranker":
                    baseline = permutation_importance(model_95, X_train, y_train)
                    sorted_features_df = perm_ranker(baseline, gene, sub_func_peaks_df, test_outdir)
                elif test == "dropcol_ranker":
                    sorted_features_df = XGB_dropcolumn_importance(best_params_95, X_train, y_train, gene, test_outdir)
                
                # Aggregate importance
                sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
                for peak, importance_values in sorted_feats_dict.items():
                    if peak in peak_importance_dict_95:
                        peak_importance_dict_95[peak].extend(importance_values)
                    else:
                        peak_importance_dict_95[peak] = importance_values
        
        # Calculate average R2 for 95% peaks
        average_score_95 = np.mean(r2_fold_scores_95)
        print(f"Average Score (95% peaks): {average_score_95}")
        npeaks_95 = len(sub_func_peaks_df.columns)
        results_dict[npeaks_95] = average_score_95
        
        # Save results in organized directory structure
        model_results_dir = os.path.join(gene_outdir, "model_results")
        os.makedirs(model_results_dir, exist_ok=True)
        final_df = pd.DataFrame(results_dict.items(), columns=["nPeaks", "R2"])
        filename = os.path.join(model_results_dir, f"{gene}_XGB_{test}_results.txt")
        final_df.to_csv(filename, index=False)
        
        return {
            'all_peaks': {'n_peaks': npeaks_all, 'r2': average_score_all},
            '95_peaks': {'n_peaks': npeaks_95, 'r2': average_score_95}
        }
    
    def _build_LGBM_model(self, func_peaks_df, func_gex_df, gene, gene_outdir, test):
        """Build LightGBM model with all peaks and top 95% peaks - with hanging fix"""
        # Fix column names for LightGBM
        func_peaks_df = func_peaks_df.copy()
        func_peaks_df.columns = func_peaks_df.columns.str.replace(':', '_').str.replace('-', '_')
        
        # Grid search for best parameters
        best_params = self._LGBM_gridsearch(func_peaks_df, func_gex_df)
        
        # Create model with fixed parameters
        model = lgbm.LGBMRegressor(**best_params)
        
        # Suppress warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Perform k-fold cross validation for all peaks
            r2_fold_scores, peak_importance_dict, test_outdir = self._init_LGBM_kfold_crossval(
                model, best_params, func_peaks_df, func_gex_df, gene_outdir, test, gene
            )
            
            # Fit and save model with all peaks
            model.fit(func_peaks_df, func_gex_df.values.ravel())
            model_name = os.path.join(test_outdir, "trained_model_all_peaks.pkl")
            joblib.dump(model, model_name)
        
        # Calculate average R2 for all peaks
        average_score_all = np.mean(r2_fold_scores)
        print(f"Average Score (all peaks): {average_score_all}")
        npeaks_all = len(func_peaks_df.columns)
        results_dict = {npeaks_all: average_score_all}
        
        # Calculate average feature importances
        average_importance_df = self._avg_feature_importances(peak_importance_dict)
        
        # Select top 95% cumulative important peaks
        total = average_importance_df["Average Importance"].sum()
        thresh = total * 0.95
        current_sum = 0
        rows_to_keep = []
        
        for index, row in average_importance_df.iterrows():
            current_sum += row["Average Importance"]
            rows_to_keep.append(index)
            if current_sum > thresh:
                break
        
        # Extract top 95% peaks
        extracted_average_importance_df = average_importance_df.loc[rows_to_keep]
        sub_func_peaks_df = func_peaks_df[extracted_average_importance_df["Peak"].tolist()]
        
        # Rebuild model with selected peaks
        best_params_95 = self._LGBM_gridsearch(sub_func_peaks_df, func_gex_df)
        model_95 = lgbm.LGBMRegressor(**best_params_95)
        
        # Fit and save model with 95% peaks
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model_95.fit(sub_func_peaks_df, func_gex_df.values.ravel())
            model_name_95 = os.path.join(test_outdir, "trained_model_top95_peaks.pkl")
            joblib.dump(model_95, model_name_95)
        
        # Cross-validation for 95% peaks
        num_kfold_columns = 3
        folds_per_column = 5
        skf_columns = [StratifiedKFold(n_splits=folds_per_column, shuffle=True, random_state=0) 
                       for _ in range(num_kfold_columns)]
        
        r2_fold_scores_95 = []
        peak_importance_dict_95 = {}
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            for column_idx, skf in enumerate(skf_columns):
                for fold, (train_idx, test_idx) in enumerate(skf.split(sub_func_peaks_df, func_gex_df[gene])):
                    X_train, y_train = sub_func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
                    X_test, y_test = sub_func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
                    
                    # Train model
                    model_95.fit(X_train, y_train)
                    
                    # Predict and evaluate
                    y_pred = model_95.predict(X_test)
                    score = model_95.score(X_test, y_test)
                    r2_fold_scores_95.append(score)
                    
                    # Save predictions
                    pred_act_dir = os.path.join(test_outdir, "cross_validations_top95_peaks")
                    os.makedirs(pred_act_dir, exist_ok=True)
                    y_test_copy = y_test.copy()
                    y_test_copy["Predicted"] = y_pred.tolist()
                    outname = os.path.join(pred_act_dir, f"Column_{column_idx}_Fold_{fold}.csv")
                    y_test_copy.to_csv(outname)
                    
                    # Feature ranking for 95% peaks
                    if test == "lgbm_ranker":
                        sorted_features_df = lgbm_ranker(model_95, gene, sub_func_peaks_df, test_outdir)
                    elif test == "perm_ranker":
                        baseline = permutation_importance(model_95, X_train, y_train, n_jobs=1)
                        sorted_features_df = perm_ranker(baseline, gene, sub_func_peaks_df, test_outdir)
                    elif test == "dropcol_ranker":
                        sorted_features_df = LGBM_dropcolumn_importance(best_params_95, X_train, y_train, gene, test_outdir)
                    
                    # Aggregate importance
                    sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
                    for peak, importance_values in sorted_feats_dict.items():
                        if peak in peak_importance_dict_95:
                            peak_importance_dict_95[peak].extend(importance_values)
                        else:
                            peak_importance_dict_95[peak] = importance_values
        
        # Calculate average R2 for 95% peaks
        average_score_95 = np.mean(r2_fold_scores_95)
        print(f"Average Score (95% peaks): {average_score_95}")
        npeaks_95 = len(sub_func_peaks_df.columns)
        results_dict[npeaks_95] = average_score_95
        
        # Save results in organized directory structure
        model_results_dir = os.path.join(gene_outdir, "model_results")
        os.makedirs(model_results_dir, exist_ok=True)
        final_df = pd.DataFrame(results_dict.items(), columns=["nPeaks", "R2"])
        filename = os.path.join(model_results_dir, f"{gene}_LGBM_{test}_results.txt")
        final_df.to_csv(filename, index=False)
        
        return {
            'all_peaks': {'n_peaks': npeaks_all, 'r2': average_score_all},
            '95_peaks': {'n_peaks': npeaks_95, 'r2': average_score_95}
        }
    
    # Grid search methods
    def _RFR_gridsearch(self, func_peaks_df, func_gex_df):
        """Grid search for Random Forest"""
        gs_dict = {
            "n_estimators": [5, 15, 30, 50, 100],
            "max_depth": [2, 5, 10, 20, None],
            "min_samples_split": [2, 4, 8, 20],
            "min_samples_leaf": [1, 5, 20, 40],
            "max_features": [round(math.sqrt(len(func_peaks_df.columns)))]
        }
        rf = RandomForestRegressor()
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
        gs_model = GridSearchCV(estimator=rf, param_grid=gs_dict, cv=inner_cv, n_jobs=-1)
        gs_model.fit(func_peaks_df, func_gex_df.values.ravel())
        return gs_model.best_params_
    
    def _XGB_gridsearch(self, func_peaks_df, func_gex_df):
        """Grid search for XGBoost"""
        gs_dict = {
            "n_estimators": [5, 15, 30, 50, 100],
            "max_depth": [2, 3, 5, 10, 20, None],
            "min_child_weight": [1, 2, 4, 8, 20],
            "alpha": [0],
            "learning_rate": [0.01, 0.1, 0.2, 0.3],
            "importance_type": ["total_gain"],
            "subsample": [0.5, 1]
        }
        xgb_mod = xgb.XGBRegressor()
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
        gs_model = GridSearchCV(estimator=xgb_mod, param_grid=gs_dict, cv=inner_cv, n_jobs=-1)
        gs_model.fit(func_peaks_df, func_gex_df.values.ravel())
        return gs_model.best_params_
    
    def _LGBM_gridsearch(self, func_peaks_df, func_gex_df):
        """Grid search for LightGBM with hanging fix"""
        gs_dict = {
            "n_estimators": [30, 50, 100, 200],
            "max_depth": [2, 3, 5, 10],
            "min_child_weight": [1, 2, 4, 8],
            "reg_alpha": [0.0, 0.1],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample_for_bin": [200, 300, 400],
            "subsample": [0.5, 1],
            "num_leaves": [4, 9, 25, 50]
        }
        
        # Create base model with fixes for hanging
        lgbm_mod = lgbm.LGBMRegressor(
            force_col_wise=True,  # Force column-wise to avoid the 60+ second overhead
            verbosity=-1,         # Suppress warnings
            n_jobs=1,            # Use single thread for stability
            random_state=42
        )
        
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
        
        # Use RandomizedSearchCV for efficiency
        gs_model = RandomizedSearchCV(
            estimator=lgbm_mod, 
            param_distributions=gs_dict, 
            cv=inner_cv, 
            n_jobs=1,  # Single job to avoid conflicts
            n_iter=50,  # Limit iterations
            random_state=42
        )
        
        # Suppress warnings during fit
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            gs_model.fit(func_peaks_df, func_gex_df.values.ravel())
        
        # Add the fixed parameters to best params
        best_params = gs_model.best_params_.copy()
        best_params.update({
            'force_col_wise': True,
            'verbosity': -1,
            'n_jobs': 1,
            'random_state': 42
        })
        
        return best_params
    
    # Cross-validation methods (simplified versions shown here)
    def _init_RFR_kfold_crossval(self, model, best_params, func_peaks_df, func_gex_df, 
                                 gene_outdir, test, gene):
        """Random Forest k-fold cross-validation"""
        return self._generic_kfold_crossval(
            model, best_params, func_peaks_df, func_gex_df, gene_outdir, test, gene,
            "RF", RandomForestRegressor
        )
    
    def _init_XGB_kfold_crossval(self, model, best_params, func_peaks_df, func_gex_df, 
                                 gene_outdir, test, gene):
        """XGBoost k-fold cross-validation"""
        return self._generic_kfold_crossval(
            model, best_params, func_peaks_df, func_gex_df, gene_outdir, test, gene,
            "XGB", xgb.XGBRegressor
        )
    
    def _init_LGBM_kfold_crossval(self, model, best_params, func_peaks_df, func_gex_df, 
                                gene_outdir, test, gene):
        """LightGBM k-fold cross-validation with hanging fix"""
        num_kfold_columns = 3
        folds_per_column = 5
        skf_columns = [StratifiedKFold(n_splits=folds_per_column, shuffle=True, random_state=0) 
                    for _ in range(num_kfold_columns)]
        
        r2_fold_scores = []
        peak_importance_dict = {}
        
        # Determine output directory - use organized structure
        feature_rankings_dir = os.path.join(gene_outdir, "feature_rankings")
        os.makedirs(feature_rankings_dir, exist_ok=True)
        
        if test == "lgbm_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "lgbm_ranker")
        elif test == "perm_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "lgbm_permranker")
        elif test == "dropcol_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "lgbm_dropcolranker")
        else:
            test_outdir = os.path.join(feature_rankings_dir, test)
        
        os.makedirs(test_outdir, exist_ok=True)
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            for column_idx, skf in enumerate(skf_columns):
                for fold, (train_idx, test_idx) in enumerate(skf.split(func_peaks_df, func_gex_df[gene])):
                    X_train, y_train = func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
                    X_test, y_test = func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
                    
                    # Create fresh model with fixed params for each fold
                    fold_model = lgbm.LGBMRegressor(**best_params)
                    fold_model.fit(X_train, y_train)
                    
                    # Predict and evaluate
                    y_pred = fold_model.predict(X_test)
                    score = fold_model.score(X_test, y_test)
                    r2_fold_scores.append(score)
                    
                    # Feature ranking
                    npeaks, sorted_features_df, _ = self._LGBM_init_peakranker(
                        fold_model, best_params, X_train, y_train, gene_outdir, test, gene
                    )
                    
                    # Save predictions
                    pred_act_dir = os.path.join(test_outdir, "cross_validations_all_peaks")
                    os.makedirs(pred_act_dir, exist_ok=True)
                    y_test_copy = y_test.copy()
                    y_test_copy["Predicted"] = y_pred.tolist()
                    outname = os.path.join(pred_act_dir, f"Column_{column_idx}_Fold_{fold}.csv")
                    y_test_copy.to_csv(outname)
                    
                    # Aggregate importance
                    sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
                    for peak, importance_values in sorted_feats_dict.items():
                        if peak in peak_importance_dict:
                            peak_importance_dict[peak].extend(importance_values)
                        else:
                            peak_importance_dict[peak] = importance_values
        
        return r2_fold_scores, peak_importance_dict, test_outdir
    
    def _init_LR_kfold_crossval(self, model, func_peaks_df, func_gex_df, 
                                gene_outdir, test, gene):
        """Linear Regression k-fold cross-validation"""
        # LR doesn't have best_params
        return self._generic_kfold_crossval(
            model, {}, func_peaks_df, func_gex_df, gene_outdir, test, gene,
            "LR", LinearRegression
        )
    
    def _generic_kfold_crossval(self, model, best_params, func_peaks_df, func_gex_df,
                                gene_outdir, test, gene, model_prefix, model_class):
        """Generic k-fold cross-validation implementation"""
        num_kfold_columns = 3
        folds_per_column = 5
        skf_columns = [StratifiedKFold(n_splits=folds_per_column, shuffle=True, random_state=0) 
                       for _ in range(num_kfold_columns)]
        
        r2_fold_scores = []
        peak_importance_dict = {}
        
        # Determine output directory - use organized structure
        feature_rankings_dir = os.path.join(gene_outdir, "feature_rankings")
        os.makedirs(feature_rankings_dir, exist_ok=True)
        
        if test == "rf_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "rf_ranker")
        elif test == "xgb_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "xgb_ranker")
        elif test == "lgbm_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "lgbm_ranker")
        elif test == "perm_ranker":
            test_outdir = os.path.join(feature_rankings_dir, f"{model_prefix.lower()}_permranker")
        elif test == "dropcol_ranker":
            test_outdir = os.path.join(feature_rankings_dir, f"{model_prefix.lower()}_dropcolranker")
        else:
            test_outdir = os.path.join(feature_rankings_dir, test)
        
        os.makedirs(test_outdir, exist_ok=True)
        
        for column_idx, skf in enumerate(skf_columns):
            for fold, (train_idx, test_idx) in enumerate(skf.split(func_peaks_df, func_gex_df[gene])):
                X_train, y_train = func_peaks_df.iloc[train_idx], func_gex_df.iloc[train_idx]
                X_test, y_test = func_peaks_df.iloc[test_idx], func_gex_df.iloc[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model.predict(X_test)
                score = model.score(X_test, y_test)
                r2_fold_scores.append(score)
                
                # Feature ranking
                if model_prefix == "RF":
                    npeaks, sorted_features_df, _ = self._RF_init_peakranker(
                        model, best_params, X_train, y_train, gene_outdir, test, gene
                    )
                elif model_prefix == "XGB":
                    npeaks, sorted_features_df, _ = self._XGB_init_peakranker(
                        model, best_params, X_train, y_train, gene_outdir, test, gene
                    )
                elif model_prefix == "LGBM":
                    npeaks, sorted_features_df, _ = self._LGBM_init_peakranker(
                        model, best_params, X_train, y_train, gene_outdir, test, gene
                    )
                elif model_prefix == "LR":
                    npeaks, sorted_features_df, _ = self._LR_init_peakranker(
                        model, func_peaks_df, func_gex_df, gene_outdir, test, gene
                    )
                
                # Save predictions
                pred_act_dir = os.path.join(test_outdir, "cross_validations_all_peaks")
                os.makedirs(pred_act_dir, exist_ok=True)
                y_test_copy = y_test.copy()
                y_test_copy["Predicted"] = y_pred.tolist()
                outname = os.path.join(pred_act_dir, f"Column_{column_idx}_Fold_{fold}.csv")
                y_test_copy.to_csv(outname)
                
                # Aggregate importance
                sorted_feats_dict = sorted_features_df.groupby("Peak")["Importance"].apply(list).to_dict()
                for peak, importance_values in sorted_feats_dict.items():
                    if peak in peak_importance_dict:
                        peak_importance_dict[peak].extend(importance_values)
                    else:
                        peak_importance_dict[peak] = importance_values
        
        return r2_fold_scores, peak_importance_dict, test_outdir
    
    # Feature importance rankers
    def _RF_init_peakranker(self, model, best_params, func_peaks_df, func_gex_df, 
                            gene_outdir, test, gene):
        """Random Forest peak ranker"""
        # Use organized directory structure
        feature_rankings_dir = os.path.join(gene_outdir, "feature_rankings")
        os.makedirs(feature_rankings_dir, exist_ok=True)
        
        if test == "rf_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "rf_ranker")
            os.makedirs(test_outdir, exist_ok=True)
            sorted_features_df = rf_ranker(model, gene, func_peaks_df, test_outdir)
        elif test == "perm_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "rf_permranker")
            os.makedirs(test_outdir, exist_ok=True)
            baseline = permutation_importance(model, func_peaks_df, func_gex_df)
            sorted_features_df = perm_ranker(baseline, gene, func_peaks_df, test_outdir)
        elif test == "dropcol_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "rf_dropcolranker")
            os.makedirs(test_outdir, exist_ok=True)
            sorted_features_df = RF_dropcolumn_importance(best_params, func_peaks_df, 
                                                         func_gex_df, gene, test_outdir)
            sorted_features_df.columns = ["Peak", "Importance"]
        
        npeaks = len(sorted_features_df)
        return npeaks, sorted_features_df, test_outdir
    
    def _XGB_init_peakranker(self, model, best_params, func_peaks_df, func_gex_df, 
                             gene_outdir, test, gene):
        """XGBoost peak ranker"""
        # Use organized directory structure
        feature_rankings_dir = os.path.join(gene_outdir, "feature_rankings")
        os.makedirs(feature_rankings_dir, exist_ok=True)
        
        if test == "xgb_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "xgb_ranker")
            os.makedirs(test_outdir, exist_ok=True)
            sorted_features_df = xgb_ranker(model, gene, func_peaks_df, test_outdir)
        elif test == "perm_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "xgb_permranker")
            os.makedirs(test_outdir, exist_ok=True)
            baseline = permutation_importance(model, func_peaks_df, func_gex_df)
            sorted_features_df = perm_ranker(baseline, gene, func_peaks_df, test_outdir)
        elif test == "dropcol_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "xgb_dropcolranker")
            os.makedirs(test_outdir, exist_ok=True)
            sorted_features_df = XGB_dropcolumn_importance(best_params, func_peaks_df, 
                                                          func_gex_df, gene, test_outdir)
            sorted_features_df.columns = ["Peak", "Importance"]
        
        npeaks = len(sorted_features_df)
        return npeaks, sorted_features_df, test_outdir
    
    def _LGBM_init_peakranker(self, model, best_params, func_peaks_df, func_gex_df, 
                              gene_outdir, test, gene):
        """LightGBM peak ranker"""
        # Use organized directory structure
        feature_rankings_dir = os.path.join(gene_outdir, "feature_rankings")
        os.makedirs(feature_rankings_dir, exist_ok=True)
        
        if test == "lgbm_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "lgbm_ranker")
            os.makedirs(test_outdir, exist_ok=True)
            sorted_features_df = lgbm_ranker(model, gene, func_peaks_df, test_outdir)
        elif test == "perm_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "lgbm_permranker")
            os.makedirs(test_outdir, exist_ok=True)
            baseline = permutation_importance(model, func_peaks_df, func_gex_df)
            sorted_features_df = perm_ranker(baseline, gene, func_peaks_df, test_outdir)
        elif test == "dropcol_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "lgbm_dropcolranker")
            os.makedirs(test_outdir, exist_ok=True)
            sorted_features_df = LGBM_dropcolumn_importance(best_params, func_peaks_df, 
                                                           func_gex_df, gene, test_outdir)
            sorted_features_df.columns = ["Peak", "Importance"]
        
        npeaks = len(sorted_features_df)
        return npeaks, sorted_features_df, test_outdir
    
    def _LR_init_peakranker(self, model, func_peaks_df, func_gex_df, 
                            gene_outdir, test, gene):
        """Linear Regression peak ranker"""
        # Use organized directory structure
        feature_rankings_dir = os.path.join(gene_outdir, "feature_rankings")
        os.makedirs(feature_rankings_dir, exist_ok=True)
        
        if test == "perm_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "lr_permranker")
            os.makedirs(test_outdir, exist_ok=True)
            baseline = permutation_importance(model, func_peaks_df, func_gex_df)
            sorted_features_df = perm_ranker(baseline, gene, func_peaks_df, test_outdir)
        elif test == "dropcol_ranker":
            test_outdir = os.path.join(feature_rankings_dir, "lr_dropcolranker")
            os.makedirs(test_outdir, exist_ok=True)
            sorted_features_df = LR_dropcolumn_importance(func_peaks_df, func_gex_df, 
                                                         gene, test_outdir)
            sorted_features_df.columns = ["Peak", "Importance"]
        
        npeaks = len(sorted_features_df)
        return npeaks, sorted_features_df, test_outdir
    
    def _avg_feature_importances(self, peak_importance_dict):
        """Calculate average feature importances"""
        average_importance_dict = {}
        for peak, importance_values in peak_importance_dict.items():
            average_importance = sum(importance_values) / len(importance_values)
            average_importance_dict[peak] = average_importance
        
        average_importance_df = pd.DataFrame(
            list(average_importance_dict.items()), 
            columns=["Peak", "Average Importance"]
        )
        average_importance_df = average_importance_df.sort_values(
            by="Average Importance", 
            ascending=False
        ).reset_index(drop=True)
        
        return average_importance_df