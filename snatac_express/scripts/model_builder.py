#!/usr/bin/env python3
"""
Unified model builder for snATAC-Express
Handles both Phase 1 (full modeling) and Phase 2 (refined modeling)
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
    """Unified model builder for both phases of snATAC-Express"""
    
    def __init__(self, config, phase="phase1"):
        """
        Initialize model builder with configuration
        
        Args:
            config: Configuration dictionary
            phase: "phase1" or "phase2"
        """
        self.config = config
        self.phase = phase
        self.phase_config = config[phase]
        self.model_config = config['models']
        self.advanced_config = config.get('advanced', {})
        
    def get_cv_splitter(self):
        """Get cross-validation splitter based on configuration"""
        cv_config = self.phase_config['cross_validation']
        n_folds = cv_config.get('n_folds', 5)
        
        if cv_config['type'] == 'stratified':
            return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
        else:
            return KFold(n_splits=n_folds, shuffle=True, random_state=0)
    
    def get_param_grid(self, model_name, n_features):
        """Get parameter grid for grid search"""
        model_params = self.model_config[model_name]['param_grid']
        
        # Special handling for random forest max_features
        if model_name == 'random_forest' and 'max_features' not in model_params:
            model_params = model_params.copy()
            model_params['max_features'] = [round(math.sqrt(n_features))]
            
        return model_params
    
    def grid_search(self, model_class, param_grid, X, y):
        """Perform grid search for hyperparameter tuning"""
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
        
        if len(param_grid) > 100:  # Use RandomizedSearch for large grids
            gs_model = RandomizedSearchCV(
                estimator=model_class(),
                param_distributions=param_grid,
                cv=inner_cv,
                n_jobs=self.config.get('n_jobs', -1),
                n_iter=50
            )
        else:
            gs_model = GridSearchCV(
                estimator=model_class(),
                param_grid=param_grid,
                cv=inner_cv,
                n_jobs=self.config.get('n_jobs', -1)
            )
            
        gs_model.fit(X, y.values.ravel())
        return gs_model.best_params_
    
    def cross_validate_model(self, model, X, y, gene, gene_outdir, test_method, model_name):
        """Perform k-fold cross-validation with feature ranking"""
        cv_config = self.phase_config['cross_validation']
        n_columns = cv_config.get('n_columns', 3)
        n_folds = cv_config.get('n_folds', 5)
        
        cv_splitter = self.get_cv_splitter()
        cv_columns = [cv_splitter for _ in range(n_columns)]
        
        r2_fold_scores = []
        peak_importance_dict = {}
        
        for column_idx, skf in enumerate(cv_columns):
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y[gene] if hasattr(y, '__getitem__') else y)):
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model.predict(X_test)
                score = model.score(X_test, y_test)
                r2_fold_scores.append(score)
                
                # Get feature importance
                importance_df = self._get_feature_importance(
                    model, X_train, y_train, gene, gene_outdir, test_method, model_name
                )
                
                # Save predictions
                self._save_predictions(
                    y_test, y_pred, gene_outdir, test_method, model_name,
                    column_idx, fold, self.phase
                )
                
                # Aggregate importance scores
                self._aggregate_importance(importance_df, peak_importance_dict)
                
        return r2_fold_scores, peak_importance_dict
    
    def _get_feature_importance(self, model, X, y, gene, gene_outdir, test_method, model_name):
        """Get feature importance based on the specified method"""
        test_outdir = os.path.join(gene_outdir, f"{model_name}_{test_method}")
        os.makedirs(test_outdir, exist_ok=True)
        
        if model_name == "linear_regression":
            if test_method == "perm_ranker":
                baseline = permutation_importance(model, X, y)
                return perm_ranker(baseline, gene, X, test_outdir)
            elif test_method == "dropcol_ranker":
                return LR_dropcolumn_importance(X, y, gene, test_outdir)
                
        elif model_name == "random_forest":
            if test_method == "rf_ranker":
                return rf_ranker(model, gene, X, test_outdir)
            elif test_method == "perm_ranker":
                baseline = permutation_importance(model, X, y)
                return perm_ranker(baseline, gene, X, test_outdir)
            elif test_method == "dropcol_ranker":
                best_params = model.get_params()
                return RF_dropcolumn_importance(best_params, X, y, gene, test_outdir)
                
        elif model_name == "xgboost":
            if test_method == "xgb_ranker":
                return xgb_ranker(model, gene, X, test_outdir)
            elif test_method == "perm_ranker":
                baseline = permutation_importance(model, X, y)
                return perm_ranker(baseline, gene, X, test_outdir)
            elif test_method == "dropcol_ranker":
                best_params = model.get_params()
                return XGB_dropcolumn_importance(best_params, X, y, gene, test_outdir)
                
        elif model_name == "lightgbm":
            if test_method == "lgbm_ranker":
                return lgbm_ranker(model, gene, X, test_outdir)
            elif test_method == "perm_ranker":
                baseline = permutation_importance(model, X, y)
                return perm_ranker(baseline, gene, X, test_outdir)
            elif test_method == "dropcol_ranker":
                best_params = model.get_params()
                return LGBM_dropcolumn_importance(best_params, X, y, gene, test_outdir)
    
    def _save_predictions(self, y_test, y_pred, gene_outdir, test_method, model_name,
                         column_idx, fold, phase):
        """Save prediction results"""
        pred_dir = os.path.join(
            gene_outdir, 
            f"{model_name}_{test_method}",
            f"cross_validations_{phase}"
        )
        os.makedirs(pred_dir, exist_ok=True)
        
        y_test = y_test.copy()
        y_test["Predicted"] = y_pred.tolist()
        outname = os.path.join(pred_dir, f"Column_{column_idx}_Fold_{fold}.csv")
        y_test.to_csv(outname)
    
    def _aggregate_importance(self, importance_df, peak_importance_dict):
        """Aggregate importance scores across folds"""
        if "Peak" in importance_df.columns:
            grouped = importance_df.groupby("Peak")["Importance"].apply(list).to_dict()
        else:
            # Handle different column names
            peak_col = importance_df.columns[0]
            imp_col = importance_df.columns[1]
            grouped = importance_df.groupby(peak_col)[imp_col].apply(list).to_dict()
            
        for peak, importance_values in grouped.items():
            if peak in peak_importance_dict:
                peak_importance_dict[peak].extend(importance_values)
            else:
                peak_importance_dict[peak] = importance_values
    
    def average_feature_importances(self, peak_importance_dict):
        """Calculate average importance for each peak"""
        average_importance = {
            peak: sum(values) / len(values)
            for peak, values in peak_importance_dict.items()
        }
        
        df = pd.DataFrame(
            list(average_importance.items()),
            columns=["Peak", "Average Importance"]
        )
        return df.sort_values(by="Average Importance", ascending=False).reset_index(drop=True)
    
    def select_top_features(self, importance_df, threshold=0.95):
        """Select top features based on cumulative importance"""
        total = importance_df["Average Importance"].sum()
        thresh = total * threshold
        current_sum = 0
        rows_to_keep = []
        
        for index, row in importance_df.iterrows():
            current_sum += row["Average Importance"]
            rows_to_keep.append(index)
            if current_sum > thresh:
                break
                
        return importance_df.loc[rows_to_keep]
    
    def build_and_evaluate_model(self, model_name, X, y, gene, gene_outdir, test_method):
        """Main method to build and evaluate a model"""
        
        # Get model class and parameters
        model_classes = {
            'linear_regression': LinearRegression,
            'random_forest': RandomForestRegressor,
            'xgboost': xgb.XGBRegressor,
            'lightgbm': lgbm.LGBMRegressor
        }
        
        model_class = model_classes[model_name]
        
        # Prepare data (handle LightGBM column names)
        if model_name == 'lightgbm':
            X = X.copy()
            X.columns = X.columns.str.replace(':', '_').str.replace('-', '_')
        
        # Get best parameters (skip for linear regression)
        if model_name != 'linear_regression':
            param_grid = self.get_param_grid(model_name, len(X.columns))
            
            # Add default parameters for LightGBM to prevent hanging
            if model_name == 'lightgbm':
                # Create a custom estimator with default parameters
                base_estimator = lgbm.LGBMRegressor(
                    random_state=42,
                    verbose=-1,  # Suppress verbose output
                    n_jobs=1,    # Use single thread to avoid conflicts
                    force_col_wise=True  # Force column-wise for better compatibility
                )
                
                if len(param_grid) > 100:  # Use RandomizedSearch for large grids
                    gs_model = RandomizedSearchCV(
                        estimator=base_estimator,
                        param_distributions=param_grid,
                        cv=KFold(n_splits=5, shuffle=True, random_state=0),
                        n_jobs=1,  # Use single job to avoid conflicts
                        n_iter=50,
                        random_state=42
                    )
                else:
                    gs_model = GridSearchCV(
                        estimator=base_estimator,
                        param_grid=param_grid,
                        cv=KFold(n_splits=5, shuffle=True, random_state=0),
                        n_jobs=1  # Use single job to avoid conflicts
                    )
            else:
                # Use original grid search for other models
                best_params = self.grid_search(model_class, param_grid, X, y)
                model = model_class(**best_params)
                best_params = best_params
        else:
            model = model_class()
            best_params = {}
        
        # For LightGBM, get best params from grid search
        if model_name == 'lightgbm' and model_name != 'linear_regression':
            gs_model.fit(X, y.values.ravel())
            best_params = gs_model.best_params_
            # Create model with best params plus defaults
            model = lgbm.LGBMRegressor(
                **best_params,
                random_state=42,
                verbose=-1,
                n_jobs=1,
                force_col_wise=True
            )
        
        # Cross-validation
        r2_scores, importance_dict = self.cross_validate_model(
            model, X, y, gene, gene_outdir, test_method, model_name
        )
        
        # Train final model and save
        model.fit(X, y)
        model_path = os.path.join(
            gene_outdir,
            f"{model_name}_{test_method}",
            f"trained_model_{self.phase}.pkl"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        
        # Calculate average performance
        avg_r2 = np.mean(r2_scores)
        n_peaks = len(X.columns)
        
        # Save results
        results_df = pd.DataFrame({
            "nPeaks": [n_peaks],
            "R2": [avg_r2]
        })
        results_path = os.path.join(
            gene_outdir,
            f"{gene}_{model_name.upper()}_{test_method}_results.txt"
        )
        results_df.to_csv(results_path, index=False)
        
        # Return average importance
        avg_importance = self.average_feature_importances(importance_dict)
        
        return {
            'model': model,
            'avg_r2': avg_r2,
            'n_peaks': n_peaks,
            'importance': avg_importance,
            'best_params': best_params
        }