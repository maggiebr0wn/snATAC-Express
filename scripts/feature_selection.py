#!/usr/sbin/anaconda

"""
Feature Selection Module for snATAC-Express

This module implements various feature selection methods for identifying important
regulatory regions in single-cell ATAC-seq data that predict gene expression levels.
"""

from typing import Dict, List, Tuple, Union, Optional
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
from config import (
    FEATURE_IMPORTANCE_THRESHOLD,
    get_gene_output_dir,
    get_method_output_dir
)

def rf_ranker(
    X: pd.DataFrame,
    y: pd.Series,
    gene: str,
    method: str = "rf"
) -> pd.DataFrame:
    """
    Rank features using Random Forest importance scores.

    Args:
        X: Feature matrix (peak accessibility data)
        y: Target vector (gene expression data)
        gene: Name of the target gene
        method: Method identifier for output files

    Returns:
        DataFrame containing feature importance scores
    """
    # Train Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance scores
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    # Save results
    output_dir = get_method_output_dir(gene, method)
    importance.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    return importance

def xgb_ranker(
    X: pd.DataFrame,
    y: pd.Series,
    gene: str,
    method: str = "xgb"
) -> pd.DataFrame:
    """
    Rank features using XGBoost importance scores.

    Args:
        X: Feature matrix (peak accessibility data)
        y: Target vector (gene expression data)
        gene: Name of the target gene
        method: Method identifier for output files

    Returns:
        DataFrame containing feature importance scores
    """
    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X, y)
    
    # Get feature importance scores
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    # Save results
    output_dir = get_method_output_dir(gene, method)
    importance.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    return importance

def lgbm_ranker(
    X: pd.DataFrame,
    y: pd.Series,
    gene: str,
    method: str = "lgbm"
) -> pd.DataFrame:
    """
    Rank features using LightGBM importance scores.

    Args:
        X: Feature matrix (peak accessibility data)
        y: Target vector (gene expression data)
        gene: Name of the target gene
        method: Method identifier for output files

    Returns:
        DataFrame containing feature importance scores
    """
    # Train LightGBM model
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    lgb_model.fit(X, y)
    
    # Get feature importance scores
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': lgb_model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    # Save results
    output_dir = get_method_output_dir(gene, method)
    importance.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    return importance

def perm_ranker(
    X: pd.DataFrame,
    y: pd.Series,
    gene: str,
    method: str = "perm"
) -> pd.DataFrame:
    """
    Rank features using permutation importance.

    Args:
        X: Feature matrix (peak accessibility data)
        y: Target vector (gene expression data)
        gene: Name of the target gene
        method: Method identifier for output files

    Returns:
        DataFrame containing feature importance scores
    """
    # Train base model (Random Forest)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Calculate permutation importance
    result = permutation_importance(
        rf, X, y,
        n_repeats=10,
        random_state=42
    )
    
    # Get feature importance scores
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': result.importances_mean
    })
    importance = importance.sort_values('importance', ascending=False)
    
    # Save results
    output_dir = get_method_output_dir(gene, method)
    importance.to_csv(output_dir / 'feature_importance.csv', index=False)
    
    return importance

def select_features(
    importance_df: pd.DataFrame,
    threshold: float = FEATURE_IMPORTANCE_THRESHOLD
) -> List[str]:
    """
    Select features based on cumulative importance threshold.

    Args:
        importance_df: DataFrame containing feature importance scores
        threshold: Cumulative importance threshold (default from config)

    Returns:
        List of selected feature names
    """
    # Calculate cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
    
    # Select features above threshold
    selected_features = importance_df[
        importance_df['cumulative_importance'] <= threshold
    ]['feature'].tolist()
    
    return selected_features

def main():
    """
    Main function to run feature selection methods.
    """
    parser = argparse.ArgumentParser(description='Feature selection for snATAC-Express')
    parser.add_argument('--gene', required=True, help='Target gene name')
    parser.add_argument('--method', required=True, choices=['rf', 'xgb', 'lgbm', 'perm'],
                      help='Feature selection method')
    parser.add_argument('--peaks', required=True, help='Path to peak accessibility matrix')
    parser.add_argument('--gex', required=True, help='Path to gene expression matrix')
    args = parser.parse_args()
    
    # Load data
    X = pd.read_csv(args.peaks, index_col=0)
    y = pd.read_csv(args.gex, index_col=0).iloc[:, 0]
    
    # Run selected feature selection method
    if args.method == 'rf':
        importance_df = rf_ranker(X, y, args.gene)
    elif args.method == 'xgb':
        importance_df = xgb_ranker(X, y, args.gene)
    elif args.method == 'lgbm':
        importance_df = lgbm_ranker(X, y, args.gene)
    elif args.method == 'perm':
        importance_df = perm_ranker(X, y, args.gene)
    
    # Select features
    selected_features = select_features(importance_df)
    
    # Save selected features
    output_dir = get_method_output_dir(args.gene, args.method)
    with open(output_dir / 'selected_features.txt', 'w') as f:
        f.write('\n'.join(selected_features))

if __name__ == '__main__':
    main()







