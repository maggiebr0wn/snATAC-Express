#!/usr/sbin/anaconda

"""
Model Training Module for snATAC-Express

This module runs multiple predictive models (Random Forest, Linear Regression,
XGBoost, and LightGBM) on snATAC-seq and snRNA-seq data to predict gene
expression levels. It supports various feature ranking methods and includes
data preprocessing steps.
"""

from typing import Dict, List, Tuple, Union, Optional
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from config import (
    CV_FOLDS,
    SPLITS_PER_FOLD,
    get_gene_output_dir,
    get_method_output_dir
)

def parse_my_args() -> Dict[str, str]:
    """
    Parse command line arguments.

    Returns:
        Dictionary containing parsed arguments:
        - gene_list: Path to gene list file
        - gene_name: Name of gene to process
        - gex_matrix: Path to gene expression matrix
        - peak_matrix: Path to peak accessibility matrix
        - pseudobulk_replicate: Pseudobulk replicate version (1 or 2)
        - peak_filter: Minimum percentage of samples a peak must be present in
        - output_dir: Output directory path
    """
    parser = argparse.ArgumentParser(description='Run multiple predictive models')
    parser.add_argument('--gene_list', required=True, help='Path to gene list file')
    parser.add_argument('--gene_name', required=True, help='Name of gene to process')
    parser.add_argument('--gex_matrix', required=True, help='Path to gene expression matrix')
    parser.add_argument('--peak_matrix', required=True, help='Path to peak accessibility matrix')
    parser.add_argument('--pseudobulk_replicate', required=True, choices=['1', '2'],
                      help='Pseudobulk replicate version')
    parser.add_argument('--peak_filter', type=float, default=0.1,
                      help='Minimum percentage of samples a peak must be present in')
    parser.add_argument('--output_dir', required=True, help='Output directory path')
    args = parser.parse_args()
    return vars(args)

def build_models(
    gene: str,
    peak_matrix: pd.DataFrame,
    gex_matrix: pd.DataFrame,
    peak_filter: float,
    output_dir: str
) -> None:
    """
    Build and evaluate multiple predictive models for a given gene.

    Args:
        gene: Name of the target gene
        peak_matrix: DataFrame containing peak accessibility data
        gex_matrix: DataFrame containing gene expression data
        peak_filter: Minimum percentage of samples a peak must be present in
        output_dir: Output directory path
    """
    # Extract gene-specific data
    gene_peaks = peak_matrix[peak_matrix.index.str.contains(gene)]
    gene_exp = gex_matrix[gex_matrix.index == gene]
    
    # Filter peaks based on presence threshold
    peak_presence = (gene_peaks > 0).mean(axis=1)
    filtered_peaks = gene_peaks[peak_presence >= peak_filter]
    
    # Prepare data for modeling
    X = filtered_peaks.T
    y = gene_exp.iloc[0]
    
    # Initialize models
    models = {
        'rf': RandomForestRegressor(n_estimators=100, random_state=42),
        'lr': LinearRegression(),
        'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'lgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y,
            cv=CV_FOLDS,
            n_jobs=SPLITS_PER_FOLD,
            scoring='r2'
        )
        
        # Calculate mean and std of R² scores
        mean_r2 = cv_scores.mean()
        std_r2 = cv_scores.std()
        
        # Store results
        output_dir = get_method_output_dir(gene, name)
        results[name] = {
            'mean_r2': mean_r2,
            'std_r2': std_r2
        }
        
        # Save results
        results_df = pd.DataFrame({
            'model': [name],
            'mean_r2': [mean_r2],
            'std_r2': [std_r2]
        })
        results_df.to_csv(output_dir / 'cv_results.csv', index=False)
        
        # Train final model and save predictions
        model.fit(X, y)
        predictions = model.predict(X)
        pred_df = pd.DataFrame({
            'true': y,
            'predicted': predictions
        })
        pred_df.to_csv(output_dir / 'predictions.csv', index=False)

def main():
    """
    Main function to run the model training pipeline.
    """
    # Parse arguments
    args = parse_my_args()
    
    # Load data
    peak_matrix = pd.read_csv(args['peak_matrix'], index_col=0)
    gex_matrix = pd.read_csv(args['gex_matrix'], index_col=0)
    
    # Load gene list
    gene_list = pd.read_csv(args['gene_list'], header=None)[0].tolist()
    
    # Process each gene
    for gene in gene_list:
        if gene == args['gene_name']:
            build_models(
                gene,
                peak_matrix,
                gex_matrix,
                args['peak_filter'],
                args['output_dir']
            )

if __name__ == '__main__':
    main()
