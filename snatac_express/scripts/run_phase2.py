#!/usr/bin/env python3
"""
Phase 2 implementation for snATAC-Express
Includes aggregation of Phase 1 results and re-running with aggregated features
"""

import os
import yaml
import argparse
import logging
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import workflow modules
from .data_preprocessing import (
    get_pseudobulk, load_peak_input, subset_peaks,
    load_gex_input, subset_gex, make_all_pseudobulk
)
from .model_builder import ModelBuilder


def aggregate_peak_importances(phase1_results_dir, output_dir, include_lr=False):
    """
    Aggregate peak importance scores across all genes from Phase 1
    
    Args:
        phase1_results_dir: Directory containing Phase 1 results
        output_dir: Where to save aggregated results
        include_lr: Whether to include Linear Regression in aggregation
    
    Returns:
        Dictionary mapping genes to their aggregated peak importance DataFrames
    """
    logger = logging.getLogger(__name__)
    logger.info("Aggregating peak importances across all genes")
    
    # Get all gene directories
    gene_dirs = [d for d in os.listdir(phase1_results_dir) 
                 if os.path.isdir(os.path.join(phase1_results_dir, d))]
    
    aggregated_results = {}
    
    for gene in gene_dirs:
        gene_path = os.path.join(phase1_results_dir, gene)
        logger.info(f"  Processing {gene}")
        
        # Initialize output DataFrame
        alpha_summary = pd.DataFrame()
        
        # Define test methods to aggregate
        if include_lr:
            test_list = [
                "rf_dropcolranker", "rf_permranker", "rf_ranker",
                "lr_dropcolranker", "lr_permranker",
                "xgb_dropcolranker", "xgb_permranker", "xgb_ranker",
                "lgbm_dropcolranker", "lgbm_permranker", "lgbm_ranker"
            ]
        else:
            test_list = [
                "rf_dropcolranker", "rf_permranker", "rf_ranker",
                "xgb_dropcolranker", "xgb_permranker", "xgb_ranker",
                "lgbm_dropcolranker", "lgbm_permranker", "lgbm_ranker"
            ]
        
        # Collect importance scores from each method
        for test in test_list:
            test_dir = os.path.join(gene_path, test)
            if not os.path.exists(test_dir):
                continue
                
            # Find importance files (use the one with most peaks - typically all peaks)
            importance_files = glob.glob(os.path.join(test_dir, "*importance.csv"))
            if not importance_files:
                continue
                
            # Sort by file size to get the file with all peaks
            importance_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
            selected_file = importance_files[0]
            
            # Read importance scores
            peak_rank_df = pd.read_csv(selected_file)
            
            # Fix peak coordinates for LightGBM (they use underscores)
            if 'lgbm' in test:
                peak_rank_df['Peak'] = peak_rank_df['Peak'].apply(
                    lambda x: x.replace('_', ':').replace(':', '-', 1)
                )
            
            # Calculate z-scores for importance values
            mu = peak_rank_df['Importance'].mean()
            sigma = peak_rank_df['Importance'].std()
            
            temp_summary = pd.DataFrame()
            temp_summary['Peaks'] = peak_rank_df['Peak']
            temp_summary[f'{test}_Zscore'] = (peak_rank_df['Importance'] - mu) / sigma
            
            # Merge with main summary
            if alpha_summary.empty:
                alpha_summary = temp_summary
            else:
                alpha_summary = pd.merge(alpha_summary, temp_summary, 
                                       on='Peaks', how='outer')
        
        if not alpha_summary.empty:
            # Calculate average z-score across all methods
            zscore_columns = [col for col in alpha_summary.columns if 'Zscore' in col]
            alpha_summary['Average_Zscore'] = alpha_summary[zscore_columns].mean(axis=1)
            
            # Sort by average z-score
            alpha_summary = alpha_summary.sort_values(by='Average_Zscore', 
                                                    ascending=False).reset_index(drop=True)
            
            # Save gene-specific aggregated results
            gene_output_dir = os.path.join(output_dir, gene)
            os.makedirs(gene_output_dir, exist_ok=True)
            
            filename = os.path.join(gene_output_dir, 
                                  f"aggregated_peak_importances_{'inclLR' if include_lr else 'exclLR'}.csv")
            alpha_summary.to_csv(filename, index=True)
            
            aggregated_results[gene] = alpha_summary
            logger.info(f"    Aggregated {len(alpha_summary)} peaks")
    
    return aggregated_results


def get_top_aggregated_peaks(aggregated_results, top_percentage=0.95):
    """
    Get top peaks based on aggregated importance across all genes
    
    Args:
        aggregated_results: Dictionary of gene -> aggregated importance DataFrames
        top_percentage: Percentage of cumulative importance to keep
    
    Returns:
        Dictionary mapping genes to their selected top peaks
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Selecting top {top_percentage*100}% peaks for each gene")
    
    selected_peaks = {}
    
    for gene, importance_df in aggregated_results.items():
        if importance_df.empty:
            continue
            
        # Adjust z-scores to be positive
        min_zscore = importance_df['Average_Zscore'].min()
        importance_df['Adjusted_Zscore'] = importance_df['Average_Zscore'] + abs(min_zscore)
        
        # Sort by adjusted z-score
        df_sorted = importance_df.sort_values(by='Adjusted_Zscore', ascending=False)
        
        # Find cumulative top percentage
        total_sum = df_sorted['Adjusted_Zscore'].sum()
        threshold = top_percentage * total_sum
        
        # Select peaks up to threshold
        cumsum = df_sorted['Adjusted_Zscore'].cumsum()
        top_peaks = df_sorted[cumsum <= threshold]
        
        selected_peaks[gene] = top_peaks['Peaks'].tolist()
        logger.info(f"  {gene}: selected {len(top_peaks)} / {len(importance_df)} peaks")
    
    return selected_peaks


def run_phase2_for_gene(gene, window, selected_peaks, config, peak_df, gex_df, pb_keep):
    """
    Run Phase 2 analysis for a single gene using aggregated peaks
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running Phase 2 for gene {gene}")
    
    # Create output directory for Phase 2
    gene_outdir = os.path.join(config['output_dir'], 'phase2_results', gene)
    os.makedirs(gene_outdir, exist_ok=True)
    
    # Extract gene data
    gene_peaks = subset_peaks(peak_df, window)
    gene_exp = subset_gex(gex_df, gene)
    
    # Create pseudobulk
    pb_peak_df, gex_peak_df = make_all_pseudobulk(
        gene_peaks, gene_exp, gene, pb_keep, gene_outdir, peak_df, gex_df
    )
    
    # Filter to only use aggregated selected peaks
    if gene in selected_peaks:
        peak_list = selected_peaks[gene]
        pb_peak_df = pb_peak_df[pb_peak_df.index.isin(peak_list)]
        logger.info(f"  Using {len(pb_peak_df)} aggregated peaks")
    else:
        logger.warning(f"  No aggregated peaks found for {gene}, skipping")
        return None
    
    # Check if we have enough peaks
    min_peaks = config.get('advanced', {}).get('min_peaks_per_gene', 3)
    if len(pb_peak_df) < min_peaks:
        logger.warning(f"  Gene {gene} has only {len(pb_peak_df)} aggregated peaks, skipping")
        return None
    
    # Prepare data for modeling
    peaks_array = pb_peak_df.values.T
    gex_array = gex_peak_df.values.T
    
    X = pd.DataFrame(peaks_array, columns=pb_peak_df.index, index=pb_peak_df.columns.tolist())
    y = pd.DataFrame(gex_array, columns=gex_peak_df.index, index=pb_peak_df.columns.tolist())
    
    # Initialize model builder
    model_builder = ModelBuilder(config)
    
    # Run all models - Phase 2 uses simpler approach (no 95% selection within phase)
    results = {}
    models_to_run = [name for name, cfg in config['models'].items() if cfg.get('enabled', True)]
    
    for model_name in models_to_run:
        logger.info(f"  Running {model_name}")
        
        # Determine which ranking methods to use
        if model_name == 'linear_regression':
            methods = ['perm_ranker', 'dropcol_ranker']
        elif model_name == 'random_forest':
            methods = ['rf_ranker', 'perm_ranker', 'dropcol_ranker']
        elif model_name == 'xgboost':
            methods = ['xgb_ranker', 'perm_ranker', 'dropcol_ranker']
        elif model_name == 'lightgbm':
            methods = ['lgbm_ranker', 'perm_ranker', 'dropcol_ranker']
        
        # For Phase 2, we typically only need one method per model
        method = methods[0]  # Use primary ranking method
        
        try:
            # Build model with aggregated peaks only
            result = model_builder.build_and_evaluate_model(
                model_name, X, y, gene, gene_outdir, method
            )
            
            # Phase 2 only returns results for aggregated peaks
            results[f"{model_name}_{method}"] = {
                'n_peaks': len(X.columns),
                'r2': result['all_peaks']['r2']  # Only 'all_peaks' in Phase 2
            }
            
            logger.info(f"    {method}: R² = {result['all_peaks']['r2']:.4f} ({len(X.columns)} peaks)")
            
        except Exception as e:
            logger.error(f"    Error with {method}: {str(e)}")
    
    return {
        'gene': gene,
        'n_peaks_aggregated': len(X.columns),
        'results': results
    }


def run_phase2_workflow(config, phase1_results_dir):
    """
    Run complete Phase 2 workflow
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting Phase 2 workflow")
    
    # Create Phase 2 output directories
    phase2_dir = os.path.join(config['output_dir'], 'phase2_results')
    aggregated_dir = os.path.join(config['output_dir'], 'aggregated_results')
    os.makedirs(phase2_dir, exist_ok=True)
    os.makedirs(aggregated_dir, exist_ok=True)
    
    # Step 1: Aggregate Phase 1 results
    logger.info("Step 1: Aggregating Phase 1 results")
    include_lr = config.get('phase1', {}).get('aggregation', {}).get('include_linear_regression', False)
    aggregated_results = aggregate_peak_importances(
        phase1_results_dir, aggregated_dir, include_lr=include_lr
    )
    
    # Step 2: Select top peaks for each gene
    logger.info("Step 2: Selecting top peaks")
    top_percentage = config.get('phase2', {}).get('top_features_percentage', 0.95)
    selected_peaks = get_top_aggregated_peaks(aggregated_results, top_percentage)
    
    # Save selected peaks summary
    peaks_summary = pd.DataFrame([
        {'gene': gene, 'n_peaks_selected': len(peaks)}
        for gene, peaks in selected_peaks.items()
    ])
    peaks_summary.to_csv(os.path.join(aggregated_dir, 'selected_peaks_summary.csv'), index=False)
    
    # Step 3: Load data for Phase 2
    logger.info("Step 3: Loading data for Phase 2")
    
    # Load gene list
    gene_list_path = os.path.join('example_data', 'input_data', config['input_data']['gene_list'])
    gene_df = pd.read_csv(gene_list_path, sep='\t')
    gene_df.columns = ['gene', 'window']
    
    # Filter to genes with aggregated results
    gene_df = gene_df[gene_df['gene'].isin(selected_peaks.keys())]
    
    logger.info("Loading ATAC peaks...")
    peak_df = load_peak_input(
        config['input_data']['sparse_peak_matrix'],
        input_dir='example_data/input_data'
    )
    
    logger.info("Loading gene expression...")
    gex_df = load_gex_input(
        config['input_data']['sparse_gex_matrix'],
        input_dir='example_data/input_data'
    )
    
    # Get pseudobulk groups
    pb_keep = get_pseudobulk(
        config['phase1']['pseudobulk']['replicate'],
        group_coverages_csv='group_coverages.csv',
        input_dir='example_data/input_data'
    )
    
    # Step 4: Run Phase 2 analysis for each gene
    logger.info(f"Step 4: Running Phase 2 for {len(gene_df)} genes")
    
    phase2_results = []
    for idx, row in gene_df.iterrows():
        gene = row['gene']
        window = row['window']
        
        result = run_phase2_for_gene(
            gene, window, selected_peaks, config, peak_df, gex_df, pb_keep
        )
        
        if result:
            phase2_results.append(result)
    
    # Step 5: Summarize Phase 2 results
    logger.info("Step 5: Summarizing Phase 2 results")
    summarize_phase2_results(config, phase2_results)
    
    logger.info(f"Phase 2 completed. Processed {len(phase2_results)} genes.")
    
    return phase2_results


def summarize_phase2_results(config, phase2_results):
    """
    Create summary of Phase 2 results
    """
    logger = logging.getLogger(__name__)
    
    summary_data = []
    
    for result in phase2_results:
        gene = result['gene']
        n_peaks = result['n_peaks_aggregated']
        
        for method, metrics in result['results'].items():
            summary_data.append({
                'Gene': gene,
                'Method': method,
                'nPeaks': n_peaks,
                'Phase': 'Phase2_Aggregated',
                'CV_R2': metrics['r2']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(config['output_dir'], 'phase2_cv_summary.txt')
        summary_df.to_csv(summary_path, sep='\t', index=False)
        
        logger.info(f"Saved Phase 2 summary to {summary_path}")
        
        # Print summary statistics
        logger.info("\nPhase 2 Summary Statistics:")
        logger.info(f"Total genes analyzed: {summary_df['Gene'].nunique()}")
        logger.info(f"Average R²: {summary_df['CV_R2'].mean():.4f}")
        logger.info(f"Median R²: {summary_df['CV_R2'].median():.4f}")
        
        # Compare with Phase 1 if available
        phase1_summary_path = os.path.join(config['output_dir'], 'cv_summary.txt')
        if os.path.exists(phase1_summary_path):
            phase1_df = pd.read_csv(phase1_summary_path, sep='\t')
            
            # Compare performance
            phase1_select = phase1_df[phase1_df['PeakCat'] == 'Select_Peaks']
            
            logger.info("\nPhase 1 vs Phase 2 Comparison:")
            logger.info(f"Phase 1 (95% peaks) avg R²: {phase1_select['CV_R2'].mean():.4f}")
            logger.info(f"Phase 2 (aggregated) avg R²: {summary_df['CV_R2'].mean():.4f}")


def setup_logging(output_dir):
    """Set up logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(
        output_dir, 
        f"snATAC_Express_Phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # This allows the script to be run independently if needed
    parser = argparse.ArgumentParser(description='Run snATAC-Express Phase 2 workflow')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--phase1-dir', type=str, default=None,
                        help='Directory containing Phase 1 results')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine Phase 1 results directory
    if args.phase1_dir:
        phase1_results_dir = args.phase1_dir
    else:
        phase1_results_dir = os.path.join(config['output_dir'], 'results')
    
    if not os.path.exists(phase1_results_dir):
        raise ValueError(f"Phase 1 results directory not found: {phase1_results_dir}")
    
    # Setup logging
    logger = setup_logging(config['output_dir'])
    logger.info("Starting snATAC-Express Phase 2 (Aggregation + Refined Modeling)")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Phase 1 results: {phase1_results_dir}")
    
    try:
        # Run Phase 2 workflow
        results = run_phase2_workflow(config, phase1_results_dir)
        
        logger.info("\nPhase 2 workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running Phase 2 workflow: {str(e)}")
        raise