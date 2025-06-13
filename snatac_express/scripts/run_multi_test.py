#!/usr/bin/env python3
"""
Main workflow runner for snATAC-Express
Executes both Phase 1 and Phase 2 of the analysis pipeline
"""

import os
import yaml
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import workflow modules - use relative imports since we're in a package
from .data_preprocessing import (
    get_pseudobulk, load_peak_input, subset_peaks,
    load_gex_input, subset_gex, make_all_pseudobulk
)
from .model_builder import ModelBuilder
from .feature_selection import feature_selector


def setup_logging(output_dir):
    """Set up logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(
        output_dir, 
        f"snATAC_Express_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def create_output_dirs(config):
    """Create necessary output directories"""
    dirs = [
        config['output_dir'],
        os.path.join(config['output_dir'], 'phase1'),
        os.path.join(config['output_dir'], 'phase2'),
        os.path.join(config['output_dir'], 'aggregated')
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def aggregate_peak_ranks(gene_dir, config):
    """Aggregate peak importance ranks across all models and methods - preserving raw values"""
    logger = logging.getLogger(__name__)
    
    # Find all importance files
    importance_files = []
    for method_dir in os.listdir(gene_dir):
        method_path = os.path.join(gene_dir, method_dir)
        if os.path.isdir(method_path):
            for file in os.listdir(method_path):
                if 'importance.csv' in file:
                    importance_files.append(os.path.join(method_path, file))
    
    if not importance_files:
        logger.warning(f"No importance files found in {gene_dir}")
        return None
    
    # Read and aggregate importance scores
    all_importance = {}
    for file in importance_files:
        df = pd.read_csv(file)
        if 'Peak' in df.columns and 'Importance' in df.columns:
            # For drop column methods, preserve raw MSE differences
            if 'dropcolumn' in file:
                # These are raw MSE differences - don't normalize
                for idx, row in df.iterrows():
                    peak = row['Peak']
                    if peak not in all_importance:
                        all_importance[peak] = []
                    all_importance[peak].append(row['Importance'])
            else:
                # For other methods, convert to z-scores for aggregation
                if len(df) > 1:
                    z_scores = (df['Importance'] - df['Importance'].mean()) / df['Importance'].std()
                else:
                    z_scores = pd.Series([0])
                
                for idx, peak in enumerate(df['Peak']):
                    if peak not in all_importance:
                        all_importance[peak] = []
                    all_importance[peak].append(z_scores.iloc[idx])
    
    # Calculate average scores
    avg_importance = {
        peak: np.mean(scores) for peak, scores in all_importance.items()
    }
    
    # Create aggregated dataframe
    agg_df = pd.DataFrame([
        {'Peak': peak, 'Average_Zscore': score}
        for peak, score in avg_importance.items()
    ])
    
    # Save aggregated results
    agg_df = agg_df.sort_values('Average_Zscore', ascending=False)
    agg_path = os.path.join(gene_dir, 'aggregated_peak_importances.csv')
    agg_df.to_csv(agg_path, index=False)
    
    return agg_df


def run_phase1_for_gene(gene, window, config, peak_df, gex_df, pb_keep):
    """Run Phase 1 analysis for a single gene"""
    logger = logging.getLogger(__name__)
    logger.info(f"Phase 1: Processing gene {gene}")
    
    # Create output directory
    gene_outdir = os.path.join(config['output_dir'], 'phase1', gene)
    os.makedirs(gene_outdir, exist_ok=True)
    
    # Extract gene data
    gene_peaks = subset_peaks(peak_df, window)
    gene_exp = subset_gex(gex_df, gene)
    
    # Create pseudobulk
    pb_peak_df, gex_peak_df = make_all_pseudobulk(
        gene_peaks, gene_exp, gene, pb_keep, gene_outdir, peak_df, gex_df
    )
    
    # Filter peaks based on presence
    selected_filter_idx = config['phase1']['selected_peak_filter']
    selected_filter = config['phase1']['peak_filters'][selected_filter_idx]
    min_presence = selected_filter['min_sample_presence']
    filter_name = selected_filter['name']
    
    logger.info(f"  Using peak filter: {filter_name} (min_sample_presence: {min_presence})")
    
    n_samples_required = int(len(pb_peak_df.columns) * min_presence)
    peak_set = pb_peak_df.loc[
        pb_peak_df[pb_peak_df.columns].ne(0).sum(axis=1) >= n_samples_required
    ]
    
    # Check if we have enough peaks
    min_peaks = config['advanced']['min_peaks_per_gene']
    if len(peak_set) < min_peaks:
        logger.warning(f"Gene {gene} has only {len(peak_set)} peaks, skipping")
        return None
    
    # Check if gene has expression
    if gex_peak_df.max().max() == 0 or np.isnan(gex_peak_df.max().max()):
        logger.warning(f"Gene {gene} has no expression, skipping")
        return None
    
    # Prepare data for modeling
    peaks_array = peak_set.values.T
    gex_array = gex_peak_df.values.T
    
    X = pd.DataFrame(peaks_array, columns=peak_set.index, index=peak_set.columns.tolist())
    y = pd.DataFrame(gex_array, columns=gex_peak_df.index, index=peak_set.columns.tolist())
    
    # Initialize model builder
    model_builder = ModelBuilder(config, phase='phase1')
    
    # Run all models and methods
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
        
        for method in methods:
            try:
                result = model_builder.build_and_evaluate_model(
                    model_name, X, y, gene, gene_outdir, method
                )
                results[f"{model_name}_{method}"] = result
                logger.info(f"    {method}: R² = {result['avg_r2']:.4f}")
            except Exception as e:
                logger.error(f"    Error with {method}: {str(e)}")
    
    # Aggregate peak importance across all models
    agg_importance = aggregate_peak_ranks(gene_outdir, config)
    
    return {
        'gene': gene,
        'n_peaks_total': len(peak_set),
        'results': results,
        'aggregated_importance': agg_importance
    }


def run_phase2_for_gene(gene, config, phase1_results):
    """Run Phase 2 analysis for a single gene using Phase 1 results"""
    logger = logging.getLogger(__name__)
    logger.info(f"Phase 2: Processing gene {gene}")
    
    # Check if we have Phase 1 results
    phase1_dir = os.path.join(config['output_dir'], 'phase1', gene)
    if not os.path.exists(phase1_dir):
        logger.warning(f"No Phase 1 results found for gene {gene}")
        return None
    
    # Load aggregated importance
    agg_path = os.path.join(phase1_dir, 'aggregated_peak_importances.csv')
    if not os.path.exists(agg_path):
        logger.warning(f"No aggregated importance found for gene {gene}")
        return None
    
    # Load data
    peaks_df = pd.read_csv(os.path.join(phase1_dir, 'peaks.csv'), index_col=0)
    gex_df = pd.read_csv(os.path.join(phase1_dir, 'gex.csv'), index_col=0)
    agg_importance = pd.read_csv(agg_path)
    
    # Select top features
    threshold = config['phase2']['top_features_percentage']
    min_zscore = agg_importance['Average_Zscore'].min()
    agg_importance['Adjusted_Zscore'] = agg_importance['Average_Zscore'] + abs(min_zscore)
    
    total_sum = agg_importance['Adjusted_Zscore'].sum()
    cumsum_threshold = threshold * total_sum
    top_peaks = agg_importance[
        agg_importance['Adjusted_Zscore'].cumsum() <= cumsum_threshold
    ]['Peak'].tolist()
    
    # Subset to selected peaks
    selected_peaks = peaks_df.loc[peaks_df.index.isin(top_peaks)]
    
    if len(selected_peaks) == 0:
        logger.warning(f"No peaks selected for gene {gene}")
        return None
    
    # Create output directory
    gene_outdir = os.path.join(config['output_dir'], 'phase2', gene)
    os.makedirs(gene_outdir, exist_ok=True)
    
    # Prepare data
    peaks_array = selected_peaks.values.T
    gex_array = gex_df.values.T
    
    X = pd.DataFrame(peaks_array, columns=selected_peaks.index, index=selected_peaks.columns.tolist())
    y = pd.DataFrame(gex_array, columns=gex_df.index, index=selected_peaks.columns.tolist())
    
    # Initialize model builder for Phase 2
    model_builder = ModelBuilder(config, phase='phase2')
    
    # Run models
    results = {}
    models_to_run = [name for name, cfg in config['models'].items() if cfg.get('enabled', True)]
    
    for model_name in models_to_run:
        logger.info(f"  Running {model_name}")
        
        # Use same methods as Phase 1
        if model_name == 'linear_regression':
            methods = ['perm_ranker', 'dropcol_ranker']
        elif model_name == 'random_forest':
            methods = ['rf_ranker', 'perm_ranker', 'dropcol_ranker']
        elif model_name == 'xgboost':
            methods = ['xgb_ranker', 'perm_ranker', 'dropcol_ranker']
        elif model_name == 'lightgbm':
            methods = ['lgbm_ranker', 'perm_ranker', 'dropcol_ranker']
        
        for method in methods:
            try:
                result = model_builder.build_and_evaluate_model(
                    model_name, X, y, gene, gene_outdir, method
                )
                results[f"{model_name}_{method}"] = result
                logger.info(f"    {method}: R² = {result['avg_r2']:.4f}")
            except Exception as e:
                logger.error(f"    Error with {method}: {str(e)}")
    
    return {
        'gene': gene,
        'n_peaks_selected': len(selected_peaks),
        'n_peaks_original': len(peaks_df),
        'results': results
    }


def summarize_results(config, phase):
    """Summarize results across all genes for a phase"""
    logger = logging.getLogger(__name__)
    logger.info(f"Summarizing {phase} results")
    
    phase_dir = os.path.join(config['output_dir'], phase)
    summary_data = []
    
    for gene_dir in os.listdir(phase_dir):
        gene_path = os.path.join(phase_dir, gene_dir)
        if os.path.isdir(gene_path):
            # Find all result files
            for file in os.listdir(gene_path):
                if file.endswith('_results.txt'):
                    result_df = pd.read_csv(os.path.join(gene_path, file))
                    parts = file.replace('_results.txt', '').split('_')
                    gene = parts[0]
                    method = '_'.join(parts[1:])
                    
                    summary_data.append({
                        'Gene': gene,
                        'Method': method,
                        'nPeaks': result_df['nPeaks'].iloc[0],
                        'R2': result_df['R2'].iloc[0],
                        'Phase': phase
                    })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(
            config['output_dir'], 
            'aggregated', 
            f'{phase}_summary.csv'
        )
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary to {summary_path}")
        
        # Print summary statistics
        logger.info(f"\n{phase.upper()} Summary Statistics:")
        logger.info(f"Total genes analyzed: {summary_df['Gene'].nunique()}")
        logger.info(f"Average R²: {summary_df['R2'].mean():.4f}")
        logger.info(f"Median R²: {summary_df['R2'].median():.4f}")
        logger.info(f"Best R²: {summary_df['R2'].max():.4f}")
    

def main():
    parser = argparse.ArgumentParser(description='Run snATAC-Express workflow')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--phase', type=str, choices=['1', '2', 'both'],
                        default='both', help='Which phase(s) to run')
    parser.add_argument('--gene', type=str, default=None,
                        help='Run analysis for a single gene only')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    create_output_dirs(config)
    
    # Setup logging
    logger = setup_logging(config['output_dir'])
    logger.info("Starting snATAC-Express workflow")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Phase(s): {args.phase}")
    
    try:
        # Load gene list
        gene_list_path = os.path.join('example_data', 'input_data', config['input_data']['gene_list'])
        gene_df = pd.read_csv(gene_list_path, sep='\t')
        gene_df.columns = ['gene', 'window']
        
        if args.gene:
            # Filter to single gene if specified
            gene_df = gene_df[gene_df['gene'] == args.gene]
            if len(gene_df) == 0:
                raise ValueError(f"Gene {args.gene} not found in gene list")
        
        # Phase 1
        if args.phase in ['1', 'both']:
            logger.info("\n" + "="*50)
            logger.info("PHASE 1: Initial modeling and feature ranking")
            logger.info("="*50)
            
            # Load data
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
            
            logger.info(f"Processing {len(gene_df)} genes...")
            
            # Process each gene
            phase1_results = {}
            for idx, row in gene_df.iterrows():
                gene = row['gene']
                window = row['window']
                
                result = run_phase1_for_gene(
                    gene, window, config, peak_df, gex_df, pb_keep
                )
                
                if result:
                    phase1_results[gene] = result
            
            # Summarize Phase 1
            summarize_results(config, 'phase1')
            logger.info(f"Phase 1 completed. Processed {len(phase1_results)} genes.")
        
        # Phase 2
        if args.phase in ['2', 'both']:
            logger.info("\n" + "="*50)
            logger.info("PHASE 2: Refined modeling on selected features")
            logger.info("="*50)
            
            # Load Phase 1 results if not already loaded
            if args.phase == '2':
                phase1_results = {}
                phase1_dir = os.path.join(config['output_dir'], 'phase1')
                if os.path.exists(phase1_dir):
                    for gene_dir in os.listdir(phase1_dir):
                        if os.path.isdir(os.path.join(phase1_dir, gene_dir)):
                            phase1_results[gene_dir] = {'gene': gene_dir}
            
            logger.info(f"Processing {len(phase1_results)} genes with Phase 1 results...")
            
            # Process each gene
            phase2_results = {}
            for gene in phase1_results:
                if args.gene and gene != args.gene:
                    continue
                    
                result = run_phase2_for_gene(gene, config, phase1_results.get(gene))
                
                if result:
                    phase2_results[gene] = result
            
            # Summarize Phase 2
            summarize_results(config, 'phase2')
            logger.info(f"Phase 2 completed. Processed {len(phase2_results)} genes.")
        
        logger.info("\nWorkflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}")
        raise


if __name__ == "__main__":
    main()