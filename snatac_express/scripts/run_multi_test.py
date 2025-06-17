#!/usr/bin/env python3
"""
Updated workflow runner for snATAC-Express
Now supports both Phase 1 and Phase 2 with aggregation
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

# Import workflow modules
from .data_preprocessing import (
    get_pseudobulk, load_peak_input, subset_peaks,
    load_gex_input, subset_gex, make_all_pseudobulk
)
from .model_builder import ModelBuilder
from .run_phase2 import run_phase2_workflow


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


def create_output_dirs(config, phase='both'):
    """Create necessary output directories"""
    dirs = [
        config['output_dir'],
        os.path.join(config['output_dir'], 'phase1_results'),  # Changed from 'results'
        os.path.join(config['output_dir'], 'logs')
    ]
    
    if phase in ['2', 'both']:
        dirs.extend([
            os.path.join(config['output_dir'], 'phase2_results'),
            os.path.join(config['output_dir'], 'aggregated_results')
        ])
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def run_analysis_for_gene(gene, window, config, peak_df, gex_df, pb_keep):
    """Run analysis for a single gene"""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing gene {gene}")
    
    # Create output directory for gene
    gene_outdir = os.path.join(config['output_dir'], 'phase1_results', gene)
    os.makedirs(gene_outdir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['model_results', 'feature_rankings', 'cross_validation', 'trained_models', 'data']
    for subdir in subdirs:
        os.makedirs(os.path.join(gene_outdir, subdir), exist_ok=True)
    
    # Extract gene data
    gene_peaks = subset_peaks(peak_df, window)
    gene_exp = subset_gex(gex_df, gene)
    
    # Create pseudobulk and save to data subdirectory
    data_dir = os.path.join(gene_outdir, 'data')
    pb_peak_df, gex_peak_df = make_all_pseudobulk(
        gene_peaks, gene_exp, gene, pb_keep, data_dir, peak_df, gex_df
    )
    
    # Filter peaks based on presence (10% threshold as in original)
    min_presence = 0.1  # 10% threshold
    n_samples_required = int(len(pb_peak_df.columns) * min_presence)
    peak_set = pb_peak_df.loc[
        pb_peak_df[pb_peak_df.columns].ne(0).sum(axis=1) >= n_samples_required
    ]
    
    logger.info(f"  Total peaks: {len(pb_peak_df)}")
    logger.info(f"  Filtered peaks (≥10% samples): {len(peak_set)}")
    
    # Check if we have enough peaks
    min_peaks = config.get('advanced', {}).get('min_peaks_per_gene', 3)
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
    model_builder = ModelBuilder(config)
    
    # Run all models and methods
    results = {}
    
    # Check if linear regression should be excluded
    include_lr = config.get('phase1', {}).get('aggregation', {}).get('include_linear_regression', True)
    
    # Filter models based on config
    models_to_run = []
    for name, cfg in config['models'].items():
        if cfg.get('enabled', True):
            # Skip linear regression if not included in aggregation
            if name == 'linear_regression' and not include_lr:
                continue
            models_to_run.append(name)
    
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
                # This will build BOTH all-peaks and 95%-peaks models
                result = model_builder.build_and_evaluate_model(
                    model_name, X, y, gene, gene_outdir, method
                )
                results[f"{model_name}_{method}"] = result
                
                logger.info(f"    {method}:")
                logger.info(f"      All peaks: R² = {result['all_peaks']['r2']:.4f} ({result['all_peaks']['n_peaks']} peaks)")
                logger.info(f"      95% peaks: R² = {result['95_peaks']['r2']:.4f} ({result['95_peaks']['n_peaks']} peaks)")
                
            except Exception as e:
                logger.error(f"    Error with {method}: {str(e)}")
    
    return {
        'gene': gene,
        'n_peaks_total': len(peak_set),
        'results': results
    }


def summarize_results(config):
    """Summarize results across all genes"""
    logger = logging.getLogger(__name__)
    logger.info("Summarizing results")
    
    results_dir = os.path.join(config['output_dir'], 'phase1_results')  # Changed from 'results'
    summary_data = []
    
    # Process each gene directory
    for gene_dir in os.listdir(results_dir):
        gene_path = os.path.join(results_dir, gene_dir)
        if os.path.isdir(gene_path):
            # Look in model_results subdirectory first
            model_results_dir = os.path.join(gene_path, 'model_results')
            if os.path.exists(model_results_dir):
                files_to_check = os.listdir(model_results_dir)
                base_dir = model_results_dir
            else:
                # Fallback to gene directory (for backward compatibility)
                files_to_check = os.listdir(gene_path)
                base_dir = gene_path
            
            # Find all result files
            for file in files_to_check:
                if file.endswith('_results.txt'):
                    result_df = pd.read_csv(os.path.join(base_dir, file))
                    
                    # Extract method name
                    parts = file.replace('_results.txt', '').split('_')
                    gene = parts[0]
                    method = '_'.join(parts[1:])
                    
                    # Original format has 2 rows: all peaks and 95% peaks
                    if len(result_df) >= 2:
                        # All peaks (row 1)
                        summary_data.append({
                            'Gene': gene,
                            'Method': method,
                            'nPeaks': result_df.iloc[0]['nPeaks'],
                            'PeakCat': 'All_Peaks',
                            'CV_R2': result_df.iloc[0]['R2']
                        })
                        
                        # 95% peaks (row 2)
                        summary_data.append({
                            'Gene': gene,
                            'Method': method,
                            'nPeaks': result_df.iloc[1]['nPeaks'],
                            'PeakCat': 'Select_Peaks',
                            'CV_R2': result_df.iloc[1]['R2']
                        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(config['output_dir'], 'cv_summary.txt')
        summary_df.to_csv(summary_path, sep='\t', index=False)
        logger.info(f"Saved summary to {summary_path}")
        
        # Print summary statistics
        logger.info("\nSummary Statistics:")
        logger.info(f"Total genes analyzed: {summary_df['Gene'].nunique()}")
        
        # Stats for all peaks
        all_peaks_df = summary_df[summary_df['PeakCat'] == 'All_Peaks']
        logger.info(f"\nAll Peaks:")
        logger.info(f"  Average R²: {all_peaks_df['CV_R2'].mean():.4f}")
        logger.info(f"  Median R²: {all_peaks_df['CV_R2'].median():.4f}")
        
        # Stats for 95% peaks
        select_peaks_df = summary_df[summary_df['PeakCat'] == 'Select_Peaks']
        logger.info(f"\n95% Selected Peaks:")
        logger.info(f"  Average R²: {select_peaks_df['CV_R2'].mean():.4f}")
        logger.info(f"  Median R²: {select_peaks_df['CV_R2'].median():.4f}")


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
    create_output_dirs(config, args.phase)
    
    # Setup logging
    logger = setup_logging(config['output_dir'])
    logger.info("Starting snATAC-Express workflow")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Phase(s) to run: {args.phase}")
    
    try:
        # Run Phase 1 if requested
        if args.phase in ['1', 'both']:
            logger.info("\n" + "="*60)
            logger.info("PHASE 1: Initial modeling with feature selection")
            logger.info("="*60)
            
            # Load gene list
            gene_list_path = os.path.join('example_data', 'input_data', config['input_data']['gene_list'])
            gene_df = pd.read_csv(gene_list_path, sep='\t')
            gene_df.columns = ['gene', 'window']
            
            if args.gene:
                # Filter to single gene if specified
                gene_df = gene_df[gene_df['gene'] == args.gene]
                if len(gene_df) == 0:
                    raise ValueError(f"Gene {args.gene} not found in gene list")
            
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
            all_results = []
            for idx, row in gene_df.iterrows():
                gene = row['gene']
                window = row['window']
                
                result = run_analysis_for_gene(
                    gene, window, config, peak_df, gex_df, pb_keep
                )
                
                if result:
                    all_results.append(result)
            
            # Summarize results
            summarize_results(config)
            
            logger.info(f"\nPhase 1 completed. Processed {len(all_results)} genes.")
        
        # Run Phase 2 if requested
        if args.phase in ['2', 'both']:
            logger.info("\n" + "="*60)
            logger.info("PHASE 2: Aggregation and refined modeling")
            logger.info("="*60)
            
            # Check if Phase 1 results exist
            phase1_results_dir = os.path.join(config['output_dir'], 'phase1_results')  # Changed from 'results'
            if not os.path.exists(phase1_results_dir) or not os.listdir(phase1_results_dir):
                if args.phase == '2':
                    raise ValueError("Phase 1 results not found. Please run Phase 1 first.")
                else:
                    logger.warning("No Phase 1 results found, skipping Phase 2")
                    return
            
            # Run Phase 2 workflow
            phase2_results = run_phase2_workflow(config, phase1_results_dir)
            
            logger.info(f"\nPhase 2 completed. Processed {len(phase2_results)} genes.")
        
        logger.info("\n" + "="*60)
        logger.info("Workflow completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}")
        raise


if __name__ == "__main__":
    main()