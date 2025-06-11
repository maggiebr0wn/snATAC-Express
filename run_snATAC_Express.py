#!/usr/bin/env python3

import os
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Import workflow modules
from scripts.data_preprocessing import preprocess_data
from scripts.model_builders import build_models
from scripts.feature_selection import select_features
from scripts.run_multites import run_models
from scripts.summarize_cv_results import summarize_results

def setup_logging(output_dir):
    """Set up logging configuration"""
    log_file = os.path.join(output_dir, f"snATAC_Express_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
        config['paths']['phase1_output'],
        config['paths']['phase2_output'],
        config['paths']['aggregated_results']
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def run_phase1(config, logger):
    """Run Phase 1 of the workflow"""
    logger.info("Starting Phase 1")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    preprocess_data(
        input_dir=config['paths']['input_data'],
        gene_list=config['paths']['gene_list'],
        output_dir=config['paths']['phase1_output']
    )
    
    # Build models
    logger.info("Building models...")
    build_models(
        input_dir=config['paths']['phase1_output'],
        output_dir=config['paths']['phase1_output'],
        config=config['phase1']
    )
    
    # Select features
    logger.info("Selecting features...")
    select_features(
        input_dir=config['paths']['phase1_output'],
        output_dir=config['paths']['aggregated_results'],
        threshold=config['phase1']['feature_selection']['variance_threshold']
    )
    
    # Run models
    logger.info("Running models...")
    run_models(
        input_dir=config['paths']['phase1_output'],
        output_dir=config['paths']['phase1_output'],
        config=config['phase1']
    )
    
    # Summarize results
    logger.info("Summarizing results...")
    summarize_results(
        input_dir=config['paths']['phase1_output'],
        output_dir=config['paths']['phase1_output']
    )
    
    logger.info("Phase 1 completed")

def run_phase2(config, logger):
    """Run Phase 2 of the workflow"""
    logger.info("Starting Phase 2")
    
    # Check if aggregated results exist
    if not os.path.exists(config['paths']['aggregated_results']):
        raise FileNotFoundError("Aggregated results from Phase 1 not found")
    
    # Build models with selected features
    logger.info("Building models with selected features...")
    build_models(
        input_dir=config['paths']['aggregated_results'],
        output_dir=config['paths']['phase2_output'],
        config=config['phase2']
    )
    
    # Run models
    logger.info("Running models...")
    run_models(
        input_dir=config['paths']['aggregated_results'],
        output_dir=config['paths']['phase2_output'],
        config=config['phase2']
    )
    
    # Summarize results
    logger.info("Summarizing results...")
    summarize_results(
        input_dir=config['paths']['phase2_output'],
        output_dir=config['paths']['phase2_output']
    )
    
    logger.info("Phase 2 completed")

def main():
    parser = argparse.ArgumentParser(description='Run snATAC-Express workflow')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--phase', type=str, choices=['1', '2', 'both'],
                      default='both', help='Which phase(s) to run')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    create_output_dirs(config)
    
    # Setup logging
    logger = setup_logging(config['output_dir'])
    
    try:
        if args.phase in ['1', 'both']:
            run_phase1(config, logger)
        if args.phase in ['2', 'both']:
            run_phase2(config, logger)
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}")
        raise

if __name__ == "__main__":
    main() 