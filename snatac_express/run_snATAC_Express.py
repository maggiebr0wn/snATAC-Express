#!/usr/bin/env python3

import os
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Import the actual workflow runner
from .scripts.run_multi_test import main as run_workflow

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
        config['output_dir'],
        os.path.join(config['output_dir'], 'phase1'),
        os.path.join(config['output_dir'], 'phase2'),
        os.path.join(config['output_dir'], 'aggregated')
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

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
        # Run the actual workflow
        run_workflow()
        logger.info("Workflow completed successfully")
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}")
        raise

if __name__ == "__main__":
    main() 