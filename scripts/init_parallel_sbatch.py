#!/usr/sbin/anaconda

"""
Slurm Job Script Generator for snATAC-Express

This script generates Slurm job scripts for parallel processing of genes in the
snATAC-Express pipeline. It creates individual job scripts for each gene in the
input list with appropriate resource allocations and command-line arguments for
the run_multitest.py script.
"""

from typing import Dict, List, Tuple, Union, Optional
import argparse
import os
import pandas as pd
from pathlib import Path
from config import (
    NUM_JOBS,
    MEMORY_LIMIT,
    GPU_REQUIREMENT,
    SLURM_ACCOUNT,
    SLURM_PARTITION,
    SLURM_TIME_LIMIT,
    SLURM_EMAIL,
    get_gene_output_dir
)

def parse_args() -> Dict[str, str]:
    """
    Parse command line arguments.

    Returns:
        Dictionary containing parsed arguments:
        - gene_list: Path to gene list file
        - output_dir: Directory to save Slurm scripts
    """
    parser = argparse.ArgumentParser(description='Generate Slurm job scripts')
    parser.add_argument('--gene_list', required=True, help='Path to gene list file')
    parser.add_argument('--output_dir', required=True, help='Directory to save Slurm scripts')
    args = parser.parse_args()
    return vars(args)

def create_slurm_script(gene: str, output_dir: str) -> str:
    """
    Create a Slurm script for processing a single gene.

    Args:
        gene: Name of the gene to process
        output_dir: Directory to save the Slurm script

    Returns:
        String containing the Slurm script content
    """
    script = f"""#!/bin/bash
#SBATCH --account={SLURM_ACCOUNT}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --time={SLURM_TIME_LIMIT}
#SBATCH --mem={MEMORY_LIMIT}
#SBATCH --cpus-per-task={NUM_JOBS}
#SBATCH --mail-user={SLURM_EMAIL}
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name={gene}_snATAC_Express
#SBATCH --output={gene}_snATAC_Express_%j.out
#SBATCH --error={gene}_snATAC_Express_%j.err
"""

    if GPU_REQUIREMENT:
        script += f"#SBATCH --gres=gpu:1\n"

    script += f"""
# Load required modules
module load python/3.8

# Activate virtual environment (if needed)
# source /path/to/venv/bin/activate

# Run the model training script
python run_multitest.py \\
    --gene_list {args['gene_list']} \\
    --gene_name {gene} \\
    --gex_matrix {args['gex_matrix']} \\
    --peak_matrix {args['peak_matrix']} \\
    --pseudobulk_replicate {args['pseudobulk_replicate']} \\
    --peak_filter {args['peak_filter']} \\
    --output_dir {output_dir}
"""
    return script

def main():
    """
    Main function to generate Slurm job scripts.
    """
    # Parse arguments
    args = parse_args()
    
    # Change to output directory
    os.chdir(args['output_dir'])
    
    # Read gene list
    gene_list = pd.read_csv(args['gene_list'], header=None)[0].tolist()
    
    # Generate Slurm scripts for each gene
    for gene in gene_list:
        script = create_slurm_script(gene, args['output_dir'])
        script_path = Path(args['output_dir']) / f"{gene}_slurm_script.sbatch"
        with open(script_path, 'w') as f:
            f.write(script)
        print(f"Created Slurm script for gene: {gene}")

if __name__ == '__main__':
    main()
    
