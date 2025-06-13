#!/usr/bin/env python3
"""
Initialize parallel SLURM jobs for snATAC-Express
"""

import argparse
import os
import pandas as pd
from pathlib import Path


def create_slurm_script(gene, config_path, output_dir):
    """Create SLURM submission script for a single gene"""
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=snatac_{gene}
#SBATCH --output={output_dir}/logs/{gene}_%j.out
#SBATCH --error={output_dir}/logs/{gene}_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Load required modules (adjust for your HPC environment)
module load python/3.8
module load gcc

# Activate conda environment if needed
# source activate snatacexpress

# Run snATAC-Express for single gene
snatac-express --config {config_path} --gene {gene} --phase both
"""
    
    return script_content


def main():
    parser = argparse.ArgumentParser(description='Initialize parallel SLURM jobs')
    parser.add_argument('-g', '--gene_list', required=True,
                        help='Path to gene list file')
    parser.add_argument('-c', '--config', default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--dry_run', action='store_true',
                        help='Create scripts without submitting')
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.output_dir}/slurm_scripts", exist_ok=True)
    
    # Load gene list
    gene_df = pd.read_csv(args.gene_list, sep='\t', header=None)
    if len(gene_df.columns) == 1:
        genes = gene_df[0].tolist()
    else:
        genes = gene_df[0].tolist()  # Assume first column is gene names
    
    print(f"Preparing to submit {len(genes)} jobs...")
    
    # Create and optionally submit jobs
    for gene in genes:
        script_path = f"{args.output_dir}/slurm_scripts/{gene}.sbatch"
        
        # Write SLURM script
        with open(script_path, 'w') as f:
            f.write(create_slurm_script(gene, args.config, args.output_dir))
        
        # Submit job unless dry run
        if not args.dry_run:
            os.system(f"sbatch {script_path}")
            print(f"Submitted job for {gene}")
        else:
            print(f"Created script for {gene} (dry run)")
    
    print(f"\nAll jobs {'created' if args.dry_run else 'submitted'}!")
    print(f"Check logs in: {args.output_dir}/logs/")


if __name__ == "__main__":
    main()