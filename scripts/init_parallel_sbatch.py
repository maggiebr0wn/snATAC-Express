#!/usr/sbin/anaconda

import argparse
import os
import pandas as pd
import sys

# 11-10-2023
# Prepare scripts to run in parallel
os.chdir("/storage/home/hcoda1/6/mfisher42/scratch/scATAC_Express/RA_Genes_01262024/Multitest_kfoldcv_95featselect_hyperparam_50perc_parallel_01302024")
genes_df = pd.read_csv("/storage/home/hcoda1/6/mfisher42/scratch/scATAC_Express/RA_Genes_01262024/Multitest_kfoldcv_95featselect_hyperparam_50perc_parallel_01302024/genelist_genebody.txt", sep = "\t")
genes_df.columns = ["gene", "window"]

# Loop through the genes and create Slurm scripts
for gene in genes_df["gene"]:
    print(gene)
    # make slurm script
    slurm_script = f"""#!/bin/bash
#SBATCH -J {gene}_10perc_parallel_pipeline
#SBATCH -A gts-ggibson3-biocluster
#SBATCH -N1 
#SBATCH --gres=gpu:V100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH -t 72:00:00
#SBATCH -q inferno
#SBATCH -o Report-j%_{gene}_output.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mfisher42@gatech.edu

cd $SLURM_SUBMIT_DIR
module load anaconda3/2022.05.0.1
conda activate LinReg
srun --cpus-per-task=24 python ./run_multitest.py -g /storage/home/hcoda1/6/mfisher42/scratch/scATAC_Express/RA_Genes_01262024/Multitest_kfoldcv_95featselect_hyperparam_50perc_parallel_01302024/genelist_genebody.txt -n {gene} -gex /storage/home/hcoda1/6/mfisher42/scratch/scATAC_Express/input_data/sparse_gex_matrix.txt -pks /storage/home/hcoda1/6/mfisher42/scratch/scATAC_Express/input_data/sparse_peak_matrix.txt -pb 1 -out /storage/home/hcoda1/6/mfisher42/scratch/scATAC_Express/RA_Genes_01262024/Multitest_kfoldcv_95featselect_hyperparam_50perc_parallel_01302024/Results
"""
    # Save the Slurm script to a file
    with open(f"{gene}_slurm_script.sbatch", "w") as file:
        file.write(slurm_script)
    print(f"Created Slurm script for gene: {gene}")
    
