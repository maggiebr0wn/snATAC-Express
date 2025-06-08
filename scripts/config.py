"""
Configuration settings for snATAC-Express pipeline.

This module contains all configurable paths and settings used throughout the pipeline.
Paths can be overridden by setting environment variables with the same names.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(os.getenv("SNATAC_EXPRESS_BASE_DIR", "/storage/home/mfisher42/scProjects/Predict_GEX"))
INPUT_DIR = Path(os.getenv("SNATAC_EXPRESS_INPUT_DIR", BASE_DIR / "input_data"))
OUTPUT_DIR = Path(os.getenv("SNATAC_EXPRESS_OUTPUT_DIR", BASE_DIR / "Results"))

# Input data files
PEAK_MATRIX = os.getenv("SNATAC_EXPRESS_PEAK_MATRIX", str(INPUT_DIR / "sparse_peak_matrix.txt"))
PEAK_ROWNAMES = os.getenv("SNATAC_EXPRESS_PEAK_ROWNAMES", str(INPUT_DIR / "sparse_peak_matrix_rownames.txt"))
PEAK_COLNAMES = os.getenv("SNATAC_EXPRESS_PEAK_COLNAMES", str(INPUT_DIR / "sparse_peak_matrix_colnames.txt"))
GEX_MATRIX = os.getenv("SNATAC_EXPRESS_GEX_MATRIX", str(INPUT_DIR / "sparse_gex_matrix.txt"))
GEX_ROWNAMES = os.getenv("SNATAC_EXPRESS_GEX_ROWNAMES", str(INPUT_DIR / "sparse_gex_matrix_rownames.txt"))
GEX_COLNAMES = os.getenv("SNATAC_EXPRESS_GEX_COLNAMES", str(INPUT_DIR / "sparse_gex_matrix_colnames.txt"))
GROUP_COVERAGES = os.getenv("SNATAC_EXPRESS_GROUP_COVERAGES", str(INPUT_DIR / "group_coverages.csv"))

# Model parameters
MIN_CELLS_PER_PSEUDOBULK = int(os.getenv("SNATAC_EXPRESS_MIN_CELLS", "25"))
CV_FOLDS = int(os.getenv("SNATAC_EXPRESS_CV_FOLDS", "3"))
SPLITS_PER_FOLD = int(os.getenv("SNATAC_EXPRESS_SPLITS", "5"))
FEATURE_IMPORTANCE_THRESHOLD = float(os.getenv("SNATAC_EXPRESS_FEATURE_THRESHOLD", "0.95"))

# Resource settings
NUM_JOBS = int(os.getenv("SNATAC_EXPRESS_NUM_JOBS", "-1"))  # -1 means use all available cores
MEMORY_LIMIT = os.getenv("SNATAC_EXPRESS_MEMORY_LIMIT", "120G")
GPU_REQUIREMENT = os.getenv("SNATAC_EXPRESS_GPU", "V100:2")

# Slurm settings
SLURM_ACCOUNT = os.getenv("SNATAC_EXPRESS_SLURM_ACCOUNT", "gts-ggibson3-biocluster")
SLURM_PARTITION = os.getenv("SNATAC_EXPRESS_SLURM_PARTITION", "inferno")
SLURM_TIME_LIMIT = os.getenv("SNATAC_EXPRESS_SLURM_TIME", "72:00:00")
SLURM_EMAIL = os.getenv("SNATAC_EXPRESS_SLURM_EMAIL", "mfisher42@gatech.edu")

def ensure_directories():
    """Create necessary directories if they don't exist."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_gene_output_dir(gene: str) -> Path:
    """Get the output directory for a specific gene."""
    gene_dir = OUTPUT_DIR / gene
    gene_dir.mkdir(parents=True, exist_ok=True)
    return gene_dir

def get_method_output_dir(gene: str, method: str) -> Path:
    """Get the output directory for a specific gene and method."""
    method_dir = get_gene_output_dir(gene) / method
    method_dir.mkdir(parents=True, exist_ok=True)
    return method_dir 