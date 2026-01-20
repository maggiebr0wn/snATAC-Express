# snATAC-Express

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/snatac-express.svg)](https://badge.fury.io/py/snatac-express)

<img src="https://github.com/maggiebr0wn/ATAC-Express/blob/main/atac-express.jpg" align="right" width="400">

**snATAC-Express** predicts gene expression from single-nucleus ATAC-seq data using machine learning, and highlights key cis-regulatory regions driving expression. *Manuscript under review*

## 🔍 Features

- **Multiple ML Models**: Random Forest Regression, XGBoost, LightGBM, Linear Regression  
- **Multiple Feature Selection Methods**: Model-based, permutation, drop-column importance
- **Robust Validation**: Nested cross-validation with hyperparameter tuning  
- **Comprehensive Outputs**: Performance metrics, predictions, ranked regulatory regions
- **Two-Phase Workflow**: Initial feature selection followed by refined modeling using aggregated importance

## 📄 Our preprint is live!

Please see our preprint here: 

[snATAC-Express infers Gene Expression from Prioritized Chromatin Accessibility Peaks using Machine Learning](https://www.biorxiv.org/content/10.1101/2025.07.25.666784v1) 

## ⚙️ Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install snatac-express
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/maggiebr0wn/ATAC-Express.git
cd ATAC-Express

# Install the package
pip install -e .
```

### Option 3: Conda Environment (Alternative)

If you prefer using conda:

```bash
# Create conda environment
mamba create -n snatacexpress -c conda-forge python=3.8 \
  numpy pandas scipy scikit-learn xgboost lightgbm \
  matplotlib pyyaml h5py joblib

# Activate and install additional packages
mamba activate snatacexpress
mamba install -c conda-forge jupyter seaborn statsmodels

# Install snATAC-Express
pip install snatac-express
```

## 🚀 Quick Start

### 1. Prepare Your Data

Organize your input files in the following structure:
```
input_data/
├── sparse_gex_matrix.txt.mtx     # Gene expression matrix (sparse format)
├── sparse_peak_matrix.txt.mtx    # Peak accessibility matrix (sparse format)
├── group_coverages.csv           # Cell group coverage information (obtained from ArchR)
└── genelist_genebody.txt         # List of genes to analyze
```

Examples are provided in the example_data folder.

### 2. Configure Your Analysis

Copy and modify the configuration file:
```bash
cp snatac_express/config.yaml my_config.yaml
# Edit my_config.yaml with your specific settings
```

### 3. Run the Analysis

#### Using the Command Line Interface
```bash
# Run both phases (recommended)
snatac-express --config my_config.yaml --phase both

# Run only Phase 1 (feature selection and initial modeling)
snatac-express --config my_config.yaml --phase 1

# Run only Phase 2 (aggregated feature selection and refined modeling)
snatac-express --config my_config.yaml --phase 2

# Run for specific gene(s)
snatac-express --config my_config.yaml --phase both --gene BACH2
```

#### Using Python API
```python
import snatac_express
from snatac_express.run_snATAC_Express import main

# Run the workflow programmatically
main()
```

### 4. Extract Gene Coordinates (Optional)

If you need to extract gene coordinates from annotation files:

```bash
# Download GENCODE annotation
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.annotation.gtf.gz
gunzip gencode.v41.annotation.gtf.gz

# Extract coordinates
snatac_express/scripts/get_gene_coords.sh -g genelist.txt \
  -a gencode.v41.annotation.gtf -r genebody -w 100000 -o coords/
```

## 📁 Output Structure

The analysis produces organized results in the following structure:

```
results/
├── phase1_results/           # Phase 1: Initial modeling results
│   └── GENE_NAME/
│       ├── model_results/           # Cross-validation performance metrics
│       ├── trained_models/          # Saved trained models (.pkl files)
│       ├── feature_rankings/        # Feature importance rankings by method
│       │   ├── rf_ranker/          # Random Forest built-in importance
│       │   ├── rf_permranker/      # Random Forest permutation importance
│       │   ├── rf_dropcolranker/   # Random Forest drop-column importance
│       │   ├── xgb_ranker/         # XGBoost built-in importance
│       │   ├── xgb_permranker/     # XGBoost permutation importance
│       │   ├── xgb_dropcolranker/  # XGBoost drop-column importance
│       │   ├── lgbm_ranker/        # LightGBM built-in importance
│       │   ├── lgbm_permranker/    # LightGBM permutation importance
│       │   └── lgbm_dropcolranker/ # LightGBM drop-column importance
│       ├── data/                    # Processed input data for this gene
│       └── cross_validation/        # Detailed CV fold results
├── phase2_results/           # Phase 2: Refined modeling results  
│   └── GENE_NAME/
│       ├── model_results/           # Final model performance metrics
│       ├── trained_models/          # Refined models using selected features
│       ├── feature_rankings/        # Feature rankings from refined models
│       └── data/                    # Selected features used in Phase 2
├── aggregated_results/       # Cross-gene aggregated results
│   ├── master_aggregated_peak_ranks.csv      # All genes' top features
│   ├── phase2_aggregated_summary.csv         # Phase 2 performance summary
│   ├── selected_peaks_summary.csv            # Selected features per gene
│   └── GENE_NAME/
│       └── aggregated_peak_importances_exclLR.csv  # Gene-specific aggregated ranks
├── cv_summary.txt            # Phase 1 cross-validation summary
├── phase2_cv_summary.txt     # Phase 2 cross-validation summary
└── logs/
    └── snATAC_Express_YYYYMMDD_HHMMSS.log
```

## 🔄 Two-Phase Workflow

### Phase 1: Feature Selection and Tuning 
1. **Data Processing**: Load and pseudobulk single-cell data
2. **Peak Filtering**: Apply sample presence thresholds (e.g., peaks in ≥10% of samples)
3. **Model Training**: Train multiple ML models with hyperparameter tuning
4. **Feature Ranking**: Generate importance scores using 3 methods per model:
   - Model-based importance (built-in)
   - Permutation importance
   - Drop-column importance
5. **Top Feature Selection**: Select top 95% of features by importance for each method

### Phase 2: Aggregated Feature Selection and Refined Modeling
1. **Importance Aggregation**: Combine feature rankings across all methods using z-scores
2. **Cumulative Selection**: Select features until cumulative importance reaches 95% of total
3. **Refined Modeling**: Train final models using only the aggregated top features
4. **Final Evaluation**: Generate final performance metrics and predictions

## 📓 Tutorial

For a walkthrough of the snATAC-Express workflow, see our interactive tutorial:

**[📖 tutorial.ipynb](tutorial.ipynb)** - Complete step-by-step guide:
- Data loading and inspection
- Configuration setup
- Running both Phase 1 and Phase 2
- Results interpretation

The tutorial uses the provided example data and demonstrates the full two-phase pipeline.

## 🧪 Example Data

The package includes example data in `snatac_express/example_data/` to help you get started:

- Sample sparse matrices
- Example gene lists
- Configuration templates

## 🔧 Configuration

The `config.yaml` file controls all aspects of the analysis:

### Key Configuration Sections:
- **Input/Output Paths**: Data locations and result directories
- **Peak Filtering**: Options for sample presence thresholds (all, 10%, 50%)
- **Model Settings**: Algorithm parameters and hyperparameter grids
- **Feature Selection**: Methods and thresholds for feature ranking
- **Cross-Validation**: Validation strategy and fold settings
- **Phase Settings**: Specific configurations for each phase
- **Advanced Options**: Gene windows, normalization, and aggregation settings

### Important Settings:
```yaml
phase1:
  selected_peak_filter: 1  # 0=all peaks, 1=≥10% samples, 2=≥50% samples
  aggregation:
    include_linear_regression: false  # Exclude LR from aggregation

phase2:
  top_features_percentage: 0.95  # Use top 95% by cumulative importance
```

See the included `config.yaml` for detailed configuration options.

## 📊 Supported Models

- **Linear Regression**: Fast baseline model
- **Random Forest**: Robust ensemble method
- **XGBoost**: Gradient boosting with regularization
- **LightGBM**: High-performance gradient boosting

## 🔬 Feature Selection Methods

- **Model Importance**: Built-in feature importance from tree-based models
- **Permutation Importance**: Robust importance estimation via feature permutation
- **Drop-Column Importance**: Feature ablation analysis (most computationally intensive)

## 🚀 High-Performance Computing

For large-scale analyses, the package supports parallel processing. Please see the example script `init_parallel_sbatch.py` and modify accoridngly for your compute environment.

```bash
# Run on SLURM cluster
python snatac_express/scripts/init_parallel_sbatch.py -g genelist.txt -o Results/
```

## 📄 License

MIT License. See [LICENSE](LICENSE) file for details.
