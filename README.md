# snATAC-Express

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/snatac-express.svg)](https://badge.fury.io/py/snatac-express)

<img src="https://github.com/maggiebr0wn/ATAC-Express/blob/main/atac-express.jpg" align="right" width="400">

**snATAC-Express** predicts gene expression from single-nucleus ATAC-seq data using machine learning, and highlights key cis-regulatory regions driving expression.

## 🔍 Features

- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, Linear Regression  
- **Advanced Feature Selection**: Model-based, permutation, drop-column importance
- **Robust Validation**: Nested cross-validation with hyperparameter tuning  
- **Comprehensive Outputs**: Performance metrics, predictions, ranked regulatory regions
- **Two-Phase Workflow**: Initial feature selection followed by refined modeling
- **Parallel Processing**: Support for high-performance computing environments

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
├── sparse_gex_matrix.mtx      # Gene expression matrix (sparse format)
├── sparse_peak_matrix.mtx     # Peak accessibility matrix (sparse format)
├── group_coverages.csv        # Cell group coverage information
└── genelist.txt              # List of genes to analyze
```

### 2. Configure Your Analysis

Copy and modify the configuration file:
```bash
cp snatac_express/config.yaml my_config.yaml
# Edit my_config.yaml with your specific settings
```

### 3. Run the Analysis

#### Using the Command Line Interface
```bash
# Run both phases
snatac-express --config my_config.yaml --phase both

# Run only Phase 1 (feature selection)
snatac-express --config my_config.yaml --phase 1

# Run only Phase 2 (refined modeling)
snatac-express --config my_config.yaml --phase 2
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
├── phase1_output/
│   ├── gene_name/
│   │   ├── model.pkl              # Trained model
│   │   ├── predictions.csv        # Predicted vs actual values
│   │   ├── feature_importance.csv # Feature rankings
│   │   └── crossval_results.txt   # Performance metrics
│   └── aggregated_results/
│       └── selected_features.csv  # Top features across all genes
├── phase2_output/
│   └── gene_name/
│       ├── refined_model.pkl      # Refined model with selected features
│       ├── refined_predictions.csv
│       └── refined_metrics.txt
└── logs/
    └── snATAC_Express_YYYYMMDD_HHMMSS.log
```

## 🧪 Example Data

The package includes example data in `snatac_express/example_data/` to help you get started:

- Sample sparse matrices
- Example gene lists
- Configuration templates

## 🔧 Configuration

The `config.yaml` file controls all aspects of the analysis:

- **Input/Output Paths**: Data locations and result directories
- **Model Settings**: Algorithm parameters and hyperparameter grids
- **Feature Selection**: Methods and thresholds for feature ranking
- **Cross-Validation**: Validation strategy and fold settings
- **Advanced Options**: Gene windows, normalization, and filtering

See the included `config.yaml` for detailed configuration options.

## 📊 Supported Models

- **Linear Regression**: Fast baseline model
- **Random Forest**: Robust ensemble method
- **XGBoost**: Gradient boosting with regularization
- **LightGBM**: High-performance gradient boosting

## 🔬 Feature Selection Methods

- **Model Importance**: Built-in feature importance from tree-based models
- **Permutation Importance**: Robust importance estimation
- **Drop-Column Importance**: Feature ablation analysis

## 🚀 High-Performance Computing

For large-scale analyses, the package supports parallel processing:

```bash
# Run on SLURM cluster
python snatac_express/scripts/init_parallel_sbatch.py -g genelist.txt -o Results/
```

## 📄 License

MIT License. See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Citation

If you use snATAC-Express in your research, please cite:

```
Brown, M. (2024). snATAC-Express: Predicting gene expression from single-nucleus ATAC-seq data using machine learning. 
GitHub repository: https://github.com/maggiebr0wn/ATAC-Express
```

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the example data and configuration files
- Review the log files for detailed error messages

