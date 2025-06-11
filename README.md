# snATAC-Express

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

<img src="https://github.com/maggiebr0wn/ATAC-Express/blob/main/atac-express.jpg" align="right" width="500" height="300">

A computational pipeline for predicting gene expression from single-nucleus ATAC-seq data and identifying key regulatory regions.

## Overview

snATAC-Express is a machine learning-based pipeline that leverages single-nucleus ATAC-seq (snATAC-seq) data to predict gene expression levels and identify the most important regulatory regions for accurate gene expression prediction. This tool bridges the gap between chromatin accessibility and gene expression at single-cell resolution.

### Key Features

- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, and Linear Regression
- **Feature Selection**: Multiple methods for ranking regulatory regions:
  - Model-specific feature importance
  - Permutation importance
  - Drop-column importance
- **Cross-Validation**: Nested k-fold cross-validation for model evaluation
- **Hyperparameter Optimization**: Automated tuning of model parameters
- **Comprehensive Output**: Detailed results including:
  - Model performance metrics
  - Feature importance rankings
  - Prediction vs actual comparisons
  - Cross-validation results

## Installation

### Prerequisites

- Python 3.8 or higher
- Unix-like operating system (Linux/MacOS)
- Basic command-line tools (wget, gunzip)

### Recommended: Conda Environment Setup

We recommend using [conda](https://docs.conda.io/en/latest/) to manage your Python environment and dependencies for snATAC-Express.

#### 1. Create a new environment to support dependences

```bash
conda env create -f environment.yml
conda activate snatac-express
```

#### 2. Verify installation

To check that all packages are installed, run:

```bash
python -c "import numpy, pandas, scipy, sklearn, xgboost, lightgbm, matplotlib, seaborn, statsmodels, h5py"
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/maggiebr0wn/ATAC-Express.git
cd snATAC-Express
```

2. Download and prepare the GENCODE annotation file:
```bash
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.annotation.gtf.gz
gunzip gencode.v41.annotation.gtf.gz
```

## Workflow

The snATAC-Express pipeline consists of several steps that can be run sequentially or in parallel. Here's a detailed guide on how to implement the workflow:

### 1. Data Preparation

1. **Prepare Input Data**:
   - Organize your snATAC-seq and snRNA-seq data in the following format:
     ```
     input_data/
     ├── sparse_peak_matrix.txt        # Peak accessibility matrix
     ├── sparse_peak_matrix_rownames.txt  # Peak coordinates
     ├── sparse_peak_matrix_colnames.txt  # Cell barcodes
     ├── sparse_gex_matrix.txt         # Gene expression matrix
     ├── sparse_gex_matrix_rownames.txt   # Gene names
     ├── sparse_gex_matrix_colnames.txt   # Cell barcodes
     └── group_coverages.csv           # Pseudobulk information
     ```

2. **Create Gene List**:
   - Prepare a tab-separated file containing genes to analyze:
     ```
     gene    window
     GENE1   chr1:1000-2000
     GENE2   chr2:3000-4000
     ```

### 2. Gene Coordinate Extraction

1. **Run the coordinate extraction script**:
   ```bash
   ./get_gene_coords.sh -g <gene_list> \
                        -a <gencode_annotations> \
                        -r <region_type: tss/genebody> \
                        -w <window_size> \
                        -o <output_directory>
   ```

   Parameters:
   - `-g`: Path to gene list file
   - `-a`: Path to GENCODE annotation file
   - `-r`: Region type (tss or genebody)
   - `-w`: Window size in base pairs
   - `-o`: Output directory

### 3. Model Training

1. **Single Gene Analysis**:
   ```bash
   python scripts/run_multitest.py \
     -g <gene_list> \
     -n <gene_name> \
     -gex <gex_matrix> \
     -pks <peak_matrix> \
     -pb <pseudobulk_replicate> \
     -f <peak_filter> \
     -out <output_directory>
   ```

   Parameters:
   - `-g`: Path to gene list file
   - `-n`: Name of gene to process
   - `-gex`: Path to gene expression matrix
   - `-pks`: Path to peak accessibility matrix
   - `-pb`: Pseudobulk replicate version (1 or 2)
   - `-f`: Minimum percentage of samples a peak must be present in (1-100)
   - `-out`: Output directory

2. **Parallel Processing**:
   ```bash
   python scripts/init_parallel_sbatch.py \
     -g <gene_list> \
     -o <output_directory>
   ```

   This will create individual Slurm scripts for each gene that can be submitted to a cluster.

### 4. Results Analysis

1. **Model Performance**:
   - Check the output directory for each gene:
     ```
     Results/
     ├── <gene_name>/
     │   ├── rf_ranker/
     │   ├── perm_ranker/
     │   └── dropcol_ranker/
     │       ├── model.pkl
     │       ├── predictions.csv
     │       └── feature_importance.csv
     ```

2. **Feature Importance**:
   - Review the feature importance rankings in each method's directory
   - Compare performance across different feature selection methods
   - Analyze the top 95% cumulatively important peaks

### 5. Output Files

For each gene and method, the pipeline generates:
- Trained model files (`.pkl`)
- Prediction results (`.csv`)
- Feature importance rankings (`.csv`)
- Cross-validation results (`.txt`)
- Performance metrics (`.txt`)

## Configuration

The pipeline can be configured through several parameters:

1. **Data Processing**:
   - Minimum cells per pseudobulk (default: 25)
   - Peak presence threshold (default: 10% or 50%)
   - Normalization method (CPM + log2)

2. **Model Parameters**:
   - Cross-validation folds (default: 3)
   - Splits per fold (default: 5)
   - Feature importance threshold (default: 95%)

3. **Resource Allocation**:
   - Number of CPU cores
   - Memory allocation
   - GPU requirements (if using)

## Troubleshooting

Common issues and solutions:

1. **Memory Issues**:
   - Reduce the number of parallel jobs
   - Increase memory allocation in Slurm scripts
   - Process genes in smaller batches

2. **Performance Issues**:
   - Check input data format
   - Verify pseudobulk calculations
   - Monitor cross-validation results

3. **Feature Selection**:
   - Adjust peak presence threshold
   - Modify feature importance threshold
   - Check for data sparsity

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use snATAC-Express in your research, please cite our work (citation information will be available upon publication).

## Contact

For questions and support, please open an issue in the GitHub repository.

## Input Requirements

### Required Input Files

1. **Gene Expression (GEX) Data**:
   - `sparse_gex_matrix.txt`: Sparse matrix in Matrix Market format containing gene expression values
   - `sparse_gex_matrix_rownames.txt`: Gene names (one per line)
   - `sparse_gex_matrix_colnames.txt`: Cell barcodes (one per line)
   - Format: Sparse matrix with genes as rows and cells as columns
   - Data type: uint8 (0s and 1s)

2. **ATAC-seq Peak Data**:
   - `sparse_peak_matrix.txt`: Sparse matrix in Matrix Market format containing peak accessibility values
   - `sparse_peak_matrix_rownames.txt`: Peak coordinates in format "chr:start-end" (one per line)
   - `sparse_peak_matrix_colnames.txt`: Cell barcodes (one per line)
   - Format: Sparse matrix with peaks as rows and cells as columns
   - Data type: uint8 (0s and 1s)

3. **Gene List**:
   - `genelist_genebody.txt`: Tab-separated file with two columns:
     - Column 1: Gene name
     - Column 2: Genomic window for cis-regulatory elements (format: "chr:start-end")
   - Example:
     ```
     gene    genebody_window_100000
     BACH2   chr6:89826528-90396843
     ```

4. **Group Coverages**:
   - `group_coverages.csv`: Output from ArchR after creating pseudobulk replicates and running peak calling
   - Format: CSV file with the following columns:
     - `peakId`: Unique identifier for each peak (format: "chr:start-end")
     - `group`: Group/replicate identifier
     - `coverage`: Peak coverage value for the group
   - Example:
     ```
     peakId,group,coverage
     chr1:1000-2000,group1,45
     chr1:2000-3000,group1,67
     chr1:1000-2000,group2,52
     chr1:2000-3000,group2,71
     ```
   - Requirements:
     - Peak IDs must match the format in `sparse_peak_matrix_rownames.txt`
     - Coverage values should be non-negative integers
     - Each peak should have coverage values for all groups
     - Groups should represent pseudobulk replicates

### Input Data Preparation

1. **Creating Sparse Matrices**:
   - Convert your single-cell gene expression and ATAC-seq data to sparse matrices
   - Ensure cell barcodes match between GEX and ATAC data
   - Save matrices in Matrix Market format with .txt extension
   - Save row and column names as plain text files

2. **Gene List Preparation**:
   - Create a list of genes of interest
   - For each gene, specify a genomic window to search for cis-regulatory elements
   - Windows should be large enough to capture potential regulatory elements
   - Format as tab-separated text file

3. **ArchR Output**:
   - Run ArchR to create pseudobulk replicates
   - Perform peak calling
   - Export group coverages as CSV
   - Ensure peak IDs match your peak matrix format
   - Verify coverage values are complete for all peaks and groups

### Example Data

The `example_data` directory contains sample input files:
- `input_data/`: Directory containing:
  - Example sparse matrices and their row/column names
  - `group_coverages.csv`: Example group coverages file showing peak coverage values for each pseudobulk replicate
- `genelist_genebody.txt`: Example gene list with genomic windows

### Notes

- All input files should be in plain text format with .txt extension
- Sparse matrices should be in Matrix Market format
- Cell barcodes must match between GEX and ATAC data
- Genomic coordinates should be in the same format as your reference genome
- Ensure sufficient memory for handling large sparse matrices
- Group coverages CSV should be complete (no missing values) and properly formatted