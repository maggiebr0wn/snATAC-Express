# snATAC-Express

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

<img src="https://github.com/maggiebr0wn/ATAC-Express/blob/main/atac-express.jpg" align="right" width="400">

**snATAC-Express** predicts gene expression from single-nucleus ATAC-seq data using machine learning, and highlights key cis-regulatory regions driving expression.

## 🔍 Features

- ML models: Random Forest, XGBoost, LightGBM, Linear Regression  
- Feature selection: model-based, permutation, drop-column  
- Nested cross-validation + hyperparameter tuning  
- Outputs: metrics, predictions, ranked regions

## ⚙️ Installation

Set up a conda environment:

```bash
conda create -n snatacexpress -c conda-forge python=3.8
conda activate snatacexpress
conda install -c conda-forge numpy pandas scipy scikit-learn xgboost lightgbm \
  pyyaml jupyter matplotlib seaborn statsmodels h5py typing-extensions
```

Clone the repository:

```bash
git clone https://github.com/maggiebr0wn/ATAC-Express.git
cd snATAC-Express
```

Download GENCODE annotation:

```bash
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.annotation.gtf.gz
gunzip gencode.v41.annotation.gtf.gz
```

## 🚀 Usage

Prepare input files in `input_data/` (sparse matrices, gene list, group coverages).

### Run example steps

Extract gene coordinates:

```bash
./scripts/get_gene_coords.sh -g genelist.txt -a gencode.v41.annotation.gtf -r genebody -w 100000 -o coords/
```

Train model for one gene:

```bash
python scripts/run_multitest.py -g genelist.txt -n BACH2 \
  -gex sparse_gex_matrix.txt -pks sparse_peak_matrix.txt -pb 1 -f 10 -out Results/
```

Run jobs in parallel (Slurm):

```bash
python scripts/init_parallel_sbatch.py -g genelist.txt -o Results/
```

## 📁 Output

Each gene folder contains:
- `model.pkl`: Trained model  
- `predictions.csv`: Predicted vs actual  
- `feature_importance.csv`: Feature rankings  
- `crossval_results.txt`: Performance metrics

## 🧪 Example Data

See `example_data/` for sample inputs.

## 📄 License

MIT License. See [LICENSE](LICENSE).
