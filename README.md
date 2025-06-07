# snATAC-Express

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-active-success.svg)](https://github.com/maggiebr0wn/ATAC-Express)

<img src="https://github.com/maggiebr0wn/ATAC-Express/blob/main/atac-express.jpg" align="right" width="500" height="300">

A computational tool for predicting gene expression from single-nucleus ATAC-seq data and identifying key regulatory regions.

> **Note:** This project is currently under development. A manuscript is in preparation.

## Overview

snATAC-Express is a computational pipeline that leverages single-nucleus ATAC-seq (snATAC-seq) data to predict gene expression levels and identify the most important regulatory regions for accurate gene expression prediction. This tool bridges the gap between chromatin accessibility and gene expression at single-cell resolution.

## Features

- Predict gene expression from snATAC-seq data
- Identify key regulatory regions for gene expression
- Support for custom cis-regulatory window analysis
- Integration with standard genomic annotation formats
- Compatible with single-cell analysis workflows

## Requirements

### Software Dependencies
- Python 3.8 or higher
- Unix-like operating system (Linux/MacOS)
- Basic command-line tools (wget, gunzip)

### Input Data Requirements
- snATAC-seq peak matrix (cell x peak matrix)
- snRNA-seq raw counts matrix (cell x gene matrix)
- GENCODE annotation file (GTF format)

## Installation

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

## Usage

### Step 1: Get Coordinates of Interest

The first step involves defining the genomic regions of interest for each gene. This script allows you to specify:
- Custom window size around genes
- Transcription start site (TSS) regions
- Gene body regions

```bash
./get_gene_coords.sh -g <gene_list> \
                     -a <gencode_annotations> \
                     -r <region_type: tss/genebody> \
                     -w <window_size> \
                     -o <output_directory>
```

#### Parameters:
- `-g`: Input gene list file
- `-a`: GENCODE annotation file path
- `-r`: Region type (tss or genebody)
- `-w`: Window size in base pairs
- `-o`: Output directory path

## Documentation

Detailed documentation and tutorials are currently under development. Please check back soon for:
- Step-by-step tutorials
- Example workflows
- API documentation
- Best practices guide

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use snATAC-Express in your research, please cite our work (citation information will be available upon publication).

## Contact

For questions and support, please open an issue in the GitHub repository or contact the maintainers.

---

*This project is actively maintained. For updates and announcements, please watch this repository.*
