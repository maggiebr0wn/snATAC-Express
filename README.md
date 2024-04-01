# scATAC-Express

***Manuscript in prep***
----------
OBJECTIVE
----------
<img src="https://github.com/maggiebr0wn/ATAC-Express/blob/main/atac-express.jpg" align = "right" width = 500, height = 300>

The objective of this project is to use scATAC-seq data to predict gene expression data, in addition to identifying which peaks are most important for accurate gene expression prediction.

## Requirements

### Input data
<li> scATAC-seq peak matrix </li>
<li> scRNA-seq raw counts matrix </li>

-------------------
STEPS & HOW TO RUN
-------------------

Clone this repository, then enter it and download the gencode.v41.annotation.gtf.gz file:

    cd scATAC-Express
    wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.annotation.gtf.gz
    gunzip gencode.v41.annotation.gtf.gz


Vignette and practice data in progress;;
