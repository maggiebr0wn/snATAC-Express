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

### **Step 1:** Get coordinates of interest for each gene

This script obtains a specified cis-regulatoru window around a gene depending on the user's input. The user can specify whether they want to choose a window of any size, around a gene's transciption start site or gene body. Outputs a file called gene_list.txt which lists each gene and the defined region's coordinates.

    ./get_gene_coords.sh -g <gene_list> -a <gencode annotations> -r <region type: tss/genebody> -w <window size> -o <output dir>




Vignette and practice data in progress;;
