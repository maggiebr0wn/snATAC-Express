#!/usr/sbin/anaconda

"""
Data Preprocessing Module for snATAC-Express

This module handles the preprocessing of single-cell ATAC-seq and RNA-seq data,
including pseudobulking, normalization, and data formatting for downstream analysis.
"""

from typing import Dict, List, Tuple, Union, Optional
import argparse
import fnmatch
import h5py
import seaborn as sns
import math
from math import log2
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
from scipy import sparse, io
import statsmodels.api as sm
import sys
from pathlib import Path

# Import configuration
from config import (
    GROUP_COVERAGES,
    PEAK_MATRIX,
    PEAK_ROWNAMES,
    PEAK_COLNAMES,
    GEX_MATRIX,
    GEX_ROWNAMES,
    GEX_COLNAMES,
    MIN_CELLS_PER_PSEUDOBULK,
    ensure_directories,
    get_gene_output_dir
)

def get_pseudobulk(pseudobulk_replicate: str) -> pd.DataFrame:
    """
    Get pseudobulk information for a specific replicate.

    Args:
        pseudobulk_replicate: String indicating the replicate number (e.g., "1" or "2")

    Returns:
        DataFrame containing pseudobulk information filtered for:
        - Specified replicate
        - Groups with at least MIN_CELLS_PER_PSEUDOBULK cells
    """
    pb_info = pd.read_csv(GROUP_COVERAGES, sep=",")
    rep = str("Rep" + pseudobulk_replicate)
    pb_rep = pb_info[pb_info.PB_Name.str.contains(rep)]
    pb_keep = pb_rep[pb_rep["CellNames"].str.len()/29 >= MIN_CELLS_PER_PSEUDOBULK]
    return pb_keep

def load_peak_input(peak_matrix: str) -> pd.DataFrame:
    """
    Load and format peak accessibility matrix.

    Args:
        peak_matrix: Path to the sparse peak matrix file

    Returns:
        DataFrame containing peak accessibility data with:
        - Rows: Genomic regions (peaks)
        - Columns: Cell barcodes
    """
    sparse_peak_matrix = io.mmread(peak_matrix)
    sparse_peak_matrix = sparse_peak_matrix.astype(np.uint8)  # memory efficient datatype
    pm_dense = sparse_peak_matrix.toarray()
    coords = np.genfromtxt(PEAK_ROWNAMES, dtype=str)
    col_names = np.genfromtxt(PEAK_COLNAMES, dtype=str, comments="+")
    peak_df = pd.DataFrame(pm_dense, columns=col_names, index=coords)
    return peak_df

def subset_peaks(peak_df: pd.DataFrame, window: str) -> pd.DataFrame:
    """
    Subset peaks to a specific genomic window.

    Args:
        peak_df: DataFrame containing peak accessibility data
        window: Genomic window in format "chr:start-end"

    Returns:
        DataFrame containing only peaks within the specified window
    """
    # Parse peak coordinates
    rownames = peak_df.index.to_list()
    chr = [i.split(":", 1)[0] for i in rownames]
    region = [i.split(":", 1)[1] for i in rownames]
    start = [i.split("-", 1)[0] for i in region]
    end = [i.split("-", 1)[1] for i in region]
    
    # Add coordinate columns
    peak_df["chr"] = chr
    peak_df["start"] = start
    peak_df["end"] = end
    
    # Parse window coordinates
    chr = window.split(":")[0]
    start = (window.split(":")[1]).split("-")[0]
    stop = int((window.split(":")[1]).split("-")[1])
    
    # Subset peaks
    if len(start) > 0:
        start = int(start)
    else:
        start = 0
    
    gene_peaks = peak_df[
        (peak_df["chr"] == chr) & 
        (peak_df["start"].astype(int) >= start) & 
        (peak_df["end"].astype(int) <= stop)
    ]
    return gene_peaks

def load_gex_input(gex_matrix: str) -> pd.DataFrame:
    """
    Load and format gene expression matrix.

    Args:
        gex_matrix: Path to the sparse gene expression matrix file

    Returns:
        DataFrame containing gene expression data with:
        - Rows: Genes
        - Columns: Cell barcodes
    """
    sparse_gex_matrix = io.mmread(gex_matrix)
    sparse_gex_matrix = sparse_gex_matrix.astype(np.uint8)  # memory efficient datatype
    gm_dense = sparse_gex_matrix.toarray()
    genes = np.genfromtxt(GEX_ROWNAMES, dtype=str)
    col_names = np.genfromtxt(GEX_COLNAMES, dtype=str, comments="+")
    gex_df = pd.DataFrame(gm_dense, columns=col_names, index=genes)
    return gex_df

def subset_gex(gex_df: pd.DataFrame, gene: str) -> pd.DataFrame:
    """
    Subset gene expression data for a specific gene.

    Args:
        gex_df: DataFrame containing gene expression data
        gene: Name of the target gene

    Returns:
        DataFrame containing expression data for the specified gene
    """
    gex_df["gene"] = gex_df.index
    gene_exp = gex_df[gex_df["gene"] == gene]
    return gene_exp

def make_all_pseudobulk(
    gene_peaks: pd.DataFrame,
    gene_exp: pd.DataFrame,
    gene: str,
    pb_keep: pd.DataFrame,
    outdir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create pseudobulked matrices for both peak accessibility and gene expression.

    This function:
    1. Aggregates single-cell data into pseudobulks
    2. Normalizes the data (CPM and log2 transformation)
    3. Saves the processed matrices

    Args:
        gene_peaks: DataFrame containing peak accessibility data
        gene_exp: DataFrame containing gene expression data
        gene: Name of the target gene
        pb_keep: DataFrame containing pseudobulk information
        outdir: Output directory for saving processed data

    Returns:
        Tuple containing:
        - DataFrame of pseudobulked peak accessibility data
        - DataFrame of pseudobulked gene expression data
    """
    pb_peak_df = pd.DataFrame()
    gex_peak_df = pd.DataFrame()
    
    # Iterate through pseudobulk groups
    for pb_group in pb_keep.PB_Name:
        cellnames = eval(pb_keep[pb_keep.PB_Name == pb_group].CellNames.tolist()[0])
        
        # Aggregate peak accessibility
        peak_subset = gene_peaks[cellnames].sum(axis=1).to_frame()
        peak_subset.columns = [pb_group]
        pb_peak_df = pd.concat([pb_peak_df, peak_subset], axis=1)
        
        # Aggregate gene expression
        gex_subset = gene_exp[cellnames].sum(axis=1).to_frame()
        gex_subset.columns = [pb_group]
        gex_peak_df = pd.concat([gex_peak_df, gex_subset], axis=1)
    
    # Normalize matrices
    peaks_cpm = pb_peak_df/pb_peak_df.values.sum() * 1000000
    peaks_pseudobulk = peaks_cpm.applymap(lambda x: math.log2(x + 1))
    gex_cpm = gex_peak_df/gex_peak_df.values.sum() * 1000000
    gex_pseudobulk = gex_cpm.applymap(lambda x: math.log2(x + 1))
    
    # Save processed matrices
    gene_dir = get_gene_output_dir(gene)
    peaks_filename = gene_dir / "peaks.csv"
    gex_filename = gene_dir / "gex.csv"
    
    peaks_pseudobulk.to_csv(peaks_filename, index=True)
    gex_pseudobulk.to_csv(gex_filename, index=True)
    
    return pb_peak_df, gex_peak_df
