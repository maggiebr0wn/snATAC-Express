#!/usr/sbin/anaconda

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

# ============================================
def get_pseudobulk(pseudobulk_replicate):
    # pseduobulk info; filter for >=10 cells and Rep1 or Rep2
    pb_info = pd.read_csv("/storage/home/mfisher42/scProjects/Predict_GEX/input_data/group_coverages.csv", sep = ",")
    rep = str("Rep" + pseudobulk_replicate)
    pb_rep = pb_info[pb_info.PB_Name.str.contains(rep)]
    pb_keep = pb_rep[pb_rep["CellNames"].str.len()/29 >= 10] # 29 is length of each cell barcode; keep min 10 cells per PB
    return pb_keep

# ============================================
def load_peak_input(peak_matrix):
    ## load input, format into DF for gene of interest
    sparse_peak_matrix = io.mmread(peak_matrix) # this step takes a few minutes
    sparse_peak_matrix = sparse_peak_matrix.astype(np.uint8) # mem efficient datatype
    pm_dense = sparse_peak_matrix.toarray()
    coords = np.genfromtxt("/storage/home/mfisher42/scProjects/Predict_GEX/input_data/sparse_peak_matrix_rownames.txt", dtype=str)
    col_names = np.genfromtxt("/storage/home/mfisher42/scProjects/Predict_GEX/input_data/sparse_peak_matrix_colnames.txt", dtype=str, comments = "+")
    peak_df = pd.DataFrame(pm_dense, columns=col_names, index=coords)
    return peak_df
    
# ============================================
def subset_peaks(peak_df, window):
    ## break up regions into columns for peak_df
    rownames = peak_df.index.to_list()
    chr = [i.split(":", 1)[0] for i in rownames]
    region = [i.split(":", 1)[1] for i in rownames]
    start = [i.split("-", 1)[0] for i in region]
    end = [i.split("-", 1)[1] for i in region]
    peak_df["chr"] = chr
    peak_df["start"] = start
    peak_df["end"] = end
    ## get peaks for gene of interest
    chr = window.split(":")[0]
    start = (window.split(":")[1]).split("-")[0]
    stop = int((window.split(":")[1]).split("-")[1])
    if len(start) > 0:
        ## extract peaks in window for this gene
        start = int((window.split(":")[1]).split("-")[0])
        gene_peaks = peak_df[(peak_df["chr"] == chr) & (peak_df["start"].astype(int) >= start) & (peak_df["end"].astype(int) <= stop)]
        return gene_peaks
    else:
        start = 0
        ## extract peaks in window for this gene
        gene_peaks = peak_df[(peak_df["chr"] == chr) & (peak_df["start"].astype(int) >= start) & (peak_df["end"].astype(int) <= stop)]
        return gene_peaks

# ============================================
def load_gex_input(gex_matrix):
    ## load input, format into DF for gene of interest
    sparse_gex_matrix = io.mmread(gex_matrix) # this step takes a few minutes
    sparse_gex_matrix = sparse_gex_matrix.astype(np.uint8) # mem efficient datatype
    gm_dense = sparse_gex_matrix.toarray()
    genes = np.genfromtxt("/storage/home/mfisher42/scProjects/Predict_GEX/input_data/sparse_gex_matrix_rownames.txt", dtype=str)
    col_names = np.genfromtxt("/storage/home/mfisher42/scProjects/Predict_GEX/input_data/sparse_gex_matrix_colnames.txt", dtype=str, comments = "+")
    gex_df = pd.DataFrame(gm_dense, columns=col_names, index=genes)
    return gex_df

# ============================================
def subset_gex(gex_df, gene):
    ## subset gex for each gene
    gex_df["gene"] = gex_df.index
    gene_exp = gex_df[gex_df["gene"] == gene]
    return gene_exp
  
# ============================================
def make_all_pseudobulk(gene_peaks, gene_exp, gene, pb_keep, outdir, peak_df, gex_df):
    pb_peak_df = pd.DataFrame()
    gex_peak_df = pd.DataFrame()
    # iterative through pseudobulk groups
    for pb_group in pb_keep.PB_Name:
        cellnames = eval(pb_keep[pb_keep.PB_Name == pb_group].CellNames.tolist()[0])
        # extract pb_group from peak_mat, sum peak values
        peak_subset = gene_peaks[cellnames].sum(axis = 1).to_frame()
        peak_subset.columns = [pb_group]
        pb_peak_df = pd.concat([pb_peak_df, peak_subset], axis = 1)
        # extract pb_group from gex_mat, average expression values
        gex_subset = gene_exp[cellnames].sum(axis = 1).to_frame()
        gex_subset.columns = [pb_group]
        gex_peak_df = pd.concat([gex_peak_df, gex_subset], axis=1)
    ## normalize matrices
    # get total counts for pseudobulks, divide each feature by total counts
    peaks_cpm = pb_peak_df/pb_peak_df.values.sum() * 1000000
    peaks_pseudobulk = peaks_cpm.applymap(lambda x: math.log2(x + 1))
    gex_cpm = gex_peak_df/gex_peak_df.values.sum() * 1000000
    gex_pseudobulk = gex_cpm.applymap(lambda x: math.log2(x + 1))
    # save pseudobulk peak matrices
    peaks_filename = outdir + "/" + gene + "/" + "peaks.csv"
    peaks_pseudobulk.to_csv(peaks_filename, index=True)
    # save pseudobulk gex matrices
    gex_filename = outdir + "/" + gene + "/" + "gex.csv"
    gex_pseudobulk.to_csv(gex_filename, index=True)
    return pb_peak_df, gex_peak_df
