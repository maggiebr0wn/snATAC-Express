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
# New helper to resolve paths relative to a base directory
DEFAULT_INPUT_DIR = os.getenv("SNATAC_EXPRESS_INPUT", os.getcwd())

def _resolve_path(filename: str, input_dir: str = None):
    """Return `filename` if it is an absolute path otherwise
    join it to the provided `input_dir` or to the DEFAULT_INPUT_DIR.
    Raises FileNotFoundError if the resulting path does not exist."""
    if os.path.isabs(filename):
        path = filename
    else:
        path = os.path.join(input_dir or DEFAULT_INPUT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


# ============================================
def get_pseudobulk(pseudobulk_replicate, group_coverages_csv="group_coverages.csv", input_dir=None):
    """Return DataFrame filtered for the requested replicate. The `group_coverages_csv`
    can be given as an absolute path or a filename relative to `input_dir`."""
    pb_path = _resolve_path(group_coverages_csv, input_dir)
    pb_info = pd.read_csv(pb_path, sep=",")
    rep = f"Rep{pseudobulk_replicate}"
    pb_rep = pb_info[pb_info.PB_Name.str.contains(rep)]
    pb_keep = pb_rep[pb_rep["CellNames"].str.len()/29 >= 10]
    return pb_keep


# ============================================
def load_peak_input(peak_matrix, input_dir=None):
    """Load peak matrix (MM coordinates). Expects row/col name helper files in the
    same directory as `peak_matrix` or relative to `input_dir`."""
    peak_matrix = _resolve_path(peak_matrix, input_dir)
    sparse_peak_matrix = io.mmread(peak_matrix).astype(np.uint8)
    pm_dense = sparse_peak_matrix.toarray()
    # Infer directory for annotation files
    base_dir = os.path.dirname(peak_matrix) if input_dir is None else input_dir
    coords = np.genfromtxt(_resolve_path("sparse_peak_matrix_rownames.txt", base_dir), dtype=str)
    col_names = np.genfromtxt(_resolve_path("sparse_peak_matrix_colnames.txt", base_dir), dtype=str, comments="+")
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
def load_gex_input(gex_matrix, input_dir=None):
    gex_matrix = _resolve_path(gex_matrix, input_dir)
    sparse_gex_matrix = io.mmread(gex_matrix).astype(np.uint8)
    gm_dense = sparse_gex_matrix.toarray()
    base_dir = os.path.dirname(gex_matrix) if input_dir is None else input_dir
    genes = np.genfromtxt(_resolve_path("sparse_gex_matrix_rownames.txt", base_dir), dtype=str)
    col_names = np.genfromtxt(_resolve_path("sparse_gex_matrix_colnames.txt", base_dir), dtype=str, comments="+")
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


# ============================================
def load_independent_peaks(test_peak_matrix, input_dir=None):
    test_peak_matrix = _resolve_path(test_peak_matrix, input_dir)
    sparse_peak_matrix = io.mmread(test_peak_matrix).astype(np.uint8)
    pm_dense = sparse_peak_matrix.toarray()
    base_dir = os.path.dirname(test_peak_matrix) if input_dir is None else input_dir
    coords = np.genfromtxt(_resolve_path("sparse_peak_matrix_rownames.txt", base_dir), dtype=str)
    col_names = np.genfromtxt(_resolve_path("sparse_peak_matrix_colnames.txt", base_dir), dtype=str, comments="+")
    test_peak_df = pd.DataFrame(pm_dense, columns=col_names, index=coords)
    return test_peak_df


# ============================================
def load_independent_gex_(test_gex_matrix, input_dir=None):
    test_gex_matrix = _resolve_path(test_gex_matrix, input_dir)
    sparse_gex_matrix = io.mmread(test_gex_matrix).astype(np.uint8)
    gm_dense = sparse_gex_matrix.toarray()
    base_dir = os.path.dirname(test_gex_matrix) if input_dir is None else input_dir
    genes = np.genfromtxt(_resolve_path("sparse_gex_matrix_rownames.txt", base_dir), dtype=str)
    col_names = np.genfromtxt(_resolve_path("sparse_gex_matrix_colnames.txt", base_dir), dtype=str, comments="+")
    test_gex_df = pd.DataFrame(gm_dense, columns=col_names, index=genes)
    return test_gex_df


# ============================================
def get_independent_pseudobulk(pseudobulk_replicate):
    # pseduobulk info; filter for >=10 cells and Rep1 or Rep2
    GroupCoveragesDir = "test_data/group-coverages"
    cols = ["PB_Name", "CellNames"]
    PB_list = []
    for filename in os.listdir(GroupCoveragesDir):
        file = os.path.join(GroupCoveragesDir, filename)
        print(file)
        # read in file
        f = h5py.File(file, "r")
        # pseudobulkname
        pbname = file.split("/")[-1]
        # get cellnames
        group = f["Coverage"]
        info = group["Info"]
        cellnames = info["CellNames"]
        cellnames_final = [x.decode("utf-8") for x in cellnames[()]]
        # append to PB_list
        PB_list.append([pbname, cellnames_final])
    PB_df = pd.DataFrame(PB_list, columns = cols)
    #pb_info = pd.read_csv("/storage/home/mfisher42/scProjects/Predict_GEX/input_data/group_coverages.csv", sep = ",")
    rep = str("Rep" + pseudobulk_replicate)
    pb_rep = PB_df[PB_df.PB_Name.str.contains(rep)]
    pb_test = pb_rep[pb_rep["CellNames"].str.len()/29 >= 10] # 29 is length of each cell barcode; keep min 10 cells per PB
    return pb_test

