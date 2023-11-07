#!/usr/sbin/anacondag

import argparse
import fnmatch
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
from scipy import sparse, io
import statsmodels.api as sm
import sys

import warnings
from sklearn.exceptions import DataConversionWarning

random.seed(12345)

# 10-12-2023
# This script runs many predictive models.
# Example for how to run:
# python ./run_multitest.py  -g genelist_genebody.txt -gex sparse_gex_matrix.txt -pks sparse_peak_matrix.txt -pb 1 -out ./Results

# import custom functions
os.chdir("/storage/home/mfisher42/scProjects/Predict_GEX/Multitest_kfoldcv_95featselect_hyperparam_10312023")
from data_preprocessing import get_pseudobulk, load_peak_input, subset_peaks, load_gex_input, subset_gex, make_all_pseudobulk
from model_builders import build_RFR_model, build_LR_model, build_XGB_model, build_LGBM_model
from feature_selection import feature_selector

# ============================================
def parse_my_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g", "--gene_list", type = str, help = "gene list")
    parser.add_argument("-gex", "--gex_matrix", type = str, help = "sparse gex matrix file")
    parser.add_argument("-pks", "--peak_matrix", type = str, help = "sparse peak matrix file")
    parser.add_argument("-pb", "--pseudobulk_replicate", type = str, help = "pseudobulk replicate version: 1 or 2")
    parser.add_argument("-out", "--output_dir", type = str, help = "output directory path")
    return vars(parser.parse_args())

# ============================================
def build_models(gene):
    global peak_df, gex_df, pb_keep, outdir, alpha_summary
    print("Extracting information for " + gene)
    window = genes_df.loc[genes_df["gene"] == gene, "window"].iloc[0]
    # make output directory for gene
    gene_outdir = outdir + gene
    if not os.path.exists(gene_outdir):
        os.makedirs(gene_outdir)
    # subset gene and region from peak_df and gex_df
    gene_peaks = subset_peaks(peak_df, window)
    gene_exp = subset_gex(gex_df, gene)
    # get pseudobulk values for gene/region
    pb_peak_df, gex_peak_df = make_all_pseudobulk(gene_peaks, gene_exp, gene, pb_keep, outdir, peak_df, gex_df)
    # shuffle GEX values for permutation testing
    # TBD
    # Filter peaks:
    peak_set = pb_peak_df
    #filt10perc_peaks = pb_peak_df.loc[pb_peak_df[pb_peak_df.columns].ne(0).sum(axis=1) >= len(pb_peak_df.columns)*.1]
    #filt50perc_peaks = pb_peak_df.loc[pb_peak_df[pb_peak_df.columns].ne(0).sum(axis=1) >= len(pb_peak_df.columns)*.5]
    # For each set of peaks, run models:
    #peakset_list = [all_peaks, filt10perc_peaks, filt50perc_peaks]
    if len(peak_set) < 3:
        print("DataFrame has less than 3 peaks. Exiting function.")
        return
    elif (gex_peak_df.max().max() == 0) or np.isnan(gex_peak_df.max().max()):
        print("Max gene expression value is 0. Exiting function.")
    else:
        # 5.3) implement random forest classifier; rerank after each built model
        test = "rf_ranker"
        build_RFR_model(peak_set, gex_peak_df, gene, gene_outdir, test)
        test = "perm_ranker"
        build_RFR_model(peak_set, gex_peak_df, gene, gene_outdir, test)
        test = "dropcol_ranker"
        build_RFR_model(peak_set, gex_peak_df, gene, gene_outdir, test)
        # 5.4) build linear regression models; rerank after each built model
        test = "perm_ranker"
        build_LR_model(peak_set, gex_peak_df, gene, gene_outdir, test)
        test = "dropcol_ranker"
        build_LR_model(peak_set, gex_peak_df, gene, gene_outdir, test)
        # 5.5) implement XGBoost; rerank after each built model
        test = "xgb_ranker"
        build_XGB_model(peak_set, gex_peak_df, gene, gene_outdir, test)
        test = "perm_ranker"
        build_XGB_model(peak_set, gex_peak_df, gene, gene_outdir, test)
        test = "dropcol_ranker"
        build_XGB_model(peak_set, gex_peak_df, gene, gene_outdir, test)
        # 5.6) implement LightGBM; rerank after each model built
        test = "lgbm_ranker"
        build_LGBM_model(peak_set, gex_peak_df, gene, gene_outdir, test)
        test = "perm_ranker"
        build_LGBM_model(peak_set, gex_peak_df, gene, gene_outdir, test)
        test = "dropcol_ranker"
        build_LGBM_model(peak_set, gex_peak_df, gene, gene_outdir, test)
        # 5.7) Feature selection
        summary = feature_selector(gene, gene_outdir)
        alpha_summary = pd.concat([alpha_summary, summary], ignore_index = True)
        alpha_summary.to_csv("/storage/home/mfisher42/scProjects/Predict_GEX/Multitest_kfoldcv_95featselect_hyperparam_10312023/alpha_summary.csv", index=False)
    return alpha_summary

# ============================================
if __name__ == "__main__":
    os.chdir("/storage/home/mfisher42/scProjects/Predict_GEX/Multitest_kfoldcv_95featselect_hyperparam_10312023/")
    # 1.) parse arguments
    args = parse_my_args()
    gene_list = args["gene_list"]
    gex_matrix = args["gex_matrix"]
    peak_matrix = args["peak_matrix"]
    pseudobulk_replicate = args["pseudobulk_replicate"]
    outdir = args["output_dir"]
    outdir = "/storage/home/mfisher42/scProjects/Predict_GEX/Multitest_kfoldcv_95featselect_hyperparam_10312023/Results/"
    # 2.) load/fix/format peaks
    print("Loading ATAC peaks... this may take a few minutes.")
    peak_df = load_peak_input(peak_matrix)
    # 3.) load/fix/format gex
    print("Loading gene expression... this may take a few minutes.")
    gex_df = load_gex_input(gex_matrix)
    # 4.) get pseudbulk ID values for selected replicate:
    print("Data loaded!")
    pb_keep = get_pseudobulk(pseudobulk_replicate)
    # 5.) For each gene, extract values, make pseudobulk, run models:
    # intiate summary output
    columnnames = ["gene", "celltype", "method", "peak_filter", "npeaks_kept", "cv_R2"]
    alpha_summary = pd.DataFrame(columns = columnnames)
    # load gene list
    genes_df = pd.read_csv(gene_list, sep = "\t")
    genes_df.columns = ["gene", "window"]
    # Build models; run in parallel
    gene = [x for x in genes_df["gene"]]
    num_processes = 1
    pool = multiprocessing.Pool(processes=num_processes)
    pool.map(build_models, gene)
    pool.close()
    pool.join()
