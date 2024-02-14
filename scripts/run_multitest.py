#!/usr/sbin/anaconda

import argparse
import fnmatch
import math
import numpy as np
import os
import pandas as pd
import random
from scipy import sparse, io
import statsmodels.api as sm
import sys

random.seed(12345)

# 10-12-2023
# This script runs many predictive models.
# Example for how to run:
# python ./run_multitest.py  -g genelist_genebody.txt -n <gene name> -gex sparse_gex_matrix.txt -pks sparse_peak_matrix.txt -pb 1 -f 10 -out ./Results

# import custom functions
os.chdir("/storage/home/mfisher42/scProjects/Predict_GEX/Multitest_kfoldcv_95featselect_hyperparam_10312023")
from data_preprocessing import get_pseudobulk, load_peak_input, subset_peaks, load_gex_input, subset_gex, make_all_pseudobulk
from model_builders import build_RFR_model, build_LR_model, build_XGB_model, build_LGBM_model
from feature_selection import feature_selector

# ============================================
def parse_my_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g", "--gene_list", type = str, help = "gene list")
    parser.add_argument("-n", "--gene_name", type = str, help = "gene name")
    parser.add_argument("-gex", "--gex_matrix", type = str, help = "sparse gex matrix file")
    parser.add_argument("-pks", "--peak_matrix", type = str, help = "sparse peak matrix file")
    parser.add_argument("-pb", "--pseudobulk_replicate", type = str, help = "pseudobulk replicate version: 1 or 2")
    parser.add_argument("-f", "--peak_filter", type = str, help = "peaks must be in at least X% of samples (1 to 100)")
    parser.add_argument("-out", "--output_dir", type = str, help = "output directory path")
    return vars(parser.parse_args())

# ============================================
def build_models(gene):
    global peak_df, gex_df, genes_df, pb_keep, peak_filter, outdir
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
    # Filter peaks:
    filt = int(peak_filter)/100
    peak_set = pb_peak_df.loc[pb_peak_df[pb_peak_df.columns].ne(0).sum(axis=1) >= len(pb_peak_df.columns)*filt]
    # Run models:
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

# ============================================
if __name__ == "__main__":
    os.chdir("/storage/home/mfisher42/scProjects/Predict_GEX/Multitest_kfoldcv_95featselect_hyperparam_10312023/")
    # 1.) parse arguments
    args = parse_my_args()
    gene_list = args["gene_list"]
    gex_matrix = args["gex_matrix"]
    peak_matrix = args["peak_matrix"]
    peak_filter = args["peak_filter"]
    pseudobulk_replicate = args["pseudobulk_replicate"]
    outdir = args["output_dir"]
    # 2.) load/fix/format peaks
    print("Loading ATAC peaks... this may take a few minutes.")
    peak_df = load_peak_input(peak_matrix)
    # 3.) load/fix/format gex
    print("Loading gene expression... this may take a few minutes.")
    gex_df = load_gex_input(gex_matrix)
    # 4.) get pseudbulk ID values for selected replicate:
    print("Data loaded!")
    pb_keep = get_pseudobulk(pseudobulk_replicate)
    # 5.) For gene, extract values, make pseudobulk, run models:
    # load gene list
    genes_df = pd.read_csv(gene_list, sep = "\t")
    genes_df.columns = ["gene", "window"]
    # Build models
    print(gene)
    build_models(gene)
