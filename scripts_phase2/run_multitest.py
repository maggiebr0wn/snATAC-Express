#!/usr/sbin/anaconda


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


# import custom functions
os.chdir("/storage/home/hcoda1/6/mfisher42/scratch/scATAC_Express/SLE_Genes_02062024/Multitest_kfoldcv_Aggreg95featselect_hyperparam_10perc_parallel_02132024")
from data_preprocessing import get_pseudobulk, load_peak_input, subset_peaks, load_gex_input, subset_gex, make_all_pseudobulk
from model_builders import build_RFR_model, build_LR_model, build_XGB_model, build_LGBM_model
from feature_selection import feature_selector


# ============================================
def parse_my_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g", "--gene_list", type = str, help = "gene list")
    parser.add_argument("-n", "--gene_name", type = str, help = "gene name")
    parser.add_argument("-agg", "--aggregated_peaks", type = str, help = "path to dir with aggregated peak scores")
    parser.add_argument("-out", "--output_dir", type = str, help = "output directory path")
    return vars(parser.parse_args())


# ============================================
def build_models(gene):
    global outdir, aggregated_peaks_dir
    print("Extracting information for " + gene)
    window = genes_df.loc[genes_df["gene"] == gene, "window"].iloc[0]
    # make output directory for gene
    gene_outdir = outdir + gene
    if not os.path.exists(gene_outdir):
        os.makedirs(gene_outdir)
    # get peak and gex df from previous run
    gene_peaks = pd.read_csv(aggregated_peaks_dir + "/Results/" + gene + "/peaks.csv", sep = ",", index_col = 0)
    gex_peak_df = pd.read_csv(aggregated_peaks_dir + "/Results/" + gene + "/gex.csv", sep = ",", index_col = 0)
    # if gene has aggregated results, then run:
    file_path = aggregated_peaks_dir + "/Results/" + gene + "/aggregated_peak_importances_inclLR.csv"
    if os.path.exists(file_path):
        # get aggregaed peaks, find cumulatively top 95% important peals
        aggregated_ranks = pd.read_csv(aggregated_peaks_dir + "/Results/" + gene + "/aggregated_peak_importances_inclLR.csv", sep = ",", index_col = 0)
        # find top 95% cumulatively important features
        min_zscore = aggregated_ranks["Average_Zscore"].min()
        aggregated_ranks["Adjusted_Zscore"] = aggregated_ranks["Average_Zscore"] + abs(min_zscore)
        df_sorted = aggregated_ranks.sort_values(by = "Adjusted_Zscore", ascending = False)
        total_sum = df_sorted["Adjusted_Zscore"].sum()
        threshold = 0.95 * total_sum
        top_95_peaks = df_sorted[df_sorted["Adjusted_Zscore"].cumsum() <= threshold] 
        # subset peak matrix:
        pb_peak_df = gene_peaks[gene_peaks.index.isin(top_95_peaks["Peaks"].tolist())]
        peak_set = pb_peak_df
        # run models
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
    else:
        print("This gene did not pass criteria for modeling.")


# ============================================
if __name__ == "__main__":
    os.chdir("/storage/home/hcoda1/6/mfisher42/scratch/scATAC_Express/SLE_Genes_02062024/Multitest_kfoldcv_Aggreg95featselect_hyperparam_10perc_parallel_02132024/")
    # 1.) parse arguments
    args = parse_my_args()
    gene_list = args["gene_list"]
    gene = args["gene_name"]
    aggregated_peaks_dir = args["aggregated_peaks"]
    outdir = args["output_dir"]
    outdir = "/storage/home/hcoda1/6/mfisher42/scratch/scATAC_Express/SLE_Genes_02062024/Multitest_kfoldcv_Aggreg95featselect_hyperparam_10perc_parallel_02132024/Results/"
    # load gene list
    genes_df = pd.read_csv(gene_list, sep = "\t")
    genes_df.columns = ["gene", "window"]
    # Build models:
    print(gene)
    build_models(gene)
