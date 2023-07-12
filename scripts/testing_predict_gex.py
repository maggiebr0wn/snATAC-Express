#!/usr/sbin/anaconda

import argparse
import h5py
import multiprocessing
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from scipy import sparse, io
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
import sys

### 07-07-2023 ###
# This script runs random forest and linear regression models on each gene
# This is modified from an origanl script from Oct 2023 with the modification 
# of running a backwards stepwise and ranking peak importances after each 
# model is built based on how much the R2 value changes with the presence or 
# absense of a peak.
#
# This script takes in the following:
#   1.) gene_list: this is a tab delimited text file containing the genenames and windows for peak selection
#   2.) GEX matrix file
#   3.) Peak matrix file
#   4.) Pseudobulk replicated version, choose option (1 or 2)
#   5.) output_dir: this is the output directory path with which to write to
#
# example:
# python ./predict_gex.py -g Run_1_10202022/genelist_tss.txt -gex /storage/home/mfisher42/scProjects/Predict_GEX/input_data/sparse_gex_matrix.txt -pks "/storage/home/mfisher42/scProjects/Predict_GEX/input_data/sparse_peak_matrix.txt" -pb 1 -out Run_1_10202022

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
def make_pseudobulk(gene_peaks, gene_exp, pb_keep):
    pb_peak_df = pd.DataFrame()
    gex_peak_df = pd.DataFrame()
    # iterative through pseudobulk groups
    for pb_group in pb_keep.PB_Name:
        cellnames = eval(pb_keep[pb_keep.PB_Name == pb_group].CellNames.tolist()[0])
        # extract pb_group from peak_mat, average peak values
        peak_subset = gene_peaks[cellnames].mean(axis = 1).to_frame()
        peak_subset.columns = [pb_group]
        pb_peak_df = pd.concat([pb_peak_df, peak_subset], axis = 1)
        # extract pb_group from gex_mat, average expression values
        gex_subset = gene_exp[cellnames].mean(axis = 1).to_frame()
        gex_subset.columns = [pb_group]
        gex_peak_df = pd.concat([gex_peak_df, gex_subset], axis=1)
    return pb_peak_df, gex_peak_df

# ============================================
def Runfun(gene):
    global gex_df, peak_df, genes_df, pb_keep, outdir
    print("Running models for gene: " + gene)
    window = genes_df.loc[genes_df["gene"] == gene, "window"].iloc[0]
    # make output directory for gene
    if not os.path.exists("/storage/home/mfisher42/scProjects/Predict_GEX/Feature_Importance_07062023/Results/" + gene):
        os.makedirs("/storage/home/mfisher42/scProjects/Predict_GEX/Feature_Importance_07062023/Results/" + gene)
    # 5.1) subset gene and region from peak_df and gex_df:
    gene_peaks = subset_peaks(peak_df, window)
    gene_exp = subset_gex(gex_df, gene)
    # 5.2) get pseudobulk values for gene/region
    pb_peak_df, pb_gex_df = make_pseudobulk(gene_peaks, gene_exp, pb_keep)
    # 5.3) only keep peaks that exist in at least 10% of pseudobulk replicates
    final_pb_peak_df = pb_peak_df.loc[pb_peak_df[pb_peak_df.columns].ne(0).sum(axis=1) >= len(pb_peak_df.columns)*.10]
    # 5.4) implement random forest classifier to select peaks (10% filt peaks)
    # build RF models; rerank after each built model
    test = "rf_ranker"
    build_RFR_model(final_pb_peak_df, pb_gex_df, gene, outdir, test)
    test = "perm_ranker"
    build_RFR_model(final_pb_peak_df, pb_gex_df, gene, outdir, test)
    test = "dropcol_ranker"
    build_RFR_model(final_pb_peak_df, pb_gex_df, gene, outdir, test)
    # build linear regression models; rerank after each built model
    test = "perm_ranker"
    build_LinReg_model(final_pb_peak_df, pb_gex_df, gene, outdir, test)
    test = "dropcol_ranker"
    build_LinReg_model(final_pb_peak_df, pb_gex_df, gene, outdir, test)
    # 5.5) determine best set of features for each model
    feature_selector(gene, outdir)

# ============================================
if __name__ == "__main__":
    # import custom functions
    os.chdir("/storage/home/mfisher42/scProjects/Predict_GEX/Feature_Importance_07062023")
    from feature_selection import rf_ranker, perm_ranker, RF_dropcolumn_importance, LinReg_dropcolumn_importance, feature_selector
    from model_builders import build_RFR_model, build_LinReg_model 
    from assess_RF_params import grid_search_init
    # 1.) parse arguments
    args = parse_my_args()
    gene_list = args["gene_list"]
    gex_matrix = args["gex_matrix"]
    peak_matrix = args["peak_matrix"]
    pseudobulk_replicate = args["pseudobulk_replicate"]
    outdir = args["output_dir"]
    outdir = "/storage/home/mfisher42/scProjects/Predict_GEX/Feature_Importance_07062023/Results"
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
    genes_df = pd.read_csv(gene_list, sep = "\t")
    genes_df.columns = ["gene", "window"]
    test_gene_list = ["IL6ST", "LYST", "MS4A1", "NCOA5", "NR4A2", "NUP98", "RAB3A", "SYN1", "TNFRSF13C", "UBP1"]
    # run in parallel
    num_processes = 10  # Specify the number of genes to run in parallel (each run ~8% memory)
    pool = multiprocessing.Pool(processes=num_processes)
    pool.map(Runfun, test_gene_list)
    pool.close() # No more tasks will be submitted to the pool
    pool.join() # Wait for all processes to complete
    # 6.) for each gene and model, assess stability of model
    for gene in test_gene_list:
        # get selected_peaks.csv file
        selected_peaks = pd.read_csv(outdir + "/" + gene + "/selected_peaks.csv")
        test_list = selected_peaks["Result"]
        for test in test_list: 
            if "_RFR_" in test:
                print("Interrogating " + test)
                test_name = test.split("_")[2] + "_" + test.split("_")[3]
                npeaks = selected_peaks.loc[selected_peaks["Result"] == test, "nPeaks"].item()
                # load peak list
                peak_filename = gene + "_" + npeaks + "peaks_rfranker_importance.csv"
                peak_list = pd.read_csv(outdir + "/" + gene + "/" + test_name + "/" 
                grid_search_init(final_pb_peak_df, pb_gex_df, gene, test, outdir)


