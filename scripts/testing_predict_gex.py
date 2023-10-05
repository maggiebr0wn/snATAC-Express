#!/usr/sbin/anaconda

import argparse
import fnmatch
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
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

random.seed(12345)

### 08-15-2023 ###
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
# python ./testing_predict_gex.py -g Run_1_10202022/genelist_tss.txt -gex /storage/home/mfisher42/scProjects/Predict_GEX/input_data/sparse_gex_matrix.txt -pks "/storage/home/mfisher42/scProjects/Predict_GEX/input_data/sparse_peak_matrix.txt" -pb 1 -out Results
#
#
# Modifications:
# 1.) split pseudobulks in half, for training and testing, from within the make_pseudobulk() function

input_data_path = "input_data"
output_data_path = "Results"

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
    pb_info = pd.read_csv(input_data_path + "/group_coverages.csv", sep = ",")
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
    coords = np.genfromtxt(input_data_path + "/sparse_peak_matrix_rownames.txt", dtype=str)
    col_names = np.genfromtxt(input_data_path + "/sparse_peak_matrix_colnames.txt", dtype=str, comments = "+")
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
    genes = np.genfromtxt(input_data_path + "sparse_gex_matrix_rownames.txt", dtype=str)
    col_names = np.genfromtxt(input_data_path + "/sparse_gex_matrix_colnames.txt", dtype=str, comments = "+")
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
    global outdir
    training_pb_peak_df = pd.DataFrame()
    training_gex_peak_df = pd.DataFrame()
    testing_pb_peak_df = pd.DataFrame()
    testing_gex_peak_df = pd.DataFrame()
    # iterative through pseudobulk groups; split into testing and training (50/50)
    for pb_group in pb_keep.PB_Name:
        cellnames = eval(pb_keep[pb_keep.PB_Name == pb_group].CellNames.tolist()[0])
        num_cells = len(cellnames)//2
        # divide pseudobulks for training and testing
        training_cellnames = random.sample(cellnames, num_cells)
        testing_cellnames = list(set(cellnames) - set(training_cellnames))
        ## Training pseudobulk
        # extract pb_group from peak_mat, average peak values
        training_peak_subset = gene_peaks[training_cellnames].mean(axis = 1).to_frame()
        training_peak_subset.columns = [pb_group]
        training_pb_peak_df = pd.concat([training_pb_peak_df, training_peak_subset], axis = 1)
        # extract pb_group from gex_mat, average expression values
        training_gex_subset = gene_exp[training_cellnames].mean(axis = 1).to_frame()
        training_gex_subset.columns = [pb_group]
        training_gex_peak_df = pd.concat([training_gex_peak_df, training_gex_subset], axis=1)
        ## Testing pseudobulk
        # extract pb_group from peak_mat, average peak values
        testing_peak_subset = gene_peaks[testing_cellnames].mean(axis = 1).to_frame()
        testing_peak_subset.columns = [pb_group]
        testing_pb_peak_df = pd.concat([testing_pb_peak_df, testing_peak_subset], axis = 1)
        # extract pb_group from gex_mat, average expression values
        testing_gex_subset = gene_exp[testing_cellnames].mean(axis = 1).to_frame()
        testing_gex_subset.columns = [pb_group]
        testing_gex_peak_df = pd.concat([testing_gex_peak_df, testing_gex_subset], axis=1)
    ## normalize matrices
    # TRAINING: get total counts for pseudobulks, divide each feature by total counts
    training_peaks_cpm = training_pb_peak_df/training_pb_peak_df.values.sum() * 1000000
    training_peaks_pseudobulk = training_peaks_cpm.applymap(lambda x: math.log2(x + 1))
    training_gex_cpm = training_gex_peak_df/training_gex_peak_df.values.sum() * 1000000
    training_gex_pseudobulk = training_gex_cpm.applymap(lambda x: math.log2(x + 1))
    # TESTING: get total counts for pseudobulks, divide each feature by total counts
    testing_peaks_cpm = testing_pb_peak_df/training_pb_peak_df.values.sum() * 1000000
    testing_peaks_pseudobulk = testing_peaks_cpm.applymap(lambda x: math.log2(x + 1))
    testing_gex_cpm = testing_gex_peak_df/training_gex_peak_df.values.sum() * 1000000
    testing_gex_pseudobulk = testing_gex_cpm.applymap(lambda x: math.log2(x + 1))
    # save pseudobulk peak matrices
    training_peaks_filename = outdir + "/" + gene + "/" + "training_peaks.csv"
    training_peaks_pseudobulk.to_csv(training_peaks_filename, index=False)
    testing_peaks_filename = outdir + "/" + gene + "/" + "testing_peaks.csv"
    testing_peaks_pseudobulk.to_csv(testing_peaks_filename, index=False)
    # save pseudobulk peak matrices
    training_peaks_filename = outdir + "/" + gene + "/" + "training_peaks.csv"
    training_pb_peak_df.to_csv(training_peaks_filename, index=True)
    testing_peaks_filename = outdir + "/" + gene + "/" + "testing_peaks.csv"
    testing_pb_peak_df.to_csv(testing_peaks_filename, index=True)
    # save pseudobulk gex matrices
    training_gex_filename = outdir + "/" + gene + "/" + "training_gex.csv"
    training_gex_peak_df.to_csv(training_gex_filename, index=True)
    testing_gex_filename = outdir + "/" + gene + "/" + "testing_gex.csv"
    testing_gex_peak_df.to_csv(testing_gex_filename, index=True)
    return training_pb_peak_df, training_gex_peak_df, testing_pb_peak_df, testing_gex_peak_df

# ============================================
def build_models(gene):
    global training_pb_peak_df, training_gex_peak_df
    print("Running models for gene: " + gene)
    outdir = output_data_path + "/" + gene 
    # make output directory for gene
    if not os.path.exists(output_data_path + "/" + gene):
        os.makedirs(output_data_path + "/" + gene)
    # 5.1) Filter peaks present in at least 10% samples
    final_pb_peak_df = training_pb_peak_df.loc[training_pb_peak_df[training_pb_peak_df.columns].ne(0).sum(axis=1) >= len(training_pb_peak_df.columns)*.1]
    gene_exp = training_gex_peak_df
    # exit function if fewer than 3 peaks left
    if len(final_pb_peak_df) < 3:
        print("DataFrame has less than 3 peaks. Exiting function.")
        return
    else:        
        # 5.3) implement random forest classifier to select peaks (10% filt peaks)
        # build RF models; rerank after each built model
        test = "rf_ranker"
        build_RFR_model(final_pb_peak_df, gene_exp, gene, outdir, test)
        test = "perm_ranker"
        build_RFR_model(final_pb_peak_df, gene_exp, gene, outdir, test)
        test = "dropcol_ranker"
        build_RFR_model(final_pb_peak_df, gene_exp, gene, outdir, test)
        # build linear regression models; rerank after each built model
        test = "perm_ranker"
        build_LinReg_model(final_pb_peak_df, gene_exp, gene, outdir, test)
        test = "dropcol_ranker"
        build_LinReg_model(final_pb_peak_df, gene_exp, gene, outdir, test)
        # 5.4) Try XGBoost
        test = "perm_ranker"
        build_xgboost_model(final_pb_peak_df, gene_exp, gene, outdir, test)
        test = "dropcol_ranker"
        build_xgboost_model(final_pb_peak_df, gene_exp, gene, outdir, test)
        # 5.5) determine best set of features for each model
        feature_selector(gene, outdir)

# ============================================
#def celltype_split(final_pb_peak_df):
#    celltype_list = final_pb_peak_df.columns.tolist()
#    # dict to hold lists for each cell type
#    lists_by_celltype = {}
#    for sample in celltype_list:
#        # get celltype
#        celltype = sample.split('_')[0]
#        # add to list
#        if celltype in lists_by_celltype:
#            lists_by_celltype[celltype].append(sample)
#        else:
#            lists_by_celltype[celltype] = [sample]
#    # separate dict into lists
#    result_lists = list(lists_by_celltype.values())
#    return result_lists

# ============================================
def run_cross_validations(gene): # 1.) LOO, and Pseudo-Split
    global training_pb_peak_df, training_gex_peak_df, testing_pb_peak_df, testing_gex_peak_df, genes_df, pb_keep, outdir
    print("Running models for gene: " + gene)
    outdir = output_data_path + "/" + gene
    # intiate summary output
    columnnames = ["gene", "celltype", "method", "cross_val", "npeaks_kept", "cv_R2"]
    summary = pd.DataFrame(columns = columnnames)
    # if gene previously modeled:
    if os.path.exists(outdir + "/selected_peaks.csv"):
        # get selected_peaks.csv file
        selected_peaks = pd.read_csv(outdir + "/selected_peaks.csv")
        test_list = selected_peaks["Result"].tolist()
        # subset for cell type
        training_pb_peak_df_sub = training_pb_peak_df
        training_pb_gex_df_sub = training_gex_peak_df
        testing_pb_peak_df_sub = testing_pb_peak_df
        testing_pb_gex_df_sub = testing_gex_peak_df
        if training_pb_gex_df_sub.values.max() > 0 and testing_pb_gex_df_sub.values.max() > 0:
            cv_dict = {}
            for test in test_list:
                print("Interrogating " + test)
                final_peak_list, testdir = select_peaks(test, selected_peaks, gene, outdir)
                # only keep peaks in final_peak_list
                training_sub_pb_peak_df = training_pb_peak_df_sub[training_pb_peak_df_sub.index.isin(final_peak_list)]
                testing_sub_pb_peak_df = testing_pb_peak_df_sub[testing_pb_peak_df_sub.index.isin(final_peak_list)]
                # plot correlation heatmap for final peaks:
                # training:
                corr = (training_sub_pb_peak_df.transpose()).corr(method = 'spearman')
                figname = outdir + "/" + testdir + "_training_peakcorrelations.pdf"
                plt.figure(figsize=(20, 20))
                sns.heatmap(corr, annot = True)
                plt.savefig(figname)
                # testing:
                corr = (testing_sub_pb_peak_df.transpose()).corr(method = 'spearman')
                figname = outdir + "/" + testdir + "_testing_peakcorrelations.pdf"
                plt.figure(figsize=(20, 20))
                sns.heatmap(corr, annot = True)
                plt.savefig(figname)
                ## 1.0) run cross validations on training data
                LOO_y_pred_list, LOO_actual_values, SPLIT_y_pred, SPLIT_actual_values = cross_validation_fun(test, training_sub_pb_peak_df, training_pb_gex_df_sub, testing_sub_pb_peak_df, testing_pb_gex_df_sub, gene, outdir)
                # 1.1) make pred vs act plot for leave-one-out cross validation
                filename = outdir + "/" + testdir + "_loocv_pred_vs_act.pdf"
                r2_value = make_cv_plot(LOO_y_pred_list, LOO_actual_values, filename)
                ### add values to summary dataframe
                new_row = [gene, "all", testdir, "LOO", len(final_peak_list), r2_value]
                summary.loc[len(summary)] = new_row
                # 1.2) make pred vs act plot for split pseudo cross validation
                filename = outdir + "/" + testdir + "_splitpseudocv_pred_vs_act.pdf"
                r2_value = make_cv_plot(SPLIT_y_pred, SPLIT_actual_values, filename)
                ### add values to summary dataframe
                new_row = [gene, "all", testdir, "split-pseudo", len(final_peak_list), r2_value]
                summary.loc[len(summary)] = new_row
        else:
            print("All GEX values are 0 for this celltype.")
            return
    else:
        print("Unable to model gene. Exiting function.")
        return
    return summary

# ============================================
if __name__ == "__main__":
    # import custom functions
    os.chdir(os.getcwd())
    from feature_selection import rf_ranker, perm_ranker, RF_dropcolumn_importance, LinReg_dropcolumn_importance, feature_selector, pareto_frontier
    from model_builders import build_RFR_model, build_LinReg_model 
    #from assess_RF_params import grid_search_init
    from misc_helper_functions import load_files_with_match
    from cross_validation import select_peaks, make_cv_plot, cross_validation_fun, grid_search_init
    # 1.) parse arguments
    args = parse_my_args()
    gene_list = args["gene_list"]
    gex_matrix = args["gex_matrix"]
    peak_matrix = args["peak_matrix"]
    pseudobulk_replicate = args["pseudobulk_replicate"]
    outdir = args["output_dir"]
    outdir = "Results"
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
    columnnames = ["gene", "celltype", "method", "cross_val", "npeaks_kept", "cv_R2"]
    alpha_summary = pd.DataFrame(columns = columnnames)
    genes_df = pd.read_csv(gene_list, sep = "\t")
    genes_df.columns = ["gene", "window"]
    test_gene_list =  ["TRAF3IP2", "TRPT1", "TSPAN14", "TUBD1", "USP4"]
    # For each gene, split dataset by cell type then run models
    for gene in test_gene_list:
    #for gene in genes_df["gene"]:
        print(gene)
        window = genes_df.loc[genes_df["gene"] == gene, "window"].iloc[0]
        # make output directory for gene
        if not os.path.exists(output_data_path + "/" + gene):
            os.makedirs(output_data_path + "/" + gene)
        # 5.1) subset gene and region from peak_df and gex_df:
        gene_peaks = subset_peaks(peak_df, window)
        gene_exp = subset_gex(gex_df, gene)
        # 5.2) get pseudobulk values for gene/region
        outdir = "/storage/home/mfisher42/scProjects/Predict_GEX/Groups_Celltypes_Split_Pseudos_peakfilt10perc_paretofront_08302023/Results"
        training_pb_peak_df, training_gex_peak_df, testing_pb_peak_df, testing_gex_peak_df = make_pseudobulk(gene_peaks, gene_exp, pb_keep)
        # 5.4) run models on each cell type, in parallel
        build_models(gene)
        # 6.0) for each gene and model (with optimal peak set), perform cross validation (incorporate grid search for RFR)
        print("Performing cross validations")
        # 6.2) run cross validations
        #run_cross_validations(gene) # Cross validation using training and testing from splitting the pseudobulk data
        summary = run_cross_validations(gene)
        alpha_summary = pd.concat([alpha_summary, summary], ignore_index = True)
        alpha_summary.to_csv("alpha_summary.csv", index=False)
