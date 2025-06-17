#!/usr/sbin/anaconda


import argparse
import glob
import numpy as np
import os
import pandas as pd
import re


# 03-18-2024
# Get predictions, aggregate


# ============================================
def parse_my_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--model_results_directory", type = str, help = "model results directory")


# ============================================
def get_aggregates(gene, results_dir):
    prediction_files = glob.glob(gene_path + "/*_predictions.csv")
    gene_df = pd.DataFrame()
    for predict_file in prediction_files:
        prediction = pd.read_csv(predict_file, index_col = 0)
        gene_df = pd.concat([gene_df, prediction["Predicted"]], axis = 1)
    # aggregate predictions
    gene_df[gene] = gene_df.mean(axis = 1)
    final_gene_df = gene_df.iloc[:, -1]
    return final_gene_df


# ============================================
if __name__ == "__main__":
    # 1.) parse arguments
    args = parse_my_args()
    results_dir = args["model_results_directory"]
    os.chdir(results_dir)
    # 2.) aggregate predictions for models
    genes = os.listdir(results_dir)
    prediction_mat = pd.DataFrame()
    for gene in genes:
        gene_path = results_dir + gene
        gene_files = os.listdir(gene_path)
        if len(gene_files) == 0:
            print("The directory for " + gene + "if empty.")
        elif sorted(gene_files) == ['gex.csv', 'peaks.csv']:
            print("This gene did not pass checks for modeling: " + gene)
        else:
            # load and aggregate predictions
            prediction_aggregates = get_aggregates(gene, gene_path)
            prediction_mat = pd.concat([prediction_mat, prediction_aggregates], axis = 1)
    prediction_mat.to_csv("model_aggregated_predictions.csv", index = True)
