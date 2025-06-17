!/usr/sbin/anaconda


import argparse
import glob
import numpy as np
import os
import pandas as pd
import re
from sklearn.linear_model import LinearRegression


# 03-12-2024
# Compute correlation of coefficient R^2 values:
# - for each gene, for each model, load in cross-validation results
# - output two things:
# 1.) For each gene: the sample, real GEX, and mean(cv-predicted) GEX
# 2.) For all genes in one summary file: gene, method, npeaks, CV_R^2


# ============================================
def parse_my_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r", "--results_directory", type = str, help = "model results directory")
    parser.add_argument("-o", "--output_directory", type = str, help = "output directory")
    return vars(parser.parse_args())


# ============================================
def compute_values(gene):
    global gene_path
    test_list = ["rf_dropcolranker", "rf_permranker", "rf_ranker", "lr_dropcolranker", "lr_permranker", "xgb_dropcolranker", "xgb_permranker", "xgb_ranker", "lgbm_dropcolranker", "lgbm_permranker", "lgbm_ranker"]
    columns = ["Gene", "Test_Method", "CV_R2"]
    gene_df = pd.DataFrame(columns = columns)
    for test in test_list:
        print(test)
        # get cv results
        cv_dir = gene_path + "/" + test + "/" + "cross_validations_all_peaks"
        if os.path.exists(cv_dir):
            cv_files = glob.glob(f"{cv_dir}/*.csv")
            combined_df = pd.DataFrame()
            for cv_file in cv_files:
                cv_result = pd.read_csv(cv_file, sep = ",", index_col = 0)
                value = cv_result.iloc[0,1]
                if not isinstance(value, float):
                    cv_result["Predicted"] = cv_result["Predicted"].apply(lambda x: float(x.strip("[]")))
                combined_df = pd.concat([combined_df, cv_result])
            # for any duplicate samples, avergae their predictions:
            combined_df = combined_df.groupby(combined_df.index).mean()
            # save this dataframe
            filename = gene_path + "/" + test + "_aggregated_predictions.csv"
            combined_df.to_csv(filename, index = True)
            # compute coefficient of correlation R^2 value
            x = combined_df[gene].values.reshape(-1, 1)  # Independent variable
            y = combined_df["Predicted"].values  # Dependent variable
            model = LinearRegression().fit(x, y)
            r2 = model.score(x, y)
            # add to df
            new_row = {"Gene" : gene, "Test_Method" : test, "CV_R2" : r2}
            gene_df.loc[len(gene_df)] = new_row
        else:
            print("No results for this test")
    return gene_df


        


# ============================================
if __name__ == "__main__":
    # 1.) parse arguments
    args = parse_my_args()
    results_dir = args["results_directory"]
    out_dir = args["output_directory"]
    # 2.) for each gene, get R-squared values for coefficient of correlation:
    genes = os.listdir(results_dir)
    columns = ["Gene", "Test_Method", "CV_R2"]
    output_summary = pd.DataFrame(columns = columns)
    for gene in genes:
        gene_path = results_dir + gene
        gene_files = os.listdir(gene_path)
        if len(gene_files) == 0:
            print("The directory for " + gene + "is empty.")
        elif sorted(gene_files) == ['gex.csv', 'peaks.csv']:
            print("This gene did not pass checks for modeling: " + gene)
        else:
            # get aggregated cv r_2 values:
            print(gene)
            gene_df = compute_values(gene)
            output_summary = pd.concat([output_summary, gene_df])
    filename = out_dir + "coefficient_of_correlation_summary.csv"
    output_summary.to_csv(filename, index = False)
