#!/usr/sbin/anaconda


import argparse
import glob
import numpy as np
import os
import pandas as pd
import re


# 12-04-2023
# Aggregate peak ranks for models per gene
# run like this:
# ./python ./aggregate_peak_ranks.py -g <GWAS_CSV_FILE> -r <MODEL_RESULTS_DIR> -o <output>


# ============================================
def parse_my_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g", "--gwas_snps", type = str, help = "gwas snps")
    parser.add_argument("-r", "--model_results_directory", type = str, help = "model results directory")
    parser.add_argument("-o", "--output_directory", type = str, help = "output directory")
    return vars(parser.parse_args())


# ============================================
def count_lines(file_path):
    with open(file_path, 'r') as file:
        return sum(1 for line in file)


# ============================================
def modify_peak_coordinates(peak_list):
    modified_peaks = []
    for coord in peak_list:
        match = re.match(r'chr(\d+)_(\d+)_(\d+)', coord)
        if match:
            chromosome_number = match.group(1)
            start_position = match.group(2)
            end_position = match.group(3)
            modified_peaks.append(f'chr{chromosome_number}:{start_position}-{end_position}')
        else:
           # Handle the case where the chromosome number is not found
            modified_peaks.append(coord)
    return modified_peaks


# ============================================
def check_snps(gene_snps, peak_list):
    snp_list = [f"{row['Chr']}:{row['Variant Position']}" for index, row in gene_snps.iterrows()]
    snp_dict = {}
    for snp in snp_list:
        snp_chr = snp.split(":")[0]
        snp_pos = int(snp.split(":")[1])
        overlapping_peak = next((peak for peak in peak_list if peak.startswith(snp_chr) and
                         int(peak.split(':')[1].split('-')[0]) <= snp_pos <= int(peak.split('-')[1])), None)
        if overlapping_peak is not None:
            snp_dict[snp] = overlapping_peak
        else:
            continue
    return snp_dict


# ============================================
def get_aggregates(gene, results_dir):
    #global gwas_df
    gene_path = results_dir + gene
    # initiate output
    alpha_summary = pd.DataFrame()
    test_list = ["rf_dropcolranker", "rf_permranker", "rf_ranker", "lr_dropcolranker", "lr_permranker", "xgb_dropcolranker", "xgb_permranker", "xgb_ranker", "lgbm_dropcolranker", "lgbm_permranker", "lgbm_ranker"]
    #test_list = ["rf_dropcolranker", "rf_permranker", "rf_ranker", "xgb_dropcolranker", "xgb_permranker", "xgb_ranker", "lgbm_dropcolranker", "lgbm_permranker", "lgbm_ranker"]
    for test in test_list:
        test_dir = gene_path + "/" + test
        # get file with all peaks
        peak_rank_files = glob.glob(f"{test_dir}/*importance.csv")
        sorted_files = sorted(peak_rank_files, key = count_lines, reverse = True)
        selected_file = sorted_files[0]
        peak_rank_df = pd.read_csv(selected_file, sep = ",")
        # reformat peak syntax for LightGBM
        model = test.split("_")[0]
        if model == "lgbm":
            all_peaks = peak_rank_df["Peak"].tolist()
            all_peaks = modify_peak_coordinates(all_peaks) # fix peak list
            peak_rank_df["Peak"] = peak_rank_df["Peak"].apply(lambda x: f"chr{x.split('_')[0][3:]}:{x.split('_')[1]}-{x.split('_')[2]}")
        # compute z-score
        mu = peak_rank_df["Importance"].mean()
        sigma = peak_rank_df["Importance"].std()
        temp_summary = pd.DataFrame()
        temp_summary["Peaks"] = peak_rank_df["Peak"]
        temp_summary[test + "_Zscore"] = (peak_rank_df["Importance"] - mu) / sigma
        # if alpha_summary is empty, crate new; otherwise, merge
        if len(alpha_summary) == 0:
            alpha_summary["Peaks"] = temp_summary["Peaks"]
            alpha_summary[test + "_Zscore"] = temp_summary[test + "_Zscore"]
        else:
            alpha_summary = pd.merge(alpha_summary, temp_summary, on = "Peaks", how = "outer")
    # Aggregate Zscores for each test (include LR)
    zscore_columns = alpha_summary.filter(like = "Zscore")
    alpha_summary["Average_Zscore"] = zscore_columns.mean(axis = 1)
    alpha_summary = alpha_summary.sort_values(by = "Average_Zscore", ascending = False)
    alpha_summary = alpha_summary.reset_index(drop = True)
    # save alpha_summary as output in gene folder
    filename = results_dir + gene + "/aggregated_peak_importances_inclLR.csv"
    alpha_summary.to_csv(filename, index = True)
    # REPEAT: but exclude LR
    remove_cols = ["lr_dropcolranker_Zscore", "lr_permranker_Zscore", "Average_Zscore"]
    exclLR_alpha_summary = alpha_summary.drop(columns = remove_cols)
    # Aggregate Zscores for each test (exclude LR)
    zscore_columns = exclLR_alpha_summary.filter(like = "Zscore")
    exclLR_alpha_summary["Average_Zscore"] = zscore_columns.mean(axis = 1)
    exclLR_alpha_summary = exclLR_alpha_summary.sort_values(by = "Average_Zscore", ascending = False)
    exclLR_alpha_summary = exclLR_alpha_summary.reset_index(drop = True)
    # save exclLR_alpha_summary as output in gene folder
    filename = results_dir + gene + "/aggregated_peak_importances_exclLR.csv"
    exclLR_alpha_summary.to_csv(filename, index = True)
    return alpha_summary, exclLR_alpha_summary


# ============================================
def get_overlaps_all(gene, alpha_summary, results_dir):
    global gwas_df, output_summary
    gene_path = results_dir + gene
    gene_files = os.listdir(gene_path)
    if len(gene_files) == 0:
        print("The directory for " + gene + "if empty.")
    elif sorted(gene_files) == ['gex.csv', 'peaks.csv']:
        print("This gene did not pass checks for modeling: " + gene)
    else:
        print("This gene passed checks for modeling. Getting overlaps for " + gene)
        # get SNPs for gene
        gene_snps = gwas_df[gwas_df["Gene"] == gene]
        all_peaks = alpha_summary["Peaks"].tolist()
        peak_snp_dict = check_snps(gene_snps, all_peaks)
        if len(peak_snp_dict) == 0:
            print("No overlap")
        else:
            df = pd.DataFrame(list(peak_snp_dict.items()), columns = ["SNP", "Peak_Coord"])
            df["Peak_Rank"] = np.nan
            for peak in peak_snp_dict.values():
                rank = int(alpha_summary[alpha_summary["Peaks"] == peak].index[0] + 1)
                df.loc[df["Peak_Coord"] == peak, "Peak_Rank"] = rank
            # Prep for output for select
            df["Gene"] = gene
            df["nPeaks_Total"] = len(all_peaks)
            df["Test_Method"] = "Aggregate"
            output_summary = pd.concat([output_summary, df], ignore_index = True)
            output_summary.to_csv("snp_overlaps_aggregated_inclLR.csv", index = False)


# ============================================
def get_overlaps_exclLR(gene, alpha_summary, results_dir):
    global gwas_df, exclLR_output_summary
    gene_path = results_dir + gene
    gene_files = os.listdir(gene_path)
    if len(gene_files) == 0:
        print("The directory for " + gene + "if empty.")
    elif sorted(gene_files) == ['gex.csv', 'peaks.csv']:
        print("This gene did not pass checks for modeling: " + gene)
    else:
        print("This gene passed checks for modeling. Getting overlaps for " + gene)
        # get SNPs for gene
        gene_snps = gwas_df[gwas_df["Gene"] == gene]
        all_peaks = alpha_summary["Peaks"].tolist()
        peak_snp_dict = check_snps(gene_snps, all_peaks)
    if len(peak_snp_dict) == 0:
        print("No overlap")
    else:
        df = pd.DataFrame(list(peak_snp_dict.items()), columns = ["SNP", "Peak_Coord"])
        df["Peak_Rank"] = np.nan
        for peak in peak_snp_dict.values():
            rank = int(alpha_summary[alpha_summary["Peaks"] == peak].index[0] + 1)
            df.loc[df["Peak_Coord"] == peak, "Peak_Rank"] = rank
        # Prep for output for select
        df["Gene"] = gene
        df["nPeaks_Total"] = len(all_peaks)
        df["Test_Method"] = "Aggregate"
        exclLR_output_summary = pd.concat([exclLR_output_summary, df], ignore_index = True)
        exclLR_output_summary.to_csv("snp_overlaps_aggregated_exclLR.csv", index = False)


# ============================================
if __name__ == "__main__":
    # 1.) parse arguments
    args = parse_my_args()
    gwas_snps = args["gwas_snps"]
    results_dir = args["model_results_directory"]
    out_dir = args["output_directory"]
    os.chdir(out_dir)
    # 2.) load in data
    gwas_df = pd.read_csv(gwas_snps, sep = ",")
    # 3.) aggregate and find SNP overlaps
    genes = os.listdir(results_dir)
    columns = ["Gene", "SNP", "Peak_Coord", "Peak_Rank", "nPeaks_Total", "Test_Method"]
    output_summary = pd.DataFrame(columns = columns)
    exclLR_output_summary = pd.DataFrame(columns = columns)
    for gene in genes:
        gene_path = results_dir + gene
        gene_files = os.listdir(gene_path)
        if len(gene_files) == 0:
            print("The directory for " + gene + "if empty.")
        elif sorted(gene_files) == ['gex.csv', 'peaks.csv']:
            print("This gene did not pass checks for modeling: " + gene)
        else:
            # aggregate
            alpha_summary, exclLR_alpha_summary = get_aggregates(gene, results_dir)
            # find SNP overlaps, incl LR
            get_overlaps_all(gene, alpha_summary, results_dir)
            # find SNP overlaps, excl LR
            get_overlaps_exclLR(gene, exclLR_alpha_summary, results_dir)
