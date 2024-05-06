#!/usr/sbin/anaconda

import argparse
import fnmatch
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
from random import shuffle
from scipy import sparse, io
import statsmodels.api as sm
import sys

import warnings
from sklearn.exceptions import DataConversionWarning

random.seed(12345)

# 04-30-2024
# Split data into training and testing by donor (60/40)
os.chdir("/storage/home/hcoda1/6/mfisher42/scratch/scATAC_Express/Split_Test_Train_042024")
# load pseudobulked IDs from previous analysis
file = "/storage/home/hcoda1/6/mfisher42/scratch/scATAC_Express/Multitest_kfoldcv_95featselect_hyperparam_10perc_parallel_11102023/Results/BACH2/gex.csv"
pseudobulked_gex = pd.read_csv(file, sep = ",")
sample_ids = pseudobulked_gex.columns.tolist()
# get donor list
split_sample_ids = [sample_id.split("_", 1) for sample_id in sample_ids]
set_sample_ids = list(set([sample_id[1] for sample_id in split_sample_ids[1:]]))
### random split by donor: 60/40
# Shuffle the data for randomness
shuffle(set_sample_ids)
# Calculate split point based on 60% of the data length
split_point = int(0.6 * len(set_sample_ids))
# Split the data into two lists
train_data = set_sample_ids[:split_point]
test_data = set_sample_ids[split_point:]
# add back cell type labels
training_samples = list()
for donor in train_data:
        for sample in sample_ids:
                if donor in sample:
                        training_samples.append(sample)
testing_samples = list()
for donor in test_data:
        for sample in sample_ids:
                if donor in sample:
                        testing_samples.append(sample)
# save ids in a csv file
train_df = pd.DataFrame(list(zip(*[training_samples]))).add_prefix('Training')
train_df.to_csv("training_samples.csv", index = False)
test_df = pd.DataFrame(list(zip(*[testing_samples]))).add_prefix('Testing')
test_df.to_csv("testing_samples.csv", index = False)

