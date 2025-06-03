import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import pyarrow.parquet as pq

# Load metadata
train_meta = pd.read_csv("computage_train_meta.tsv", sep="\t")
print("train meta cols:")
print(train_meta.columns)

df_methyl = pd.read_parquet('data/train/computage_train_data_GSE40005.parquet')
print("train data cols:")
print(df_methyl.columns)
df_methyl_T = df_methyl.transpose()
print("train data rows:")
print(df_methyl_T.columns)

benchmark_meta = pd.read_csv("computage_bench_meta.tsv", sep="\t")
print("test meta cols:")
print(benchmark_meta.columns)

test_methyl = pd.read_parquet('data/benchmark/computage_bench_data_GSE32148.parquet')
print("test data cols:")
print(test_methyl.columns)
test_methyl_T = test_methyl.transpose()
print("test data rows:")
print(test_methyl_T.columns)