import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import pyarrow.parquet as pq
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

with open('elasticnet_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('trained_cols.pkl', 'rb') as f:
	trained_cols = pickle.load(f)
	
benchmark_meta = pd.read_csv("computage_train_meta.tsv", sep="\t")
benchmark_files = glob.glob('data/benchmark/*.parquet')
benchmark_subset = benchmark_files
#benchmark_subset = ["computage_train_data_GSE112696.parquet", "computage_train_data_GSE112987.parquet"]
test_merged_list = []

for f in benchmark_subset:
	print(f"Processing {f}...")
	
	test_methyl = pd.read_parquet(f)
	
	# Transform data to be able to work with
	test_methyl_T = test_methyl.transpose()
	test_methyl_T = test_methyl_T.reset_index().rename(columns={'index': 'SampleID'})
	
	test_merged = test_methyl_T.merge(
		benchmark_meta,
		on="SampleID",
		how="inner"
	)
	
	if not test_merged.empty:
		test_merged_list.append(test_merged)
	else:
		print(f"Warning: {f} - merge resulted in empty dataframe.")
	
if test_merged_list:
	benchmark_data = pd.concat(test_merged_list, ignore_index=True)

	non_methylation_cols = ['SampleID', 'DatasetID', 'PlatformID', 'Tissue', 'CellType', 'Gender', 'Age', 'Condition', 'Class']
	X_benchmark = benchmark_data.drop(columns=non_methylation_cols)
    
	X_benchmark = X_benchmark.astype('float32')
    
    # Cut the columns not needed
	valid_columns = [col for col in trained_cols if col in X_benchmark.columns]
	X_benchmark_aligned = X_benchmark[valid_columns]
    # Add columns needed
	X_benchmark_aligned = X_benchmark_aligned.reindex(columns=trained_cols)
	print("X shape:", X_benchmark_aligned.shape)
    
	with open('imputer.pkl', 'rb') as f:
		imputer = pickle.load(f)
    
	X_benchmark_imputed = imputer.transform(X_benchmark_aligned)
	y_benchmark = benchmark_data['Age']

	y_pred_benchmark = model.predict(X_benchmark_imputed)

	# Step 6: Evaluate on benchmark
	print("Benchmark MAE:", mean_absolute_error(y_benchmark, y_pred_benchmark))
	print("Benchmark R²:", r2_score(y_benchmark, y_pred_benchmark))
	
	plt.figure(figsize=(8, 8))
	sns.scatterplot(x=y_benchmark, y=y_pred_benchmark, alpha=0.6, edgecolor=None)
	
	# Draw diagonal for reference (perfect prediction line)
	max_age = max(max(y_benchmark), max(y_pred_benchmark))
	plt.plot([0, max_age], [0, max_age], color='red', linestyle='--', label='Perfect Prediction')
	
	plt.xlabel("Actual Age")
	plt.ylabel("Predicted Age")
	plt.title("Actual vs. Predicted Age")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig("actual_vs_predicted_age.png", dpi=300, bbox_inches='tight')
	plt.show()