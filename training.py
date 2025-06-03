import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import pyarrow.parquet as pq
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load metadata
train_meta = pd.read_csv("computage_train_meta.tsv", sep="\t")

# Give back rows of meta file, where the meta file's 'Condition' attributes are 'HC's (healthy)
train_healthy_meta = train_meta[train_meta['Condition'] == 'HC']

# Load all training methylation data
train_files = glob.glob('data/train/*.parquet')
train_subset = train_files

merged_list = []


for f in train_subset:
    print(f"Processing {f}...")

    # df=DataFrame
    df_methyl = pd.read_parquet(f)
    #print(df_methyl.columns)

    # Transform data to be able to work with
    df_methyl_T = df_methyl.transpose()
    df_methyl_T = df_methyl_T.reset_index().rename(columns={'index': 'SampleID'})

    # Merge train methylation data with train_healthy_meta
    df_merged = df_methyl_T.merge(
        train_healthy_meta,
        on="SampleID",
        how="inner"
    )

    samples_in_methyl = set(df_methyl_T['SampleID'])
    samples_in_meta = set(train_healthy_meta['SampleID'])

    print(f"healthy patients: {len(samples_in_methyl & samples_in_meta)}")

    if not df_merged.empty:
        merged_list.append(df_merged)
    else:
        print(f"Warning: {f} - merge resulted in empty dataframe.")


# Combine all filtered/merged data into one DataFrame
if merged_list:
    train_data_healthy = pd.concat(merged_list, ignore_index=True)

    # Drop non-methylation columns
    non_methylation_cols = ['SampleID', 'DatasetID', 'PlatformID', 'Tissue', 'CellType', 'Gender', 'Age', 'Condition', 'Class']
    X = train_data_healthy.drop(columns=non_methylation_cols)
    y = train_data_healthy['Age']
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    cpg_columns = X.columns.tolist()
    with open('cpg_columns.pkl', 'wb') as f:
        pickle.dump(cpg_columns, f)

    # instead of deleting NaN values, I compute mean, and to reduce data size, cast to float32
    X = X.astype('float32')
    imputer = SimpleImputer(strategy='mean', copy=False)

    # transformed array would be too large, so I use chunks
    chunks = []
    chunk_size = 100_000
    for start in range(0, X.shape[1], chunk_size):
        end = min(start + chunk_size, X.shape[1])
        X_chunk = X.iloc[:, start:end]
        X_chunk_imputed = imputer.fit_transform(X_chunk)
        chunks.append(pd.DataFrame(X_chunk_imputed, columns=X_chunk.columns, index=X.index))

    # Combine all chunks
    X_imputed_df = pd.concat(chunks, axis=1)

    # At a simple aging clock model compute optimal alpha and l1_ratio
    elastic_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        cv=10,
        n_alphas=100,
        random_state=42,
        max_iter=10000
    )

    elastic_cv.fit(X_imputed_df, y)

    print(f"Best lambda (alpha in sklearn): {elastic_cv.alpha_}")
    print("Optimal l1_ratio:", elastic_cv.l1_ratio_)

    print("ElasticNetCV Model training complete.")

    # Create ElasticNet Model based on cv results
    elastic_net = ElasticNet(
        alpha=elastic_cv.alpha_,
        l1_ratio=elastic_cv.l1_ratio_,
        random_state=42,
        max_iter=10000
    )
    elastic_net.fit(X_imputed_df, y)

    print("ElasticNet Model training complete.")

    # Recreate inputer
    X_train_aligned = X.reindex(columns=X_imputed_df.columns)
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train_aligned)

    with open('imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)

    with open('trained_cols.pkl', 'wb') as f:
        pickle.dump(X_imputed_df.columns, f)

    with open('elasticnet_model.pkl', 'wb') as f:
        pickle.dump(elastic_net, f)
        
    age_counts = train_data_healthy['Age'].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=age_counts.index, y=age_counts.values, color='skyblue')
    plt.xlabel("Age")
    plt.ylabel("Number of Healthy Individuals")
    plt.title("Number of Healthy People by Age")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("train_ages.png", dpi=300, bbox_inches='tight')
    plt.show()

else:
    print("No healthy methylation data found!")