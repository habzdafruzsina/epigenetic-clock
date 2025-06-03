from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='computage/computage_bench', 
    repo_type="dataset",
    local_dir='.',
    resume_download=True
)


import xgboost as xgb