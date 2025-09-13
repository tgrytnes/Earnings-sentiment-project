# Earnings Sentiment Project

This repository is a **reusable MLOps template** for AI and data science projects.  
It provides a **config-driven, hardware-independent pipeline structure** with clear separation of source code, configs, data, and reports.  
The goal is to standardize project setup, ensure reproducibility, and accelerate development across different ML projects.

⚡ **How to use this template:**  
- Do **not** develop directly in this repository.  
- Instead, click **"Use this template"** on GitHub to create a fresh project repository.  
- Update the project name in `src/yourproj` and configuration files to match your new project.  
- Keep this repo unchanged as your base MLOps template.

**For environment setup and infrastructure instructions (including Docker, RunPod, etc.), please see the dedicated infrastructure repository.**

---

## Quickstart

Once your development environment is configured (e.g., inside a Docker container from the infrastructure repository), you can run the full pipeline.

The main training pipeline can be executed with a single script, which takes a configuration file as an argument:

```bash
# Run the full pipeline using the baseline experiment configuration
bash scripts/run_train.sh configs/exp_baseline.yaml

## Azure Blob Storage (Cloud)

Use Azure Blob to store datasets, images, and artifacts without running your own object storage.

- Environment:
  - Copy `configs/azure.blob.env` to a local, untracked file and fill secrets:
    - `cp configs/azure.blob.env configs/azure.blob.env.local`
    - Edit the `*.local` file with your account and key, then:
    - `source configs/azure.blob.env.local`
- Python helpers (built in):
  ```python
  from earnings_sentiment.storage import get_container_client
  cc = get_container_client()  # uses AZURE_BLOB_CONTAINER
  cc.upload_blob("example.txt", b"hello", overwrite=True)
  data = cc.download_blob("example.txt").readall()
  ```
- Connectivity test:
  - `python scripts/azure_blob_smoke.py`

Notes:
- Env vars used: `AZURE_BLOB_ENDPOINT` or `AZURE_STORAGE_ACCOUNT`, plus `AZURE_STORAGE_KEY` (or `AZURE_STORAGE_CONNECTION_STRING`), and `AZURE_BLOB_CONTAINER`.
