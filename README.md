# MLOps Project Template

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
