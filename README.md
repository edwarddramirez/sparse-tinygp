[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edwarddramirez/sparse-tinygp/HEAD) [![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-brightgreen.svg)](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en) ![Python](https://img.shields.io/badge/python-3.11.4-blue.svg) ![Repo Size](https://img.shields.io/github/repo-size/edwarddramirez/sparse-tinygp) 

# sparse-tinygp
Bayesian inference using sparse gaussian processes via `tinygp`. Examples include 1D and 2D implementation.

# Notebooks 
1. `01_inference_sparse_gp.ipynb`: SVI with a Sparse GP
2. `02_2d_sparse_gp.ipynb`: 2D Sparse GP
3. `03_rffs_sparse_gp.ipynb`: SVI with RFF-approximation to sparse-GP (Sparse GP helps fitting, RFF helps sampling)

# Installation
Run the `environment.yml` file by running the following command on the main repo directory:
```
conda env create
```
The installation works for `conda==4.12.0`. This will install all packages needed to run the code on a CPU with `jupyter`. 

If you want to run this code with a CUDA GPU, you will need to download the appropriate `jaxlib==0.4.13` version. For example, for my GPU running on `CUDA==12.3`, I would run:
```
pip install jaxlib==0.4.13+cuda12.cudnn89
```
The key to using this code directly would be to retain the `jax` and `jaxlib` versions. 