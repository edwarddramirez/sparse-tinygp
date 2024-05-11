# sparse-tinygp
Bayesian inference using sparse gaussian processes via `tinygp`. Examples include 1D and 2D implementation.

# Notebooks 
1. `01_inference_sparse_gp.ipynb`: SVI with a Sparse GP
2. `02_2d_sparse_gp.ipynb`: 2D Sparse GP
3. `03_rffs_sparse_gp.ipynb`: SVI with RFF-approximation to sparse-GP (Bonus)

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

<!-- ### References  -->
[1]: <https://tinygp.readthedocs.io/en/latest/tutorials/derivative.html> "Derivative Observations & Pytree Data"