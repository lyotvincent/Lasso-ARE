# PARE: an adversarial learning framework for integrating few-shot biological priors into single-cell and spatial omics embedding
This repository is the methodology part of the PARE (Lasso-ARE) package, which integrates for interactive selection and high-resolution reclustering of single-cell and spatial omics data.

## Directory Structure

Please make sure the following files are correctly placed in this folder:

```
this folder
├── LassoARE/                  # LassoARE deep adversarial clustering core package
│   ├── __init__.py
│   ├── lasso_ARE.py
│   ├── recluster_scRNA.py
│   ├── reconstruction.py
│   ├── scARE.py
│   ├── utils.py
│   └── LassoARE_plugin/       # Plugin module (including spatial metric, soft guidance with Moran's I, etc.)
│       ├── __init__.py
│       ├── plugins.py
│       └── reconstruction_plugin.py
├── lassoView.cpp              # C++ pybind source code for Lasso-View
├── setup.py                   # Compilation configuration script for Lasso-View C++ module
├── lassoLPA.py                # Python wrapper for label propagation
├── do_lasso.py                # High-level LassoRefine interface and visual comparison functions
├── requirements.txt           # Pip package dependency list
├── environment.yml            # Conda virtual environment configuration file
├── install_python.sh          # One-click compilation and installation script via Pip
├── install_conda.sh           # One-click compilation and installation script via Conda
├── selection.txt              # Initial cell selection index file for demo testing
├── demo_single_cluster.ipynb  # Standard demonstration Jupyter Notebook (using DFU dataset)
├── demo_single_cluster_fast.ipynb # Fast demonstration Jupyter Notebook (using pbmc3k dataset)
└── pairpotlpa.cpython-310-x86_64-linux-gnu.so # Pre-compiled C++ module for Linux x86_64 (Python 3.10)
```

## Quick Start Guide

### 1. Compiling the C++ Module & Installing Dependencies

Lasso-View uses C++ for efficient graph adjacency weight calculation and label propagation. Before running, it needs to be compiled for your specific system.

> [!TIP]
> A pre-compiled dynamic link library `pairpotlpa.cpython-310-x86_64-linux-gnu.so` is already provided for Linux x86_64 environments with Python 3.10. If you are using a different operating system or Python version, please use the installation scripts below to recompile.

We provide two options of one-click installation and compilation scripts based on your virtual environment manager:

#### Option A: Installation using Conda (Recommended)
If you wish to manage a clean virtual environment using Conda, run the following in the `submit` folder:
```bash
chmod +x install_conda.sh
./install_conda.sh
```
This script will use `environment.yml` to create a Conda virtual environment named `lasso_are` (or update dependencies if it already exists), and automatically call `setup.py` inside that environment to compile the C++ module.
After compilation and installation, you can activate the environment using:
```bash
conda activate lasso_are
```

#### Option B: Installation using Pip
If you are using a standard Python venv environment, run the following in the `submit` folder:
```bash
chmod +x install_python.sh
./install_python.sh
```
This script will install the dependencies declared in `requirements.txt` via `pip` and compile the C++ module.

After compilation, a dynamic link library compatible with your system (e.g., `pairpotlpa.so` or `pairpotlpa.pyd`) will be generated in the current directory.

### 2. Running the Demo Workflows

We provide two Jupyter Notebooks to demonstrate the workflow. You can choose based on your computing resources:

#### Option A: Standard Demo (`demo_single_cluster.ipynb`)
This notebook runs the workflow on the primary DFU single-cell dataset.
1. **Import & Load Data**: Load the large `adata_fib.h5ad` demonstration dataset.
2. **Lasso-View Refinement**: Read `selection.txt` (the initial cell selection set) and call `do_lasso_file` to propagate labels on the graph, outputting the topology-corrected cell selection set (Lasso selected cells).
3. **LassoARE Reclustering**: Use `recluster_with_lasso_are` to perform adversarial reclustering. This process strengthens the selected cell features in the latent space and removes batch effects using the Harmony integration.
4. **Visualization**: Directly call `sc.pl.umap` to plot the UMAP figures before and after the algorithm to visually compare the clustering performance.
5. **Output**: The output AnnData files are saved under `new_adata/`.

#### Option B: Fast Demo (`demo_single_cluster_fast.ipynb`)
This notebook uses a lightweight, pre-processed Scanpy built-in dataset (`pbmc3k_processed`) for a fast, resource-friendly walkthrough.
1. **Import & Load Data**: Load `data/pbmc3k_processed.h5ad`.
2. **Simulate User Selection**: Select cells from the `B cells` cluster to simulate a user selecting cells in a visualization tool.
3. **Lasso-View Refinement**: Call the in-memory `do_lasso` function to propagate labels on the graph and obtain topology-corrected cell selections.
4. **LassoARE Reclustering**: Run `recluster_with_lasso_are` to perform adversarial reclustering in the latent space.
5. **Visualization**: Generate UMAP plots to compare and inspect clustering results.
6. **Output**: The output AnnData file is saved under `new_adata_fast/`.
