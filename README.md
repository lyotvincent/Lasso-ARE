# LassoARE & Lasso-View Methodology Submission

This folder contains the minimum viable running code (Methodology Part) for academic submission or deployment. This method combines **Lasso-View (graph-based semi-supervised label propagation algorithm)** and **LassoARE (adversarial autoencoding subpopulation clustering algorithm)** for interactive selection and high-resolution reclustering of single-cell transcriptome data (scRNA-seq).

## Directory Structure

Please make sure the following files are correctly placed in the `submit` folder:

```
submit/
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
├── demo_single_cluster.ipynb  # Demonstration Jupyter Notebook
└── adata_fib.h5ad             # Demonstration single-cell dataset (needs to be manually copied to this directory)
```

> [!IMPORTANT]
> **Note (Preparation)**:
> Due to background terminal constraints in the agent environment, please manually run the following commands in the project root directory to copy all dependencies and plugins to the `submit` folder:
> 
> ```bash
> # 1. Copy the entire LassoARE directory (including core algorithm and LassoARE_plugin subdirectory)
> cp -r LassoARE/ submit/
> 
> # 2. Copy the Lasso-View dependencies from NKT_tmp
> cp NKT_tmp/lassoView.cpp NKT_tmp/setup.py NKT_tmp/lassoLPA.py NKT_tmp/do_lasso.py submit/
> 
> # 3. Copy the demonstration dataset
> cp datasets_260407/adata_fib.h5ad submit/
> ```

---

## Quick Start Guide

### 1. Compiling the C++ Module & Installing Dependencies

Lasso-View uses C++ for efficient graph adjacency weight calculation and label propagation. Before running, it needs to be compiled for your specific system.

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

### 2. Running the Demo Workflow

Open Jupyter Notebook and run `demo_single_cluster.ipynb`:

1. **Import & Load Data**: Load the `adata_fib.h5ad` demonstration dataset.
2. **Lasso-View Refinement**: Read `selection.txt` (the initial cell selection set) and call `do_lasso_file` to propagate labels on the graph, outputting the topology-corrected cell selection set (Lasso selected cells).
3. **LassoARE Reclustering**: Use `recluster_with_lasso_are` to perform adversarial reclustering. This process strengthens the selected cell features in the latent space and removes batch effects using the Harmony integration.
4. **Visualization**: Directly call `sc.pl.umap` to plot the UMAP figures before and after the algorithm to visually compare the clustering performance.
