#!/bin/bash
# One-click installation script: Create/Update virtual environment using Conda configuration and compile the C++ module

# Exit immediately if any command returns a non-zero status
set -e

ENV_NAME="lasso_are"

echo "=== 1. Creating/Updating Conda virtual environment using environment.yml ==="
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Detected that virtual environment $ENV_NAME already exists, updating dependencies..."
    conda env update -n "$ENV_NAME" -f environment.yml --prune
else
    echo "Creating virtual environment: $ENV_NAME..."
    conda env create -f environment.yml
fi

echo "=== 2. Compiling C++ module (pairpotlpa) for Lasso-View under environment $ENV_NAME ==="
# Use conda run to execute python compilation in the specific environment to avoid activation issues in scripts
conda run -n "$ENV_NAME" python setup.py build_ext --inplace

echo "=== 3. Installation and compilation successful! ==="
echo "You can activate the virtual environment using the following command to run demo_single_cluster.ipynb:"
echo "   conda activate $ENV_NAME"
