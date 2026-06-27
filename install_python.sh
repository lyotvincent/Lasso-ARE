#!/bin/bash
# One-click installation script: Install Python dependencies and compile the C++ pybind11 module for Lasso-View

# Exit immediately if any command returns a non-zero status
set -e

echo "=== 1. Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== 2. Compiling C++ module (pairpotlpa) for Lasso-View ==="
python setup.py build_ext --inplace

echo "=== 3. Installation and compilation successful! ==="
echo "You can now run demo_single_cluster.ipynb to verify the whole workflow."
