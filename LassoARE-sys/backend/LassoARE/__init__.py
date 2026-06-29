"""
LassoARE: Lasso-guided Adversarial autoencoding with Residual connections
Refactored and independent implementation for single-cell clustering

Main modules:
- scARE: Single-cell Autoencoder with Residual connections (matrix-based)
- lasso_ARE: Adversarial training with multiple user-guided constraints (matrix-based)
- recluster_scRNA: AnnData interface for scRNA-seq data
- utils: Utility functions and layers

Usage:
    # For AnnData objects (recommended for scRNA-seq):
    from LassoARE.recluster_scRNA import recluster_with_lasso_are
    
    # For matrix-based operations (general purpose):
    from LassoARE.lasso_ARE import LassoARE
    from LassoARE.scARE import scARE
"""

from .scARE import scARE, ResidualBlock, buildNetwork
from .lasso_ARE import LassoARE, BinaryDiscriminator, MultiDiscriminator
from .recluster_scRNA import recluster_with_lasso_are, recluster_with_gan
from .reconstruction import reconstruction_with_lasso_are, reconstruction_with_ref
from .utils import ZINBLoss, MeanAct, DispAct, cluster_acc, to_dense, check_matrix

__version__ = "1.0.0"
__all__ = [
    # Main classes
    'scARE',
    'LassoARE',
    
    # High-level functions
    'recluster_with_lasso_are',
    'recluster_with_gan',  # backward compatibility
    'reconstruction_with_lasso_are',
    'reconstruction_with_ref',
    
    # Building blocks
    'ResidualBlock',
    'buildNetwork',
    'BinaryDiscriminator',
    'MultiDiscriminator',
    
    # Utilities
    'ZINBLoss',
    'MeanAct',
    'DispAct',
    'cluster_acc',
    'to_dense',
    'check_matrix',
]
