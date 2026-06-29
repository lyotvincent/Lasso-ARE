"""
Utility functions and classes for LassoARE
Independent implementation without external dependencies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


# ==================== Loss Functions ====================

class ZINBLoss(nn.Module):
    """Zero-Inflated Negative Binomial Loss"""
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        """
        ZINB loss calculation
        Args:
            x: observed counts
            mean: predicted mean
            disp: dispersion parameter
            pi: zero-inflation probability
            scale_factor: scaling factor (size factor)
            ridge_lambda: ridge regularization weight
        """
        device = mean.device
        if x.device != device:
            x = x.to(device)
        if isinstance(scale_factor, torch.Tensor) and scale_factor.device != device:
            scale_factor = scale_factor.to(device)
        if disp.device != device:
            disp = disp.to(device)
        if pi.device != device:
            pi = pi.to(device)
        
        eps = 1e-10
        
        # Handle scale_factor
        if isinstance(scale_factor, torch.Tensor):
            if scale_factor.dim() == 1:
                scale_factor = scale_factor[:, None]
        
        # Ensure x is non-negative (clamp any negative values from PCA)
        x = torch.clamp(x, min=0.0)
        
        mean = mean * scale_factor
        # Clamp mean to avoid extreme values
        mean = torch.clamp(mean, min=eps, max=1e6)
        
        # Clamp disp and pi to valid ranges
        disp = torch.clamp(disp, min=eps, max=1e4)
        pi = torch.clamp(pi, min=eps, max=1.0-eps)
        
        # ZINB likelihood with additional safety checks
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        # Check for NaN or Inf and replace with large finite value
        result = torch.where(torch.isnan(result) | torch.isinf(result), 
                           torch.tensor(1e6, device=device, dtype=result.dtype), 
                           result)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        
        result = torch.mean(result)
        return result


# ==================== Activation Functions ====================

class MeanAct(nn.Module):
    """Mean activation for ZINB mean parameter"""
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        # Clamp input to avoid overflow in exp
        x = torch.clamp(x, min=-20, max=20)
        result = torch.exp(x)
        result = torch.clamp(result, min=1e-5, max=1e6)
        # Replace any NaN or Inf with safe values
        result = torch.where(torch.isnan(result) | torch.isinf(result), 
                           torch.tensor(1.0, device=result.device, dtype=result.dtype), 
                           result)
        return result


class DispAct(nn.Module):
    """Dispersion activation for ZINB dispersion parameter"""
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        # Clamp input to avoid overflow
        x = torch.clamp(x, min=-20, max=20)
        result = F.softplus(x)
        result = torch.clamp(result, min=1e-4, max=1e4)
        # Replace any NaN or Inf with safe values
        result = torch.where(torch.isnan(result) | torch.isinf(result), 
                           torch.tensor(1.0, device=result.device, dtype=result.dtype), 
                           result)
        return result


# ==================== Utility Functions ====================

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using Hungarian algorithm
    Args:
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    Returns:
        accuracy in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum(w[row_ind, col_ind]) * 1.0 / y_pred.size


def to_dense(X):
    """
    Convert sparse or dense matrix to dense numpy array
    Args:
        X: input matrix (scipy sparse or numpy array)
    Returns:
        dense numpy array
    """
    if hasattr(X, 'todense'):
        # scipy sparse matrix
        return np.array(X.todense())
    elif hasattr(X, 'toarray'):
        # other sparse formats
        return X.toarray()
    else:
        # already dense
        return np.array(X)


def check_matrix(X):
    """
    Check if matrix is 2D and convert to appropriate format
    Args:
        X: input data
    Returns:
        2D numpy array
    """
    X = to_dense(X)
    if len(X.shape) != 2:
        raise ValueError(f"Expected 2D matrix, got shape {X.shape}")
    return X
