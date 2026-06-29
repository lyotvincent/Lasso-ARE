"""
Example usage of refactored LassoARE module

This file demonstrates how to use the refactored code with both:
1. AnnData objects (for scRNA-seq data)
2. Pure matrix operations (for general purpose clustering)
"""

import torch
import numpy as np

# ==================== Example 1: Using with AnnData (scRNA-seq) ====================
def example_with_anndata():
    """
    Example of using LassoARE with AnnData objects
    This is the recommended approach for single-cell RNA-seq data
    """
    import scanpy as sc
    from LassoARE.recluster_scRNA import recluster_with_lasso_are
    
    # Load your AnnData object
    # adata = sc.read_h5ad('your_data.h5ad')
    
    # Or create a dummy dataset for demonstration
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    
    # Define user-selected groups (multiple groups)
    # Each group is a list of cell indices
    user_selected_groups = [
        [0, 1, 2, 10, 20],      # Group 1: cells you think belong together
        [50, 51, 52, 60, 70],   # Group 2: another group of cells
        [100, 101, 102, 110]    # Group 3: yet another group
    ]
    
    # Or use a single group
    # user_selected_groups = [[0, 1, 2, 10, 20]]
    
    # Run clustering with user guidance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    adata_clustered = recluster_with_lasso_are(
        adata=adata,
        user_selected_lists=user_selected_groups,
        n_clusters=10,                    # Number of clusters (auto-detected if None)
        enc_pretrain_epoch=100,           # Reduced for quick demo
        disc_pretrain_epoch=20,
        gan_epoch=50,
        enc_layers=[256, 64],
        dec_layers=[64, 256],
        disc_layers=[128, 64],
        batch_size=256,
        device=device,
        lambda_attention=1.0,
        lambda_feature=1.0,
        lambda_cluster=1.0,
        lambda_recon=1.0,
        leiden_r=0.5,
        z_dim=32
    )
    
    # Access results
    print("Cluster labels:", adata_clustered.obs['LassoARE_clusters'])
    print("Latent representation shape:", adata_clustered.obsm['LassoARE_latent'].shape)
    print("UMAP shape:", adata_clustered.obsm['X_umap_LassoARE'].shape)
    
    # Visualize results
    sc.pl.umap(adata_clustered, color='LassoARE_clusters', title='LassoARE Clusters')
    sc.pl.umap(adata_clustered, color='leiden_LassoARE', title='Leiden on LassoARE')
    
    return adata_clustered


# ==================== Example 2: Using with Pure Matrices ====================
def example_with_matrices():
    """
    Example of using LassoARE with pure numpy matrices
    This approach is more flexible and can be used for any 2D data
    """
    from LassoARE.lasso_ARE import LassoARE
    
    # Create dummy data
    n_samples = 1000
    n_features = 500
    
    # Normalized data (e.g., log-normalized gene expression)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Raw counts (for ZINB loss)
    X_raw = np.random.poisson(lam=10, size=(n_samples, n_features)).astype(np.float32)
    
    # Size factors
    size_factors = np.ones(n_samples, dtype=np.float32)
    
    # Define user-selected groups
    user_groups = [
        [0, 1, 2, 10, 20],
        [100, 101, 102, 110, 120],
        [500, 501, 502, 510]
    ]
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lasso_are = LassoARE(
        input_dim=n_features,
        z_dim=32,
        n_clusters=10,
        enc_layers=[256, 64],
        dec_layers=[64, 256],
        disc_layers=[128, 64],
        num_user_groups=len(user_groups),
        residual=True,
        device=device
    )
    
    # Pretrain autoencoder
    lasso_are.pretrain_generator(
        X=X,
        X_raw=X_raw,
        size_factor=size_factors,
        batch_size=256,
        lr=0.001,
        epochs=100
    )
    
    # Train with adversarial learning
    y_pred, latent = lasso_are.train_adversarial(
        X=X,
        X_raw=X_raw,
        size_factor=size_factors,
        user_selected_lists=user_groups,
        n_epochs=50,
        batch_size=256,
        lr_gen=0.001,
        lr_disc=0.001,
        lambda_cluster=1.0,
        lambda_recon=1.0,
        lambda_attention=1.0,
        lambda_feature=1.0,
        lambda_consistency=1.0,
        pretrain_disc_epochs=20
    )
    
    print("Predicted clusters:", y_pred)
    print("Latent representation shape:", latent.shape)
    
    return y_pred, latent


# ==================== Example 3: Using scARE alone (without adversarial training) ====================
def example_scARE_only():
    """
    Example of using scARE (autoencoder) without adversarial training
    Useful if you don't need user-guided clustering
    """
    from LassoARE.scARE import scARE
    
    # Create dummy data
    n_samples = 1000
    n_features = 500
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    X_raw = np.random.poisson(lam=10, size=(n_samples, n_features)).astype(np.float32)
    size_factors = np.ones(n_samples, dtype=np.float32)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = scARE(
        input_dim=n_features,
        z_dim=32,
        n_clusters=10,
        encodeLayer=[256, 64],
        decodeLayer=[64, 256],
        device=device
    )
    
    # Pretrain autoencoder
    model.pretrain_autoencoder(
        X=X,
        X_raw=X_raw,
        size_factor=size_factors,
        batch_size=256,
        lr=0.001,
        epochs=100
    )
    
    # Fit clustering model
    y_pred, acc, nmi, ari, final_epoch = model.fit(
        X=X,
        X_raw=X_raw,
        sf=size_factors,
        batch_size=256,
        num_epochs=50,
        lr=1.0
    )
    
    print("Predicted clusters:", y_pred)
    print(f"Final metrics - ACC: {acc}, NMI: {nmi}, ARI: {ari}")
    
    return y_pred, model


# ==================== Example 4: Backward compatibility ====================
def example_backward_compatibility():
    """
    Example showing backward compatibility with original API
    """
    import scanpy as sc
    from LassoARE.recluster_scRNA import recluster_with_gan  # Old function name
    
    # Load data
    adata = sc.datasets.pbmc3k()
    # ... preprocessing ...
    
    # Old API: single list of indices
    user_selected_cells = [0, 1, 2, 10, 20]
    
    # This still works!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata_clustered = recluster_with_gan(
        adata=adata,
        user_selected_list=user_selected_cells,  # Single list
        n_clusters=10,
        enc_pretrain_epoch=100,
        disc_pretrain_epoch=20,
        gan_epoch=50,
        device=device
    )
    
    return adata_clustered


if __name__ == "__main__":
    print("=" * 80)
    print("LassoARE Usage Examples")
    print("=" * 80)
    
    # Uncomment the example you want to run:
    
    # Example 1: With AnnData (recommended for scRNA-seq)
    # adata = example_with_anndata()
    
    # Example 2: With pure matrices (general purpose)
    # y_pred, latent = example_with_matrices()
    
    # Example 3: scARE only (no adversarial training)
    # y_pred, model = example_scARE_only()
    
    # Example 4: Backward compatibility
    # adata = example_backward_compatibility()
    
    print("\nPlease uncomment one of the examples in __main__ to run.")
