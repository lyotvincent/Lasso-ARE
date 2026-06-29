"""
recluster_scRNA: AnnData-based interface for LassoARE
Handles all AnnData-specific operations and calls matrix-based LassoARE
Input: single-cell AnnData object
Output: modified AnnData object with clustering results
"""
from .lasso_ARE import LassoARE
import scanpy as sc
import torch
import numpy as np
import scipy.sparse
import scanpy.external.pp as pp

def fast_neighbors_faiss(adata, use_rep, n_neighbors=100):
    from scipy.sparse import csr_matrix
    import faiss
    """使用 FAISS 加速 neighbors 计算"""
    X = adata.obsm[use_rep].astype(np.float32)
    n_cells = X.shape[0]
    
    # 创建 FAISS 索引
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    
    # 搜索最近邻
    distances, indices = index.search(X, n_neighbors)
    
    # 构建 connectivities 和 distances 稀疏矩阵
    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = indices.flatten()
    dist_vals = distances.flatten()
    
    # 转换为 connectivity 权重
    sigma = np.mean(distances[:, 1])  # 使用第一个非自身邻居的平均距离
    conn_vals = np.exp(-dist_vals / (2 * sigma ** 2))
    
    connectivities = csr_matrix((conn_vals, (rows, cols)), shape=(n_cells, n_cells))
    distances_sparse = csr_matrix((dist_vals, (rows, cols)), shape=(n_cells, n_cells))
    
    # 存储到 adata
    adata.obsp['connectivities'] = connectivities
    adata.obsp['distances'] = distances_sparse
    adata.uns['neighbors'] = {
        'connectivities_key': 'connectivities',
        'distances_key': 'distances',
        'params': {'n_neighbors': n_neighbors, 'method': 'faiss'}
    }
    return adata


def recluster_with_lasso_are(adata, user_selected_lists, n_clusters=None, 
                             enc_pretrain_epoch=300, disc_pretrain_epoch=100, gan_epoch=50,
                             enc_layers=[256, 64], dec_layers=[64, 256], disc_layers=[128, 64],
                             batch_size=256, device=torch.device("cpu"),
                             lambda_attention=1.0, lambda_feature=1.0, lambda_consistency=1.0,
                             lambda_cluster=1.0, lambda_recon=1.0,
                             leiden_r=0.2, z_dim=32,
                             is_pca=False, pca_dimension=500, do_pp=False):
    """
    Recluster single-cell data using Lasso-guided Adversarial autoencoding
    
    Args:
        adata: AnnData object containing single-cell data
        user_selected_lists: list of lists of cell indices for user guidance
                           e.g., [[1,2,3], [10,11,12]] for 2 groups
                           or single list [1,2,3] for 1 group
        n_clusters: number of clusters (auto-detected if None)
        enc_pretrain_epoch: epochs for pretraining encoder
        disc_pretrain_epoch: epochs for pretraining discriminator
        gan_epoch: epochs for adversarial training
        enc_layers: encoder layer dimensions
        dec_layers: decoder layer dimensions
        disc_layers: discriminator layer dimensions
        batch_size: batch size
        device: torch device
        lambda_attention: weight for attention loss
        lambda_feature: weight for feature separation loss
        lambda_consistency: weight for consistency loss
        lambda_cluster: weight for clustering loss
        lambda_recon: weight for reconstruction loss
        leiden_r: resolution for Leiden clustering
        z_dim: latent dimension
        is_pca: whether to apply PCA dimensionality reduction
        pca_dimension: number of principal components to retain (default: 500)
        
    Returns:
        adata: modified AnnData object with clustering results in:
            - adata.obs['LassoARE_clusters']: cluster labels
            - adata.obsm['LassoARE_latent']: latent representations
            - adata.obsm['X_umap_LassoARE']: UMAP embeddings
    """
    # Auto-detect number of clusters
    if n_clusters is None:
        if 'codes' in adata.obs.columns:
            annotations = adata.obs['codes']
        elif 'annotation' in adata.obs.columns:
            annotations = adata.obs['annotation']
        elif 'anno-NK' in adata.obs.columns:
            annotations = adata.obs['anno-NK']
        elif 'leiden' in adata.obs.columns:
            annotations = adata.obs['leiden']
        else:
            sc.pp.neighbors(adata)
            sc.tl.leiden(adata)
            annotations = adata.obs['leiden']
        
        n_clusters = len(np.unique(annotations)) + 1
        print(f"Auto-detected {n_clusters} clusters")
    
    # Apply PCA if requested (before extracting data)
    original_dim = adata.shape[1]
    if is_pca:
        print(f"Applying PCA: reducing from {original_dim} to {pca_dimension} dimensions...")
        pca_dimension = min(pca_dimension, adata.shape[0], adata.shape[1])
        
        # Use scanpy's PCA on normalized data
        sc.tl.pca(adata, n_comps=pca_dimension, svd_solver='arpack')
        # Only apply harmony if batch column exists
        if 'batch' in adata.obs.columns:
            pp.harmony_integrate(adata, key='batch')
            adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
            print(f"PCA on normalized data completed with Harmony integration. Explained variance ratio: {adata.uns['pca']['variance_ratio'].sum():.4f}")
        else:
            print(f"PCA on normalized data completed (no batch correction). Explained variance ratio: {adata.uns['pca']['variance_ratio'].sum():.4f}")
        
        # For raw counts, we need to apply PCA manually on raw data
        if hasattr(adata, 'raw') and adata.raw is not None:
            # Create temporary adata for raw counts PCA
            raw_adata = sc.AnnData(X=adata.raw.X, obs=adata.obs.copy())
            sc.tl.pca(raw_adata, n_comps=pca_dimension, svd_solver='arpack')
            if 'batch' in raw_adata.obs.columns:
                pp.harmony_integrate(raw_adata, key='batch')
                raw_adata.obsm['X_pca'] = raw_adata.obsm['X_pca_harmony']
            
            # Store raw PCA results
            adata.uns['LassoARE_pca_raw'] = {
                'pca': raw_adata.varm['PCs'],  # PCA components
                'variance': raw_adata.uns['pca']['variance'],
                'variance_ratio': raw_adata.uns['pca']['variance_ratio']
            }
            raw_X_pca = raw_adata.obsm['X_pca']
            print(f"PCA on raw counts completed. Explained variance ratio: {raw_adata.uns['pca']['variance_ratio'].sum():.4f}")
        else:
            print("Warning: adata.raw not found. Using normalized data PCA for raw_X.")
            raw_X_pca = adata.obsm['X_pca'].copy()
        
        # Store PCA metadata
        adata.uns['LassoARE_pca'] = {
            'original_dim': original_dim,
            'pca_dim': pca_dimension,
            'explained_variance_ratio': adata.uns['pca']['variance_ratio'].sum(),
            'is_pca_applied': True
        }
    
    # Prepare data - extract matrices from AnnData
    if is_pca:
        # Use PCA-transformed data
        x = adata.obsm['X_pca']
        raw_X = raw_X_pca
    else:
        # Use original data
        x = adata.X.todense() if scipy.sparse.issparse(adata.X) else adata.X
        x = np.array(x)  # Ensure numpy array
        
        # Get raw counts
        if hasattr(adata, 'raw') and adata.raw is not None:
            raw_X = adata.raw.X.todense() if scipy.sparse.issparse(adata.raw.X) else adata.raw.X
            raw_X = np.array(raw_X)
        else:
            print("Warning: adata.raw not found. Using adata.X as raw counts.")
            raw_X = x.copy()
    
    # Get size factors
    size_factors = adata.obs.size_factors.values if 'size_factors' in adata.obs else np.ones(adata.shape[0])
    
    # Process user_selected_lists to list of lists
    if user_selected_lists is None:
        user_selected_lists = []
    elif not isinstance(user_selected_lists, list):
        user_selected_lists = [user_selected_lists]
    elif len(user_selected_lists) > 0 and not isinstance(user_selected_lists[0], (list, np.ndarray)):
        # Single list, wrap it
        user_selected_lists = [user_selected_lists]
    
    # Determine number of user groups
    num_user_groups = len(user_selected_lists) if user_selected_lists else 0
    print(f"Number of user-selected groups: {num_user_groups}")
    
    # add selected informations
    selected_mask = np.zeros(adata.shape[0], dtype=bool)
    for group in user_selected_lists:
        selected_mask[group] = True
    adata.obs['LassoARE_selected'] = selected_mask
        
    # Initialize LassoARE clusterer with appropriate input dimension
    input_dim = x.shape[1]  # Use PCA-reduced dimension if PCA was applied
    lasso_are = LassoARE(
        input_dim=input_dim,
        z_dim=z_dim,
        n_clusters=n_clusters,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        disc_layers=disc_layers,
        num_user_groups=num_user_groups,
        residual=True,
        device=device
    )
    
    # Pretrain autoencoder
    print("Pretraining autoencoder...")
    lasso_are.pretrain_generator(
        X=x,
        X_raw=raw_X,
        size_factor=size_factors,
        batch_size=batch_size,
        epochs=enc_pretrain_epoch
    )
    
    # Train adversarial clustering model
    print("Training adversarial clustering model...")
    y_pred, latent = lasso_are.train_adversarial(
        X=x,
        X_raw=raw_X,
        size_factor=size_factors,
        user_selected_lists=user_selected_lists,
        n_epochs=gan_epoch,
        batch_size=batch_size,
        lambda_attention=lambda_attention,
        lambda_feature=lambda_feature,
        lambda_consistency=lambda_consistency,
        lambda_cluster=lambda_cluster,
        lambda_recon=lambda_recon,
        pretrain_disc_epochs=disc_pretrain_epoch
    )
    
    # Store results in AnnData
    adata.obs['LassoARE_clusters'] = y_pred.astype(np.int32)
    adata.obsm['LassoARE_latent'] = latent
    
    # Compute UMAP and Leiden on latent representation
    leiden_adata = adata.copy()
    if do_pp:
        sc.pp.neighbors(leiden_adata, use_rep='LassoARE_latent', n_neighbors=100)
        # sc.tl.paga(leiden_adata)
        sc.tl.umap(leiden_adata, min_dist=0.5, spread=0.8)
        sc.tl.leiden(leiden_adata, key_added='leiden_LassoARE', resolution=leiden_r)
    
        # Store UMAP and Leiden results
        adata.obsm['X_umap_LassoARE'] = leiden_adata.obsm['X_umap']
        adata.obs['leiden_LassoARE'] = leiden_adata.obs['leiden_LassoARE']
        
        print("Reclustering completed!")
    return adata


# Alias for backward compatibility
def recluster_with_gan(adata, user_selected_list, n_clusters=None, 
                      enc_pretrain_epoch=300, disc_pretrain_epoch=100, gan_epoch=50,
                      enc_layers=[256, 64], dec_layers=[64, 256], disc_layers=[128, 64],
                      batch_size=256, device=torch.device("cpu"),
                      lambda_attention=1.0, lambda_feature=1.0, lambda_consistency=1.0,
                      leiden_r=0.2,
                      is_pca=False, pca_dimension=500, use_rsc=False):
    """
    Backward compatibility wrapper for recluster_with_lasso_are
    Converts single user_selected_list to list of lists format
    """
    # Convert single list to list of lists
    if user_selected_list is not None:
        if not isinstance(user_selected_list, list):
            user_selected_list = [user_selected_list]
        user_selected_lists = [user_selected_list]
    else:
        user_selected_lists = []
    
    return recluster_with_lasso_are(
        adata=adata,
        user_selected_lists=user_selected_lists,
        n_clusters=n_clusters,
        enc_pretrain_epoch=enc_pretrain_epoch,
        disc_pretrain_epoch=disc_pretrain_epoch,
        gan_epoch=gan_epoch,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        disc_layers=disc_layers,
        batch_size=batch_size,
        device=device,
        lambda_attention=lambda_attention,
        lambda_feature=lambda_feature,
        lambda_consistency=lambda_consistency,
        leiden_r=leiden_r,
        is_pca=is_pca,
        pca_dimension=pca_dimension, 
    )
    
# Alias for backward compatibility
def recluster_with_existing_pca(adata, user_selected_lists, n_clusters=None, 
                             enc_pretrain_epoch=300, disc_pretrain_epoch=100, gan_epoch=50,
                             enc_layers=[256, 64], dec_layers=[64, 256], disc_layers=[128, 64],
                             batch_size=256, device=torch.device("cpu"),
                             lambda_attention=1.0, lambda_feature=1.0, lambda_consistency=1.0,
                             lambda_cluster=1.0, lambda_recon=1.0,
                             leiden_r=0.2, z_dim=32, do_pp=False, existing_harmony=False ,do_harmony=False):
    if 'X_pca' not in adata.obsm:
        raise ValueError("adata.obsm['X_pca'] 不存在，请先在外部完成 PCA 再调用该函数。")
    
    # 若需要在本函数中执行 Harmony
    if do_harmony:
        if 'batch' not in adata.obs.columns:
            print("Warning: 未找到 batch 列，跳过 Harmony。")
        else:
            pp.harmony_integrate(adata, key='batch')
            adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
            existing_harmony = True
    
    # 选择使用的 PCA 表征
    if existing_harmony and 'X_pca_harmony' in adata.obsm:
        x = adata.obsm['X_pca_harmony']
    else:
        x = adata.obsm['X_pca']
    
    # 原始计数（若维度与 PCA 不一致则回退为 PCA 表征，避免 ZINB 维度错误）
    if hasattr(adata, 'raw') and adata.raw is not None:
        raw_X = adata.raw.X.todense() if scipy.sparse.issparse(adata.raw.X) else adata.raw.X
        raw_X = np.array(raw_X)
    else:
        raw_X = adata.X.todense() if scipy.sparse.issparse(adata.X) else adata.X
        raw_X = np.array(raw_X)

    if raw_X.shape[1] != x.shape[1]:
        print(f"Warning: raw_X dim {raw_X.shape[1]} != PCA dim {x.shape[1]}, fallback to PCA matrix for raw_X to match ZINB input.")
        raw_X = x
    
    # 自动估计簇数
    if n_clusters is None:
        if 'annotation' in adata.obs.columns:
            annotations = adata.obs['annotation']
        elif 'leiden' in adata.obs.columns:
            annotations = adata.obs['leiden']
        else:
            sc.pp.neighbors(adata)
            sc.tl.leiden(adata)
            annotations = adata.obs['leiden']
        n_clusters = len(np.unique(annotations)) + 1
        print(f"Auto-detected {n_clusters} clusters")
    
    print(f"Using PCA matrix with shape {x.shape} for LassoARE reclustering.")
    # 记录 PCA 元信息
    adata.uns['LassoARE_pca'] = {
        'original_dim': adata.shape[1],
        'pca_dim': x.shape[1],
        'explained_variance_ratio': adata.uns['pca']['variance_ratio'].sum() if 'pca' in adata.uns else None,
        'is_pca_applied': True,
        'used_existing_pca': True,
        'used_harmony': existing_harmony or do_harmony,
    }
    
    # size factor
    size_factors = adata.obs.size_factors.values if 'size_factors' in adata.obs else np.ones(adata.shape[0])
    
    # 规范化用户选区格式
    if user_selected_lists is None:
        user_selected_lists = []
    elif not isinstance(user_selected_lists, list):
        user_selected_lists = [user_selected_lists]
    elif len(user_selected_lists) > 0 and not isinstance(user_selected_lists[0], (list, np.ndarray)):
        user_selected_lists = [user_selected_lists]
    
    num_user_groups = len(user_selected_lists) if user_selected_lists else 0
    print(f"Number of user-selected groups: {num_user_groups}")
    
    selected_mask = np.zeros(adata.shape[0], dtype=bool)
    for group in user_selected_lists:
        selected_mask[group] = True
    adata.obs['LassoARE_selected'] = selected_mask
    
    # 构建并训练模型
    lasso_are = LassoARE(
        input_dim=x.shape[1],
        z_dim=z_dim,
        n_clusters=n_clusters,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        disc_layers=disc_layers,
        num_user_groups=num_user_groups,
        residual=True,
        device=device
    )
    
    print("Pretraining autoencoder...")
    lasso_are.pretrain_generator(
        X=x,
        X_raw=raw_X,
        size_factor=size_factors,
        batch_size=batch_size,
        epochs=enc_pretrain_epoch
    )
    
    print("Training adversarial clustering model...")
    y_pred, latent = lasso_are.train_adversarial(
        X=x,
        X_raw=raw_X,
        size_factor=size_factors,
        user_selected_lists=user_selected_lists,
        n_epochs=gan_epoch,
        batch_size=batch_size,
        lambda_attention=lambda_attention,
        lambda_feature=lambda_feature,
        lambda_consistency=lambda_consistency,
        lambda_cluster=lambda_cluster,
        lambda_recon=lambda_recon,
        pretrain_disc_epochs=disc_pretrain_epoch
    )
    
    adata.obs['LassoARE_clusters'] = y_pred.astype(np.int32)
    adata.obsm['LassoARE_latent'] = latent
    
    # 可选下游图嵌和聚类
    if do_pp:
        leiden_adata = adata.copy()
        sc.pp.neighbors(leiden_adata, use_rep='LassoARE_latent', n_neighbors=100)
        sc.tl.umap(leiden_adata, min_dist=0.5, spread=0.8)
        sc.tl.leiden(leiden_adata, key_added='leiden_LassoARE', resolution=leiden_r)
        adata.obsm['X_umap_LassoARE'] = leiden_adata.obsm['X_umap']
        adata.obs['leiden_LassoARE'] = leiden_adata.obs['leiden_LassoARE']
        print("Reclustering completed!")
    
    return adata