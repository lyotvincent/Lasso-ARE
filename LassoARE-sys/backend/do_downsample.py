def do_h5ad_downsample(adata, sample_rate=0.01, leiden_r=1.0, uniform_rate=0.5, add_col='orig_idxs', cluster_key='leiden', obsm_key='X_umap'):
    import scanpy as sc
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    if obsm_key not in adata.obsm:
        print(f'No {obsm_key} found, computing UMAP...')
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        obsm_key = 'X_umap'
    if cluster_key not in adata.obs:
        print(f'No {cluster_key} clustering found, computing leiden clustering...')
        sc.tl.leiden(adata, resolution=leiden_r)
        cluster_key = 'leiden'
    
    num_leiden_clusters = adata.obs[cluster_key].nunique()
    # Ensure we don't sample more than exists
    sample_cells_num = min(int(adata.n_obs * sample_rate) + num_leiden_clusters * 3, adata.n_obs)
    uniform_rate = min(1.0, max(0.0, uniform_rate)) # Clamp between 0 and 1

    # Calculate sample sizes
    n_uniform = int(sample_cells_num * uniform_rate)
    n_balanced = sample_cells_num - n_uniform

    all_indices = np.arange(adata.n_obs)
    
    # 1. Uniform Sampling
    if n_uniform > 0:
        uniform_indices = np.random.choice(all_indices, size=n_uniform, replace=False)
    else:
        uniform_indices = np.array([], dtype=int)

    # 2. Balanced Sampling (per Leiden cluster)
    balanced_indices = []
    if n_balanced > 0:
        # Distribute n_balanced across clusters
        n_per_cluster = n_balanced // num_leiden_clusters
        
        # Ensure at least 1 per cluster if n_balanced is positive but small
        if n_per_cluster < 1:
             n_per_cluster = 1
        
        for cluster in adata.obs[cluster_key].unique():
            cluster_mask = adata.obs[cluster_key] == cluster
            cluster_indices = all_indices[cluster_mask]
            
            # Sample min(available, target)
            n_sample = min(len(cluster_indices), n_per_cluster)
            if n_sample > 0:
                sampled = np.random.choice(cluster_indices, size=n_sample, replace=False)
                balanced_indices.extend(sampled)
    
    balanced_indices = np.array(balanced_indices, dtype=int)

    # Combine indices
    final_indices = np.unique(np.concatenate([uniform_indices, balanced_indices]))
    
    # Create sampled adata
    adata_sampled = adata[final_indices].copy()
    
    if add_col:
        adata_sampled.obs[add_col] = final_indices

    # Find nearest neighbors in UMAP space
    X_umap_original = adata.obsm[obsm_key]
    X_umap_sampled = adata_sampled.obsm[obsm_key]
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_umap_sampled)
    distances, indices = nbrs.kneighbors(X_umap_original)
    
    # indices is (n_obs, 1) containing indices into X_umap_sampled (i.e., 0 to n_sampled-1)
    nearest_ids = indices.flatten()
    
    return adata_sampled, nearest_ids

def get_nearest_ids(adata, downsampled_adata, obsm_key='X_umap'):
    from sklearn.neighbors import NearestNeighbors
    X_umap_original = adata.obsm[obsm_key]
    X_umap_sampled = downsampled_adata.obsm[obsm_key]
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_umap_sampled)
    distances, indices = nbrs.kneighbors(X_umap_original)
    
    nearest_ids = indices.flatten()
    return nearest_ids

def recover_full_list(selected_list, nearest_ids):
    # after selected some cells in downsampled data, recover the full list
    import numpy as np
    selected_set = set(selected_list)
    # Vectorized check
    mask = np.isin(nearest_ids, list(selected_set))
    # Get indices where mask is True
    full_selected_list = np.where(mask)[0]
    return full_selected_list
    
    