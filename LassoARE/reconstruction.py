"""
recluster_scRNA: AnnData-based interface for LassoARE
Handles all AnnData-specific operations and calls matrix-based LassoARE
Input: single-cell AnnData object
Output: modified AnnData object with clustering results
"""
from .lasso_ARE import LassoARE
from .scARE import scARE, buildNetwork
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.sparse
import scanpy.external.pp as pp
import math
from torch.utils.data import DataLoader, TensorDataset
from .utils import check_matrix

def reconstruction_with_lasso_are(adata, user_selected_lists, using_emb=None, n_clusters=None, 
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
        using_emb: the embedding for reconstruction based on (default: None, use adata.X)
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
    
    # Determine input data source
    if using_emb is not None:
        print(f"Using embedding {using_emb} as input data")
        input_data = adata.obsm[using_emb]
        if scipy.sparse.issparse(input_data):
            input_data = input_data.todense()
        input_data = np.array(input_data)
        original_dim = input_data.shape[1]
        
        # Create temp adata for processing (esp. for Harmony)
        process_adata = sc.AnnData(X=input_data, obs=adata.obs.copy())
        has_raw = False
    else:
        process_adata = adata
        original_dim = adata.shape[1]
        has_raw = hasattr(adata, 'raw') and adata.raw is not None

    # Apply PCA if requested (before extracting data)
    if is_pca:
        print(f"Applying PCA: reducing from {original_dim} to {pca_dimension} dimensions...")
        pca_dimension = min(pca_dimension, process_adata.shape[0], process_adata.shape[1])
        
        # Use scanpy's PCA on input data
        sc.tl.pca(process_adata, n_comps=pca_dimension, svd_solver='arpack')
        # Only apply harmony if batch column exists
        if 'batch' in process_adata.obs.columns:
            pp.harmony_integrate(process_adata, key='batch')
            process_adata.obsm['X_pca'] = process_adata.obsm['X_pca_harmony']
            print(f"PCA on input data completed with Harmony integration. Explained variance ratio: {process_adata.uns['pca']['variance_ratio'].sum():.4f}")
        else:
            print(f"PCA on input data completed (no batch correction). Explained variance ratio: {process_adata.uns['pca']['variance_ratio'].sum():.4f}")
        
        # For raw counts, we need to apply PCA manually on raw data
        if using_emb is None and has_raw:
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
            if using_emb is None:
                print("Warning: adata.raw not found. Using normalized data PCA for raw_X.")
            raw_X_pca = process_adata.obsm['X_pca'].copy()
        
        # Store PCA metadata
        if using_emb is None:
            adata.uns['LassoARE_pca'] = {
                'original_dim': original_dim,
                'pca_dim': pca_dimension,
                'explained_variance_ratio': process_adata.uns['pca']['variance_ratio'].sum(),
                'is_pca_applied': True
            }
    
    # Prepare data - extract matrices from AnnData
    if is_pca:
        # Use PCA-transformed data
        x = process_adata.obsm['X_pca']
        raw_X = raw_X_pca
    else:
        # Use data from process_adata
        x = process_adata.X
        if scipy.sparse.issparse(x):
            x = x.todense()
        x = np.array(x)  # Ensure numpy array
        
        # Get raw counts
        if using_emb is None and has_raw:
            raw_X = adata.raw.X.todense() if scipy.sparse.issparse(adata.raw.X) else adata.raw.X
            raw_X = np.array(raw_X)
        else:
            if using_emb is None:
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
        
        print("Reconstruction completed!")
    return adata


def reconstruction_with_ref(adata, user_selected_lists, using_emb, n_clusters=None,
                             enc_pretrain_epoch=300, disc_pretrain_epoch=100, gan_epoch=50,
                             ref_pretrain_epoch=200,
                             enc_layers=[256, 64], dec_layers=[64, 256], disc_layers=[128, 64],
                             ref_enc_layers=[256, 64],
                             batch_size=256, device=torch.device("cpu"),
                             lambda_attention=1.0, lambda_feature=1.0, lambda_consistency=1.0,
                             lambda_cluster=1.0, lambda_recon=1.0, lambda_ref=0.3,
                             leiden_r=0.2, z_dim=32,
                             is_pca=False, pca_dimension=500, do_pp=False):
    """
    Reconstruction with expression reference guidance.

    When the input embedding has low quality, the model can leverage adata.X as a soft
    reference to improve reconstruction fidelity without overriding the embedding signal.

    Strategy
    --------
    1. Pretrain a lightweight **reference encoder** on adata.X (or adata.raw.X) so that
       it learns a biologically meaningful latent space from expression data.
    2. Pretrain the main autoencoder using the expression data so the decoder captures the
       gene-expression manifold.
    3. During adversarial training, the main encoder takes `using_emb` as input.  A
       *reference alignment loss*  ``MSE(z_emb, z_ref.detach())`` is added to the
       generator objective with weight ``lambda_ref``.  Because ``z_ref`` is detached,
       the reference encoder is **not** updated—only the main encoder is nudged toward the
       expression-derived latent space.

    The balance is controlled by ``lambda_ref``:
    - ``lambda_ref = 0``  → pure embedding-driven (same as ``reconstruction_with_lasso_are``
      with ``using_emb``).
    - ``lambda_ref > 0``  → expression provides a soft anchor; raise it to make expression
      dominate (but set it too large and the embedding signal is lost).
    A typical starting value is ``0.1–0.5``.

    Parameters
    ----------
    ref_enc_layers : list
        Reference encoder hidden layer dimensions (input → z_dim).
    lambda_ref : float
        Weight for expression-reference alignment loss.  Recommended range: 0.1–0.5.

    Returns
    -------
    adata : AnnData
        Modified AnnData with:
        - ``adata.obs['LassoARE_clusters']``: cluster labels
        - ``adata.obsm['LassoARE_latent']``: latent representations
        - ``adata.obsm['X_umap_LassoARE']``: UMAP embeddings (if do_pp=True)
        - ``adata.obs['leiden_LassoARE']``: Leiden clusters (if do_pp=True)
    """
    # ------------------------------------------------------------------
    # 1. Auto-detect number of clusters
    # ------------------------------------------------------------------
    if n_clusters is None:
        for col in ('codes', 'annotation', 'anno-NK', 'leiden'):
            if col in adata.obs.columns:
                annotations = adata.obs[col]
                break
        else:
            sc.pp.neighbors(adata)
            sc.tl.leiden(adata)
            annotations = adata.obs['leiden']
        n_clusters = len(np.unique(annotations)) + 1
        print(f"Auto-detected {n_clusters} clusters")

    # ------------------------------------------------------------------
    # 2. Extract main embedding (used as encoder input)
    # ------------------------------------------------------------------
    print(f"Using embedding '{using_emb}' as main encoder input")
    emb_data = adata.obsm[using_emb]
    print(f"Original embedding shape: {emb_data.shape}, dtype: {emb_data.dtype}")
    if scipy.sparse.issparse(emb_data):
        emb_data = emb_data.todense()
    emb_data = np.array(emb_data, dtype=np.float32)

    if is_pca:
        pca_dim = min(pca_dimension, emb_data.shape[0], emb_data.shape[1])
        print(f"Applying PCA to embedding: {emb_data.shape[1]} → {pca_dim}")
        emb_adata = sc.AnnData(X=emb_data, obs=adata.obs.copy())
        sc.tl.pca(emb_adata, n_comps=pca_dim, svd_solver='arpack')
        if 'batch' in emb_adata.obs.columns:
            pp.harmony_integrate(emb_adata, key='batch')
            emb_adata.obsm['X_pca'] = emb_adata.obsm['X_pca_harmony']
        emb_data = emb_adata.obsm['X_pca'].astype(np.float32)

    # ------------------------------------------------------------------
    # 3. Extract expression data (reference)
    # ------------------------------------------------------------------
    has_raw = hasattr(adata, 'raw') and adata.raw is not None

    expr_X = adata.X
    if scipy.sparse.issparse(expr_X):
        expr_X = np.array(expr_X.todense(), dtype=np.float32)
    else:
        expr_X = np.array(expr_X, dtype=np.float32)

    if has_raw:
        raw_X = adata.raw.X
        if scipy.sparse.issparse(raw_X):
            raw_X = np.array(raw_X.todense(), dtype=np.float32)
        else:
            raw_X = np.array(raw_X, dtype=np.float32)
    else:
        print("Warning: adata.raw not found. Using adata.X as raw counts for ZINB loss.")
        raw_X = expr_X.copy()

    size_factors = (adata.obs['size_factors'].values.astype(np.float32)
                    if 'size_factors' in adata.obs
                    else np.ones(adata.shape[0], dtype=np.float32))

    # ------------------------------------------------------------------
    # 4. Process user_selected_lists
    # ------------------------------------------------------------------
    if user_selected_lists is None:
        user_selected_lists = []
    elif not isinstance(user_selected_lists, list):
        user_selected_lists = [user_selected_lists]
    elif (len(user_selected_lists) > 0
          and not isinstance(user_selected_lists[0], (list, np.ndarray))):
        user_selected_lists = [user_selected_lists]

    num_user_groups = len(user_selected_lists)
    print(f"Number of user-selected groups: {num_user_groups}")

    selected_mask = np.zeros(adata.shape[0], dtype=bool)
    for group in user_selected_lists:
        selected_mask[group] = True
    adata.obs['LassoARE_selected'] = selected_mask

    # ------------------------------------------------------------------
    # 5. Build & pretrain the main LassoARE (on expression data so the
    #    decoder learns the gene-expression manifold)
    # ------------------------------------------------------------------
    expr_dim = expr_X.shape[1]
    emb_dim = emb_data.shape[1]

    # Main model: encoder input dim = emb_dim, decoder output dim = expr_dim
    # We build this manually so that encoder / decoder can have different widths.
    main_lasso = LassoARE(
        input_dim=emb_dim,          # encoder sees embedding
        z_dim=z_dim,
        n_clusters=n_clusters,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        disc_layers=disc_layers,
        num_user_groups=num_user_groups,
        residual=True,
        device=device
    )

    # Override decoder output to match expression dimensionality
    if expr_dim != emb_dim:
        main_lasso.generator._dec_mean = nn.Sequential(
            nn.Linear(dec_layers[-1] if dec_layers else z_dim, expr_dim),
            __import__('LassoARE.utils', fromlist=['MeanAct']).MeanAct()
        ).to(device)
        main_lasso.generator._dec_disp = nn.Sequential(
            nn.Linear(dec_layers[-1] if dec_layers else z_dim, expr_dim),
            __import__('LassoARE.utils', fromlist=['DispAct']).DispAct()
        ).to(device)
        main_lasso.generator._dec_pi = nn.Sequential(
            nn.Linear(dec_layers[-1] if dec_layers else z_dim, expr_dim),
            nn.Sigmoid()
        ).to(device)
        # Also update input_dim-dependent layers in the generator
        # so ZINB decodes properly – input_dim is only used for these output heads.
        main_lasso.generator.input_dim = expr_dim

    # Pretrain main autoencoder: temporarily feed expression data so the
    # decoder learns gene space.  Encoder gets a projection from expr→emb's latent.
    # Since expr_dim != emb_dim we train a bridge: we pass expression through
    # the encoder that is also sized for emb_dim.
    # Practical approach: pretrain only the decoder+output-heads on expression
    # by using the reference encoder latent as "z" proxy.
    print("Pretraining reference encoder on expression data...")
    ref_input_dim = expr_X.shape[1]
    ref_enc_full_layers = [ref_input_dim] + ref_enc_layers + [z_dim]
    ref_encoder = buildNetwork(ref_enc_full_layers, type="encode",
                               activation="relu", residual=True).to(device)
    print(f'ref_encoder architecture: {ref_enc_full_layers}')
    # Pretrain reference encoder with a simple MSE autoencoder objective
    # (we only need the encoder; the main decoder will be calibrated separately)
    ref_optimizer = optim.Adam(ref_encoder.parameters(), lr=1e-3, amsgrad=True)
    ref_X_tensor = torch.tensor(expr_X, dtype=torch.float32)
    ref_raw_tensor = torch.tensor(raw_X, dtype=torch.float32)
    sf_tensor_ref = torch.tensor(size_factors, dtype=torch.float32)

    print('loading reference dataset into DataLoader for pretraining...')
    ref_dataset = TensorDataset(ref_X_tensor, ref_raw_tensor, sf_tensor_ref)
    ref_loader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=True,
                            pin_memory=device.type == "cuda")
    
    print("Pretraining reference encoder with ZINB loss...")
    # A small linear projection + ZINB head for pretraining the ref encoder
    _dec_last_dim = dec_layers[-1] if dec_layers else z_dim
    ref_dec_proj = nn.Linear(z_dim, _dec_last_dim).to(device)
    ref_dec_mean = nn.Sequential(nn.Linear(_dec_last_dim, expr_dim),
                                 __import__('LassoARE.utils', fromlist=['MeanAct']).MeanAct()
                                 ).to(device)
    ref_dec_disp = nn.Sequential(nn.Linear(_dec_last_dim, expr_dim),
                                 __import__('LassoARE.utils', fromlist=['DispAct']).DispAct()
                                 ).to(device)
    ref_dec_pi = nn.Sequential(nn.Linear(_dec_last_dim, expr_dim),
                               nn.Sigmoid()).to(device)

    from .utils import ZINBLoss
    zinb = ZINBLoss().to(device)

    ref_opt_all = optim.Adam(
        list(ref_encoder.parameters()) +
        list(ref_dec_proj.parameters()) +
        list(ref_dec_mean.parameters()) +
        list(ref_dec_disp.parameters()) +
        list(ref_dec_pi.parameters()),
        lr=1e-3, amsgrad=True
    )
    print("Pretraining training loop for reference encoder...")
    ref_encoder.train()
    for ep in range(ref_pretrain_epoch):
        if (ep + 1) % 20 == 0 or ep == ref_pretrain_epoch - 1:
            print(f"Ref encoder pretrain [{ep+1}/{ref_pretrain_epoch}]", sep="", end="")
        for xb, xrb, sfb in ref_loader:
            xb = xb.to(device, non_blocking=True)
            xrb = xrb.to(device, non_blocking=True)
            sfb = sfb.to(device, non_blocking=True)

            h = ref_encoder(xb)
            h_dec = ref_dec_proj(h)
            mean_ = ref_dec_mean(h_dec)
            disp_ = ref_dec_disp(h_dec)
            pi_ = ref_dec_pi(h_dec)
            loss = zinb(x=xrb, mean=mean_, disp=disp_, pi=pi_, scale_factor=sfb)

            ref_opt_all.zero_grad()
            loss.backward()
            ref_opt_all.step()

        if (ep + 1) % 20 == 0 or ep == ref_pretrain_epoch - 1:
            print(f" , ZINB loss: {loss.item():.4f}")

    ref_encoder.eval()
    # Freeze reference encoder — it is used only for soft guidance
    for param in ref_encoder.parameters():
        param.requires_grad_(False)

    # ------------------------------------------------------------------
    # KEY OPTIMISATION: pre-compute z_ref for ALL cells ONCE.
    #
    # Why this matters:
    #   The naive approach puts expr_X (n_cells × n_genes, e.g. ×20 000) into
    #   the DataLoader AND calls ref_encoder() every batch.  That is:
    #     • n_genes/z_dim × more tensor I/O per batch  (e.g. 20 000/32 ≈ 625×)
    #     • One full ref_encoder forward pass per batch even though the weights
    #       are frozen and the output would always be the same for the same cell.
    #
    #   Because ref_encoder is frozen we can encode the whole dataset once and
    #   store z_ref [n_cells, z_dim] on CPU.  DataLoader then carries only the
    #   tiny z_dim tensor instead of n_genes, and the training loop just indexes
    #   the pre-computed tensor — zero encoder cost.
    # ------------------------------------------------------------------
    print("Pre-computing reference latent representations z_ref (one-off)...")
    _zref_parts = []
    with torch.no_grad():
        for _s in range(0, len(expr_X), batch_size):
            _e = min(_s + batch_size, len(expr_X))
            _xb = torch.tensor(expr_X[_s:_e], dtype=torch.float32, device=device)
            _zref_parts.append(ref_encoder(_xb).cpu())
    z_ref_all = torch.cat(_zref_parts, dim=0)  # [n_cells, z_dim]  — stays on CPU
    print(f"  z_ref shape: {tuple(z_ref_all.shape)}  "
          f"(replaced {expr_X.shape[1]}-dim expression → {z_ref_all.shape[1]}-dim latent in DataLoader)")
    del _zref_parts  # free memory

    # ------------------------------------------------------------------
    # 6. Pretrain main autoencoder using the embedding as input
    #    and expression raw counts as reconstruction target.
    # ------------------------------------------------------------------
    print("Pretraining main autoencoder (embedding → expression)...")

    main_lasso.pretrain_generator(
        X=emb_data,
        X_raw=raw_X,
        size_factor=size_factors,
        batch_size=batch_size,
        epochs=enc_pretrain_epoch
    )

    # ------------------------------------------------------------------
    # 7. Adversarial training with reference alignment loss
    # ------------------------------------------------------------------
    print("Training adversarial model with expression reference guidance...")

    from .lasso_ARE import MultiDiscriminator
    from sklearn.cluster import KMeans

    X_check = check_matrix(emb_data)
    X_raw_check = check_matrix(raw_X)
    sf_check = size_factors

    processed_lists = []
    for ul in user_selected_lists:
        if ul is None:
            continue
        if not isinstance(ul, (list, np.ndarray)):
            ul = [ul]
        valid = [i for i in ul if 0 <= i < X_check.shape[0]]
        if valid:
            processed_lists.append(valid)
    user_selected_lists = processed_lists
    num_groups = len(user_selected_lists)

    # Possibly reinit discriminator
    if num_groups != main_lasso.num_user_groups:
        main_lasso.num_user_groups = num_groups
        main_lasso.discriminator = MultiDiscriminator(
            latent_dim=z_dim,
            n_clusters=n_clusters,
            num_groups=num_groups,
            hidden_dims=disc_layers
        ).to(device)

    X_tensor = torch.tensor(X_check, dtype=torch.float32)
    X_raw_tensor = torch.tensor(X_raw_check, dtype=torch.float32)
    sf_tensor = torch.tensor(sf_check, dtype=torch.float32)
    # z_ref_all already computed above — z_dim only (NOT n_genes); no encoder call needed in loop

    dataset = TensorDataset(
        torch.arange(len(X_tensor)),
        X_tensor,
        X_raw_tensor,
        sf_tensor,
        z_ref_all   # [n_cells, z_dim]  ← tiny compared to raw expression
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            pin_memory=device.type == "cuda")

    gen_optimizer = optim.Adam(main_lasso.generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(main_lasso.discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=20, gamma=0.5)
    disc_scheduler = optim.lr_scheduler.StepLR(disc_optimizer, step_size=20, gamma=0.5)

    user_sets = [set(ul) for ul in user_selected_lists]

    # Initialize cluster centres
    with torch.no_grad():
        lat_init = main_lasso.generator.encodeBatch(X_tensor)
        if num_groups > 0 and num_groups < n_clusters:
            grp_centers = []
            for ul in user_selected_lists:
                if ul:
                    grp_centers.append(lat_init[ul].mean(0, keepdim=True).cpu().numpy())
            all_sel = np.zeros(X_check.shape[0], dtype=bool)
            for ul in user_selected_lists:
                all_sel[ul] = True
            other_feat = lat_init[~all_sel].cpu().numpy()
            n_other = n_clusters - len(grp_centers)
            if other_feat.shape[0] >= n_other > 0:
                km = KMeans(n_clusters=n_other, n_init=20, random_state=42)
                other_ctr = km.fit(other_feat).cluster_centers_
                centers = np.vstack([other_ctr] + grp_centers)
            else:
                centers = np.vstack(grp_centers) if grp_centers else lat_init.cpu().numpy()
                while centers.shape[0] < n_clusters:
                    ri = np.random.choice(lat_init.shape[0])
                    centers = np.vstack([centers, lat_init[ri].cpu().numpy().reshape(1, -1)])
        else:
            km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
            centers = km.fit(lat_init.cpu().numpy()).cluster_centers_
        main_lasso.generator.mu.data.copy_(
            torch.tensor(centers[:n_clusters], dtype=torch.float32, device=device)
        )

    # Pretrain discriminators
    epsilon = 1e-3
    label_smoothing = 0.1
    if num_groups > 0 and disc_pretrain_epoch > 0:
        print("Pretraining discriminators...")
        for ep in range(disc_pretrain_epoch):
            for indices, xb, xrb, sfb, _zref_b in dataloader:  # _zref_b unused here
                xb = xb.to(device, non_blocking=True)
                batch_masks = [
                    torch.tensor([idx.item() in us for idx in indices],
                                 dtype=torch.bool, device=device)
                    for us in user_sets
                ]
                with torch.no_grad():
                    z, q, _, _, _ = main_lasso.generator(xb)
                disc_optimizer.zero_grad()
                att_preds = main_lasso.discriminator(z.detach(), q.detach())
                dloss = torch.tensor(0.0, device=device)
                for ap, bm in zip(att_preds, batch_masks):
                    hi = torch.ones(xb.size(0), 1, device=device) * (1 - label_smoothing)
                    lo = torch.ones(xb.size(0), 1, device=device) * label_smoothing
                    tgt = torch.where(bm.unsqueeze(1), hi, lo)
                    ap_c = torch.clamp(ap, epsilon, 1 - epsilon)
                    dloss += nn.BCELoss()(ap_c, tgt)
                dloss.backward()
                disc_optimizer.step()
            if (ep + 1) % 20 == 0:
                print(f"  Disc pretrain [{ep+1}/{disc_pretrain_epoch}], loss: {dloss.item():.4f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Main loop
    for epoch in range(gan_epoch):
        # --- Discriminator step ---
        main_lasso.discriminator.train()
        main_lasso.generator.eval()
        for indices, xb, xrb, sfb, _zref_b in dataloader:  # _zref_b unused in disc step
            xb = xb.to(device, non_blocking=True)
            batch_masks = [
                torch.tensor([idx.item() in us for idx in indices],
                             dtype=torch.bool, device=device)
                for us in user_sets
            ]
            with torch.no_grad():
                z, q, _, _, _ = main_lasso.generator(xb)
            disc_optimizer.zero_grad()
            att_preds = main_lasso.discriminator(z.detach(), q.detach())
            dloss = torch.tensor(0.0, device=device)
            for ap, bm in zip(att_preds, batch_masks):
                hi = torch.ones(xb.size(0), 1, device=device) * (1 - label_smoothing)
                lo = torch.ones(xb.size(0), 1, device=device) * label_smoothing
                tgt = torch.where(bm.unsqueeze(1), hi, lo)
                ap_c = torch.clamp(ap, epsilon, 1 - epsilon)
                dloss += nn.BCELoss()(ap_c, tgt)
            dloss.backward()
            disc_optimizer.step()

        # --- Generator step ---
        main_lasso.generator.train()
        main_lasso.discriminator.eval()
        total_g = 0.0
        total_ref = 0.0
        n_batches = 0

        for indices, xb, xrb, sfb, zref_b in dataloader:
            xb = xb.to(device, non_blocking=True)
            xrb = xrb.to(device, non_blocking=True)
            sfb = sfb.to(device, non_blocking=True)
            # zref_b: pre-computed reference latent [batch, z_dim] — no encoder call needed

            batch_masks = [
                torch.tensor([idx.item() in us for idx in indices],
                             dtype=torch.bool, device=device)
                for us in user_sets
            ]

            # Forward on embedding
            z, q, mean_t, disp_t, pi_t = main_lasso.generator(xb)

            # Reconstruction loss (ZINB against raw expression)
            recon_loss = main_lasso.generator.zinb_loss(
                x=xrb, mean=mean_t, disp=disp_t, pi=pi_t, scale_factor=sfb)

            # Clustering loss
            p = main_lasso.generator.target_distribution(q).detach()
            clust_loss = main_lasso.generator.cluster_loss(p, q)

            # Adversarial attention loss
            att_preds = main_lasso.discriminator(z, q)
            adv_loss = torch.tensor(0.0, device=device)
            for ap, bm in zip(att_preds, batch_masks):
                if torch.any(bm):
                    ap_c = torch.clamp(ap[bm], epsilon, 1 - epsilon)
                    adv_loss += nn.BCELoss()(ap_c, torch.ones_like(ap_c))

            # Feature separation loss
            inner_l, outer_l = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            if lambda_feature > 0 and num_groups > 0:
                inner_l, outer_l = main_lasso.feature_separation_loss(z, batch_masks)

            # ---- Reference alignment loss ----
            # z_ref was pre-computed for all cells before the training loop;
            # here we simply index the batch slice — zero encoder overhead.
            ref_loss = torch.tensor(0.0, device=device)
            if lambda_ref > 0:
                ref_loss = nn.MSELoss()(z, zref_b.to(device))
            total_ref += ref_loss.item()

            gen_loss = (lambda_recon * recon_loss
                        + lambda_cluster * clust_loss
                        + lambda_attention * adv_loss
                        + lambda_feature * (inner_l + outer_l)
                        + lambda_ref * ref_loss)

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            total_g += gen_loss.item()
            n_batches += 1

        gen_scheduler.step()
        disc_scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == gan_epoch - 1:
            avg_g = total_g / max(n_batches, 1)
            avg_ref = total_ref / max(n_batches, 1)
            print(f"Epoch [{epoch+1}/{gan_epoch}] Gen loss: {avg_g:.4f}  "
                  f"Ref alignment loss (unweighted): {avg_ref:.4f}")
            # Show cluster distribution for user groups
            with torch.no_grad():
                main_lasso.generator.eval()
                lat_full = main_lasso.generator.encodeBatch(X_tensor)
                q_full = main_lasso.generator.soft_assign(lat_full)
                y_tmp = torch.argmax(q_full, dim=1).cpu().numpy()
                for gi, ul in enumerate(user_selected_lists):
                    if ul:
                        lbs, cts = np.unique(y_tmp[ul], return_counts=True)
                        print(f"  Group {gi+1} distribution: {dict(zip(lbs, cts))}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 8. Collect results
    # ------------------------------------------------------------------
    with torch.no_grad():
        main_lasso.generator.eval()
        latent_full = main_lasso.generator.encodeBatch(X_tensor)
        q_full = main_lasso.generator.soft_assign(latent_full)
        y_pred = torch.argmax(q_full, dim=1).cpu().numpy()

    adata.obs['LassoARE_clusters'] = y_pred.astype(np.int32)
    adata.obsm['LassoARE_latent'] = latent_full.cpu().numpy()

    if do_pp:
        leiden_adata = adata.copy()
        sc.pp.neighbors(leiden_adata, use_rep='LassoARE_latent', n_neighbors=100)
        sc.tl.umap(leiden_adata, min_dist=0.5, spread=0.8)
        sc.tl.leiden(leiden_adata, key_added='leiden_LassoARE', resolution=leiden_r)
        adata.obsm['X_umap_LassoARE'] = leiden_adata.obsm['X_umap']
        adata.obs['leiden_LassoARE'] = leiden_adata.obs['leiden_LassoARE']

    print("Reconstruction with reference guidance completed!")
    return adata