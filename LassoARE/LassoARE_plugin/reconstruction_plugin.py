import sys
import os
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
# sys.path.append('/home/zzj/lare/LassoARE')

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import scanpy as sc
import scanpy.external.pp as pp
import scipy.sparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from LassoARE.lasso_ARE import LassoARE, MultiDiscriminator
from LassoARE.scARE import buildNetwork
from LassoARE.utils import DispAct, MeanAct, ZINBLoss, check_matrix

from .plugins import MetricPluginConfig, coerce_plugins, ensure_metric_tensor, slice_requirements, summarize_representation


def _auto_detect_clusters(adata, n_clusters: Optional[int]) -> int:
    if n_clusters is not None:
        return n_clusters

    for col in ("codes", "annotation", "anno-NK", "leiden"):
        if col in adata.obs.columns:
            annotations = adata.obs[col]
            break
    else:
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)
        annotations = adata.obs["leiden"]

    detected = len(np.unique(annotations)) + 1
    print(f"Auto-detected {detected} clusters")
    return detected


def _normalize_user_lists(user_selected_lists, n_samples: int) -> List[List[int]]:
    if user_selected_lists is None:
        return []
    if not isinstance(user_selected_lists, list):
        user_selected_lists = [user_selected_lists]
    elif len(user_selected_lists) > 0 and not isinstance(user_selected_lists[0], (list, np.ndarray)):
        user_selected_lists = [user_selected_lists]

    processed = []
    for group in user_selected_lists:
        if group is None:
            continue
        valid = [int(i) for i in group if 0 <= int(i) < n_samples]
        if valid:
            processed.append(valid)
    return processed


def _prepare_representation_inputs(adata, using_emb: str, is_pca: bool, pca_dimension: int):
    print(f"Using embedding '{using_emb}' as main encoder input")
    emb_data = adata.obsm[using_emb]
    if scipy.sparse.issparse(emb_data):
        emb_data = emb_data.todense()
    emb_data = np.array(emb_data, dtype=np.float32)
    print(f"Original embedding shape: {emb_data.shape}, dtype: {emb_data.dtype}")

    if is_pca:
        pca_dim = min(pca_dimension, emb_data.shape[0], emb_data.shape[1])
        print(f"Applying PCA to embedding: {emb_data.shape[1]} -> {pca_dim}")
        emb_adata = sc.AnnData(X=emb_data, obs=adata.obs.copy())
        # print(f'Applying highly variable gene selection on embedding with shape {emb_adata.shape}...')
        # sc.pp.highly_variable_genes(emb_adata, n_top_genes=min(3000, emb_adata.shape[1]))
        # print('PCA on embedding with highly variable gene selection...')
        sc.tl.pca(emb_adata, n_comps=pca_dim, svd_solver="arpack") # , use_highly_variable=True
        if "batch" in emb_adata.obs.columns:
            pp.harmony_integrate(emb_adata, key="batch")
            emb_adata.obsm["X_pca"] = emb_adata.obsm["X_pca_harmony"]
        emb_data = emb_adata.obsm["X_pca"].astype(np.float32)

    has_raw = hasattr(adata, "raw") and adata.raw is not None

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

    size_factors = (
        adata.obs["size_factors"].values.astype(np.float32)
        if "size_factors" in adata.obs
        else np.ones(adata.shape[0], dtype=np.float32)
    )
    return emb_data, expr_X, raw_X, size_factors


def _sanitize_matrix_for_scanpy(X):
    if scipy.sparse.issparse(X):
        clean_X = X.tocsr(copy=True)
        clean_X.data = np.nan_to_num(clean_X.data, nan=0.0, posinf=0.0, neginf=0.0)
        return clean_X
    return np.nan_to_num(np.array(X, copy=True), nan=0.0, posinf=0.0, neginf=0.0)


def _matrix_value_summary(X) -> Dict[str, float]:
    if scipy.sparse.issparse(X):
        values = X.data
        if values.size == 0:
            return {"min": 0.0, "max": 0.0, "non_integer_fraction": 0.0}
        sample = values
        if sample.size > 100_000:
            step = max(1, sample.size // 100_000)
            sample = sample[::step]
        min_val = min(0.0, float(values.min())) if X.nnz < X.shape[0] * X.shape[1] else float(values.min())
        max_val = max(0.0, float(values.max()))
    else:
        values = np.asarray(X)
        if values.size == 0:
            return {"min": 0.0, "max": 0.0, "non_integer_fraction": 0.0}
        sample = values.reshape(-1)
        if sample.size > 100_000:
            step = max(1, sample.size // 100_000)
            sample = sample[::step]
        min_val = float(values.min())
        max_val = float(values.max())

    non_integer_fraction = float(np.mean(np.abs(sample - np.round(sample)) > 1e-6))
    return {"min": min_val, "max": max_val, "non_integer_fraction": non_integer_fraction}


def _pick_hvg_flavor(adata, X) -> Optional[str]:
    summary = _matrix_value_summary(X)
    has_log1p_metadata = "log1p" in adata.uns
    looks_nonnegative = summary["min"] >= 0.0
    looks_log1p_scale = (
        looks_nonnegative
        and summary["max"] <= 30.0
        and (has_log1p_metadata or summary["non_integer_fraction"] > 0.05)
    )

    if looks_log1p_scale:
        print(
            "Applying HVG selection with Seurat flavor on log1p-like expression data "
            f"(min={summary['min']:.3f}, max={summary['max']:.3f})."
        )
        return "seurat"

    if looks_nonnegative:
        print(
            "Applying HVG selection with Seurat v3 flavor on count-like expression data "
            f"(min={summary['min']:.3f}, max={summary['max']:.3f})."
        )
        return "seurat_v3"

    print(
        "Skipping HVG selection because the expression matrix contains negative values "
        f"(min={summary['min']:.3f}, max={summary['max']:.3f})."
    )
    return None


def _prepare_lasso_are_inputs(adata, using_emb: Optional[str], is_pca: bool, pca_dimension: int):
    """
    Prepare inputs for the no-reference LassoARE pipeline.

    This mirrors the behavior of reconstruction_with_lasso_are: if using_emb is
    provided, that embedding is treated as the model input; otherwise adata.X is
    used directly.
    """
    if using_emb is not None:
        print(f"Using embedding '{using_emb}' as input data")
        input_data = adata.obsm[using_emb]
        if scipy.sparse.issparse(input_data):
            input_data = input_data.todense()
        input_data = np.array(input_data, dtype=np.float32)
        original_dim = input_data.shape[1]
        process_adata = sc.AnnData(X=input_data, obs=adata.obs.copy())
        has_raw = False
    else:
        process_adata = adata.copy() if is_pca else adata
        original_dim = adata.shape[1]
        has_raw = hasattr(adata, "raw") and adata.raw is not None

    if is_pca:
        pca_dimension = min(pca_dimension, process_adata.shape[0], process_adata.shape[1])
        print(f"Preparing input data with shape {process_adata.shape} for PCA...")
        process_adata.X = _sanitize_matrix_for_scanpy(process_adata.X)

        use_highly_variable = False
        if using_emb is None:
            hvg_flavor = _pick_hvg_flavor(adata, process_adata.X)
            if hvg_flavor is not None:
                try:
                    sc.pp.highly_variable_genes(
                        process_adata,
                        n_top_genes=min(3000, process_adata.shape[1]),
                        flavor=hvg_flavor,
                    )
                    use_highly_variable = True
                except (ImportError, OverflowError, ValueError, FloatingPointError) as exc:
                    print(
                        "WARNING: Highly variable gene selection failed; falling back to PCA on all features. "
                        f"Reason: {exc}"
                    )
        else:
            print("Skipping HVG selection because the PCA input is an embedding, not an expression matrix.")

        print(f"Applying PCA: reducing from {original_dim} to {pca_dimension} dimensions...")
        sc.tl.pca(
            process_adata,
            n_comps=pca_dimension,
            svd_solver="arpack",
            use_highly_variable=use_highly_variable,
        )
        if "batch" in process_adata.obs.columns:
            pp.harmony_integrate(process_adata, key="batch")
            process_adata.obsm["X_pca"] = process_adata.obsm["X_pca_harmony"]
            print(
                "PCA on input data completed with Harmony integration. "
                f"Explained variance ratio: {process_adata.uns['pca']['variance_ratio'].sum():.4f}"
            )
        else:
            print(
                "PCA on input data completed (no batch correction). "
                f"Explained variance ratio: {process_adata.uns['pca']['variance_ratio'].sum():.4f}"
            )

        if using_emb is None and has_raw:
            raw_adata = sc.AnnData(X=_sanitize_matrix_for_scanpy(adata.raw.X), obs=adata.obs.copy())
            sc.tl.pca(raw_adata, n_comps=pca_dimension, svd_solver="arpack")
            if "batch" in raw_adata.obs.columns:
                pp.harmony_integrate(raw_adata, key="batch")
                raw_adata.obsm["X_pca"] = raw_adata.obsm["X_pca_harmony"]

            adata.uns["LassoARE_pca_raw"] = {
                "pca": raw_adata.varm["PCs"],
                "variance": raw_adata.uns["pca"]["variance"],
                "variance_ratio": raw_adata.uns["pca"]["variance_ratio"],
            }
            raw_X_pca = raw_adata.obsm["X_pca"]
            print(
                "PCA on raw counts completed. "
                f"Explained variance ratio: {raw_adata.uns['pca']['variance_ratio'].sum():.4f}"
            )
        else:
            if using_emb is None:
                print("Warning: adata.raw not found. Using normalized data PCA for raw_X.")
            raw_X_pca = process_adata.obsm["X_pca"].copy()

        if using_emb is None:
            adata.uns["LassoARE_pca"] = {
                "original_dim": original_dim,
                "pca_dim": pca_dimension,
                "explained_variance_ratio": process_adata.uns["pca"]["variance_ratio"].sum(),
                "is_pca_applied": True,
            }

    if is_pca:
        x = np.array(process_adata.obsm["X_pca"], dtype=np.float32)
        raw_X = np.array(raw_X_pca, dtype=np.float32)
    else:
        x = process_adata.X
        if scipy.sparse.issparse(x):
            x = x.todense()
        x = np.array(x, dtype=np.float32)

        if using_emb is None and has_raw:
            raw_X = adata.raw.X.todense() if scipy.sparse.issparse(adata.raw.X) else adata.raw.X
            raw_X = np.array(raw_X, dtype=np.float32)
        else:
            if using_emb is None:
                print("Warning: adata.raw not found. Using adata.X as raw counts.")
            raw_X = x.copy()

    size_factors = (
        adata.obs["size_factors"].values.astype(np.float32)
        if "size_factors" in adata.obs
        else np.ones(adata.shape[0], dtype=np.float32)
    )
    return x, raw_X, size_factors


def _build_main_lasso(
    emb_dim: int,
    expr_dim: int,
    z_dim: int,
    n_clusters: int,
    enc_layers: Sequence[int],
    dec_layers: Sequence[int],
    disc_layers: Sequence[int],
    num_user_groups: int,
    device: torch.device,
) -> LassoARE:
    main_lasso = LassoARE(
        input_dim=emb_dim,
        z_dim=z_dim,
        n_clusters=n_clusters,
        enc_layers=list(enc_layers),
        dec_layers=list(dec_layers),
        disc_layers=list(disc_layers),
        num_user_groups=num_user_groups,
        residual=True,
        device=device,
    )

    if expr_dim != emb_dim:
        output_dim = dec_layers[-1] if dec_layers else z_dim
        main_lasso.generator._dec_mean = nn.Sequential(nn.Linear(output_dim, expr_dim), MeanAct()).to(device)
        main_lasso.generator._dec_disp = nn.Sequential(nn.Linear(output_dim, expr_dim), DispAct()).to(device)
        main_lasso.generator._dec_pi = nn.Sequential(nn.Linear(output_dim, expr_dim), nn.Sigmoid()).to(device)
        main_lasso.generator.input_dim = expr_dim
    return main_lasso


def _pretrain_reference_encoder(
    expr_X: np.ndarray,
    raw_X: np.ndarray,
    size_factors: np.ndarray,
    ref_enc_layers: Sequence[int],
    dec_layers: Sequence[int],
    z_dim: int,
    expr_dim: int,
    ref_pretrain_epoch: int,
    batch_size: int,
    device: torch.device,
):
    print("Pretraining reference encoder on expression data...")
    ref_input_dim = expr_X.shape[1]
    ref_enc_full_layers = [ref_input_dim] + list(ref_enc_layers) + [z_dim]
    ref_encoder = buildNetwork(ref_enc_full_layers, type="encode", activation="relu", residual=True).to(device)
    print(f"ref_encoder architecture: {ref_enc_full_layers}")

    ref_X_tensor = torch.tensor(expr_X, dtype=torch.float32)
    ref_raw_tensor = torch.tensor(raw_X, dtype=torch.float32)
    sf_tensor_ref = torch.tensor(size_factors, dtype=torch.float32)

    ref_dataset = TensorDataset(ref_X_tensor, ref_raw_tensor, sf_tensor_ref)
    ref_loader = DataLoader(
        ref_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    dec_last_dim = dec_layers[-1] if dec_layers else z_dim
    ref_dec_proj = nn.Linear(z_dim, dec_last_dim).to(device)
    ref_dec_mean = nn.Sequential(nn.Linear(dec_last_dim, expr_dim), MeanAct()).to(device)
    ref_dec_disp = nn.Sequential(nn.Linear(dec_last_dim, expr_dim), DispAct()).to(device)
    ref_dec_pi = nn.Sequential(nn.Linear(dec_last_dim, expr_dim), nn.Sigmoid()).to(device)

    zinb = ZINBLoss().to(device)
    ref_opt_all = optim.Adam(
        list(ref_encoder.parameters())
        + list(ref_dec_proj.parameters())
        + list(ref_dec_mean.parameters())
        + list(ref_dec_disp.parameters())
        + list(ref_dec_pi.parameters()),
        lr=1e-3,
        amsgrad=True,
    )

    ref_encoder.train()
    for ep in range(ref_pretrain_epoch):
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
            print(f"Ref encoder pretrain [{ep + 1}/{ref_pretrain_epoch}], ZINB loss: {loss.item():.4f}")

    ref_encoder.eval()
    for param in ref_encoder.parameters():
        param.requires_grad_(False)

    print("Pre-computing reference latent representations z_ref (one-off)...")
    z_ref_parts = []
    with torch.no_grad():
        for start in range(0, len(expr_X), batch_size):
            end = min(start + batch_size, len(expr_X))
            xb = torch.tensor(expr_X[start:end], dtype=torch.float32, device=device)
            z_ref_parts.append(ref_encoder(xb).cpu())
    z_ref_all = torch.cat(z_ref_parts, dim=0)
    print(
        f"  z_ref shape: {tuple(z_ref_all.shape)} "
        f"(replaced {expr_X.shape[1]}-dim expression -> {z_ref_all.shape[1]}-dim latent in DataLoader)"
    )
    return ref_encoder, z_ref_all


def _init_cluster_centers(main_lasso: LassoARE, X_tensor: torch.Tensor, user_selected_lists: List[List[int]], n_clusters: int, device):
    with torch.no_grad():
        lat_init = main_lasso.generator.encodeBatch(X_tensor)
        num_groups = len(user_selected_lists)
        if num_groups > 0 and num_groups < n_clusters:
            group_centers = []
            for group in user_selected_lists:
                if group:
                    group_centers.append(lat_init[group].mean(0, keepdim=True).cpu().numpy())
            all_selected = np.zeros(X_tensor.shape[0], dtype=bool)
            for group in user_selected_lists:
                all_selected[group] = True
            other_feat = lat_init[~all_selected].cpu().numpy()
            num_other = n_clusters - len(group_centers)
            if other_feat.shape[0] >= num_other > 0:
                km = KMeans(n_clusters=num_other, n_init=20, random_state=42)
                other_centers = km.fit(other_feat).cluster_centers_
                centers = np.vstack([other_centers] + group_centers)
            else:
                centers = np.vstack(group_centers) if group_centers else lat_init.cpu().numpy()
                while centers.shape[0] < n_clusters:
                    ridx = np.random.choice(lat_init.shape[0])
                    centers = np.vstack([centers, lat_init[ridx].cpu().numpy().reshape(1, -1)])
        else:
            km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
            centers = km.fit(lat_init.cpu().numpy()).cluster_centers_
        main_lasso.generator.mu.data.copy_(torch.tensor(centers[:n_clusters], dtype=torch.float32, device=device))


def _collect_targets(generator, X_tensor: torch.Tensor, batch_size: int, targets: Sequence[str], track_grad: bool) -> Dict[str, torch.Tensor]:
    needed = set(targets)
    if "custom" in needed:
        needed.update({"latent", "cluster_probs", "reconstruction"})
    collected: Dict[str, List[torch.Tensor]] = {name: [] for name in needed}

    num = X_tensor.shape[0]
    num_batch = int(math.ceil(float(num) / batch_size))
    context = torch.enable_grad if track_grad else torch.no_grad

    with context():
        for batch_idx in range(num_batch):
            xbatch = X_tensor[batch_idx * batch_size : min((batch_idx + 1) * batch_size, num)]
            if xbatch.device != generator.device:
                xbatch = xbatch.to(generator.device, non_blocking=True)
            z, q, mean_t, _, _ = generator(xbatch)
            if "latent" in collected:
                collected["latent"].append(z)
            if "cluster_probs" in collected:
                collected["cluster_probs"].append(q)
            if "reconstruction" in collected:
                collected["reconstruction"].append(mean_t)

    return {name: torch.cat(chunks, dim=0) for name, chunks in collected.items()}


def _plugin_representation(plugin: MetricPluginConfig, representations: Dict[str, torch.Tensor]):
    if plugin.apply_on == "custom":
        return {
            "latent": representations["latent"],
            "cluster_probs": representations["cluster_probs"],
            "reconstruction": representations["reconstruction"],
        }
    return representations[plugin.apply_on]


def _batch_representation(plugin: MetricPluginConfig, z, q, mean_t):
    if plugin.apply_on == "latent":
        return z
    if plugin.apply_on == "cluster_probs":
        return q
    if plugin.apply_on == "reconstruction":
        return mean_t
    return {"latent": z, "cluster_probs": q, "reconstruction": mean_t}


def _compute_batch_plugin_loss(
    plugins: Sequence[MetricPluginConfig],
    epoch: int,
    indices: torch.Tensor,
    n_samples: int,
    z: torch.Tensor,
    q: torch.Tensor,
    mean_t: torch.Tensor,
) -> torch.Tensor:
    total_loss = torch.tensor(0.0, device=z.device)
    batch_indices = indices.cpu().numpy().tolist()

    for plugin in plugins:
        if plugin.full_dataset_only or plugin.mode not in {"differentiable", "hybrid"}:
            continue
        weight = plugin.current_weight(epoch)
        if weight <= 0:
            continue

        rep = _batch_representation(plugin, z, q, mean_t)
        req = slice_requirements(plugin.requirements, batch_indices, n_samples)
        metric = ensure_metric_tensor(plugin.metric_fn(rep, req), device=z.device)
        plugin.update_metric_stats(metric)
        total_loss = total_loss + weight * plugin.metric_to_loss(metric)

    return total_loss


def _refresh_surrogate_plugins(
    plugins: Sequence[MetricPluginConfig],
    epoch: int,
    full_targets: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for plugin in plugins:
        if plugin.mode not in {"surrogate", "hybrid"} or not plugin.should_refresh(epoch):
            continue

        rep = _plugin_representation(plugin, full_targets)
        metric = ensure_metric_tensor(plugin.metric_fn(rep, plugin.requirements), device=device)
        plugin.update_metric_stats(metric)
        target = plugin.normalize_metric(metric).detach()
        plugin.ensure_scorer(device=device, output_dim=target.numel())
        if isinstance(rep, dict):
            rep_for_summary = rep["cluster_probs"]
        else:
            rep_for_summary = rep
        plugin.add_surrogate_example(rep_for_summary, target)
        loss_value = plugin.train_surrogate(device)
        if loss_value is not None:
            stats[f"{plugin.name}_surrogate_fit"] = loss_value
            print(f"  Plugin '{plugin.name}' surrogate fit loss: {loss_value:.4f}")
    return stats


def _full_dataset_plugin_step(
    plugins: Sequence[MetricPluginConfig],
    epoch: int,
    generator,
    gen_optimizer,
    X_tensor: torch.Tensor,
    batch_size: int,
) -> Dict[str, float]:
    active_plugins = [plugin for plugin in plugins if plugin.current_weight(epoch) > 0]
    if not active_plugins:
        return {}

    surrogate_refresh_plugins = [plugin for plugin in active_plugins if plugin.mode in {"surrogate", "hybrid"} and plugin.should_refresh(epoch)]
    if surrogate_refresh_plugins:
        refresh_targets = sorted({plugin.apply_on for plugin in surrogate_refresh_plugins} | {"cluster_probs"})
        full_targets = _collect_targets(generator, X_tensor, batch_size, refresh_targets, track_grad=False)
        _refresh_surrogate_plugins(surrogate_refresh_plugins, epoch, full_targets, generator.device)

    full_plugins = [
        plugin
        for plugin in active_plugins
        if plugin.full_dataset_only or plugin.mode in {"surrogate", "hybrid"}
    ]
    if not full_plugins:
        return {}

    needed_targets = sorted({plugin.apply_on for plugin in full_plugins} | {"cluster_probs"})
    targets = _collect_targets(generator, X_tensor, batch_size, needed_targets, track_grad=True)

    plugin_loss = torch.tensor(0.0, device=generator.device)
    plugin_stats: Dict[str, float] = {}

    for plugin in full_plugins:
        rep = _plugin_representation(plugin, targets)
        weight = plugin.current_weight(epoch)
        if weight <= 0:
            continue

        if plugin.mode in {"differentiable", "hybrid"}:
            metric = ensure_metric_tensor(plugin.metric_fn(rep, plugin.requirements), device=generator.device)
            plugin.update_metric_stats(metric.detach())
            direct_loss = plugin.metric_to_loss(metric)
            plugin_loss = plugin_loss + weight * direct_loss
            plugin_stats[f"{plugin.name}_direct"] = float(direct_loss.detach().item())

        if plugin.mode in {"surrogate", "hybrid"} and plugin.has_scorer():
            if isinstance(rep, dict):
                rep_for_summary = rep["cluster_probs"]
            else:
                rep_for_summary = rep
            summary = summarize_representation(rep_for_summary, plugin.surrogate_summary_dim).to(generator.device)
            plugin.scorer.eval()
            predicted_metric = plugin.scorer(summary).reshape(-1)
            surrogate_loss = plugin.metric_to_loss(predicted_metric, normalized=True)
            plugin_loss = plugin_loss + weight * surrogate_loss
            plugin_stats[f"{plugin.name}_surrogate"] = float(surrogate_loss.detach().item())

    if plugin_loss.requires_grad and float(plugin_loss.detach().abs().item()) > 0:
        gen_optimizer.zero_grad()
        plugin_loss.backward()
        gen_optimizer.step()
        plugin_stats["plugin_total"] = float(plugin_loss.detach().item())

    return plugin_stats


def reconstruction_with_plugins(
    adata,
    user_selected_lists,
    using_emb,
    plugins: Optional[Sequence[Any]] = None,
    n_clusters=None,
    enc_pretrain_epoch=300,
    disc_pretrain_epoch=100,
    gan_epoch=50,
    ref_pretrain_epoch=200,
    enc_layers=(256, 64),
    dec_layers=(64, 256),
    disc_layers=(128, 64),
    ref_enc_layers=(256, 64),
    batch_size=256,
    device=torch.device("cpu"),
    lambda_attention=1.0,
    lambda_feature=1.0,
    lambda_consistency=1.0,
    lambda_cluster=1.0,
    lambda_recon=1.0,
    lambda_ref=0.3,
    leiden_r=0.2,
    z_dim=32,
    is_pca=False,
    pca_dimension=500,
    do_pp=False,
):
    """
    Plugin-enabled reconstruction based on the reference-guided LassoARE pipeline.
    """
    del lambda_consistency  # Kept for API compatibility with existing entrypoints.

    device = torch.device(device)
    n_clusters = _auto_detect_clusters(adata, n_clusters)
    emb_data, expr_X, raw_X, size_factors = _prepare_representation_inputs(adata, using_emb, is_pca, pca_dimension)

    user_selected_lists = _normalize_user_lists(user_selected_lists, adata.shape[0])
    num_user_groups = len(user_selected_lists)
    print(f"Number of user-selected groups: {num_user_groups}")

    selected_mask = np.zeros(adata.shape[0], dtype=bool)
    for group in user_selected_lists:
        selected_mask[group] = True
    adata.obs["LassoARE_selected"] = selected_mask

    plugin_states = coerce_plugins(plugins)
    if plugin_states:
        print(f"Using {len(plugin_states)} plugin(s): {[plugin.name for plugin in plugin_states]}")

    expr_dim = expr_X.shape[1]
    emb_dim = emb_data.shape[1]
    main_lasso = _build_main_lasso(
        emb_dim=emb_dim,
        expr_dim=expr_dim,
        z_dim=z_dim,
        n_clusters=n_clusters,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        disc_layers=disc_layers,
        num_user_groups=num_user_groups,
        device=device,
    )

    _, z_ref_all = _pretrain_reference_encoder(
        expr_X=expr_X,
        raw_X=raw_X,
        size_factors=size_factors,
        ref_enc_layers=ref_enc_layers,
        dec_layers=dec_layers,
        z_dim=z_dim,
        expr_dim=expr_dim,
        ref_pretrain_epoch=ref_pretrain_epoch,
        batch_size=batch_size,
        device=device,
    )

    print("Pretraining main autoencoder (embedding -> expression)...")
    main_lasso.pretrain_generator(
        X=emb_data,
        X_raw=raw_X,
        size_factor=size_factors,
        batch_size=batch_size,
        epochs=enc_pretrain_epoch,
    )

    print("Training adversarial model with expression reference guidance and plugins...")
    X_check = check_matrix(emb_data)
    X_raw_check = check_matrix(raw_X)
    X_tensor = torch.tensor(X_check, dtype=torch.float32)
    X_raw_tensor = torch.tensor(X_raw_check, dtype=torch.float32)
    sf_tensor = torch.tensor(size_factors, dtype=torch.float32)

    dataset = TensorDataset(
        torch.arange(len(X_tensor)),
        X_tensor,
        X_raw_tensor,
        sf_tensor,
        z_ref_all,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    if num_user_groups != main_lasso.num_user_groups:
        main_lasso.num_user_groups = num_user_groups
        main_lasso.discriminator = MultiDiscriminator(
            latent_dim=z_dim,
            n_clusters=n_clusters,
            num_groups=num_user_groups,
            hidden_dims=list(disc_layers),
        ).to(device)

    user_sets = [set(group) for group in user_selected_lists]
    _init_cluster_centers(main_lasso, X_tensor, user_selected_lists, n_clusters, device)

    gen_optimizer = optim.Adam(main_lasso.generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(main_lasso.discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=20, gamma=0.5)
    disc_scheduler = optim.lr_scheduler.StepLR(disc_optimizer, step_size=20, gamma=0.5)

    epsilon = 1e-3
    label_smoothing = 0.1

    if num_user_groups > 0 and disc_pretrain_epoch > 0:
        print("Pretraining discriminators...")
        for ep in range(disc_pretrain_epoch):
            for indices, xb, _, _, _ in dataloader:
                xb = xb.to(device, non_blocking=True)
                batch_masks = [
                    torch.tensor([idx.item() in user_set for idx in indices], dtype=torch.bool, device=device)
                    for user_set in user_sets
                ]
                with torch.no_grad():
                    z, q, _, _, _ = main_lasso.generator(xb)
                disc_optimizer.zero_grad()
                att_preds = main_lasso.discriminator(z.detach(), q.detach())
                dloss = torch.tensor(0.0, device=device)
                for ap, bm in zip(att_preds, batch_masks):
                    hi = torch.ones(xb.size(0), 1, device=device) * (1 - label_smoothing)
                    lo = torch.ones(xb.size(0), 1, device=device) * label_smoothing
                    target = torch.where(bm.unsqueeze(1), hi, lo)
                    dloss = dloss + nn.BCELoss()(torch.clamp(ap, epsilon, 1 - epsilon), target)
                dloss.backward()
                disc_optimizer.step()
            if (ep + 1) % 20 == 0 or ep == disc_pretrain_epoch - 1:
                print(f"  Disc pretrain [{ep + 1}/{disc_pretrain_epoch}], loss: {dloss.item():.4f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    for epoch in range(gan_epoch):
        main_lasso.discriminator.train()
        main_lasso.generator.eval()
        for indices, xb, _, _, _ in dataloader:
            xb = xb.to(device, non_blocking=True)
            batch_masks = [
                torch.tensor([idx.item() in user_set for idx in indices], dtype=torch.bool, device=device)
                for user_set in user_sets
            ]
            with torch.no_grad():
                z, q, _, _, _ = main_lasso.generator(xb)
            disc_optimizer.zero_grad()
            att_preds = main_lasso.discriminator(z.detach(), q.detach())
            dloss = torch.tensor(0.0, device=device)
            for ap, bm in zip(att_preds, batch_masks):
                hi = torch.ones(xb.size(0), 1, device=device) * (1 - label_smoothing)
                lo = torch.ones(xb.size(0), 1, device=device) * label_smoothing
                target = torch.where(bm.unsqueeze(1), hi, lo)
                dloss = dloss + nn.BCELoss()(torch.clamp(ap, epsilon, 1 - epsilon), target)
            dloss.backward()
            disc_optimizer.step()

        main_lasso.generator.train()
        main_lasso.discriminator.eval()
        total_g = 0.0
        total_ref = 0.0
        total_plugin_batch = 0.0
        n_batches = 0

        for indices, xb, xrb, sfb, zref_b in dataloader:
            xb = xb.to(device, non_blocking=True)
            xrb = xrb.to(device, non_blocking=True)
            sfb = sfb.to(device, non_blocking=True)

            batch_masks = [
                torch.tensor([idx.item() in user_set for idx in indices], dtype=torch.bool, device=device)
                for user_set in user_sets
            ]

            z, q, mean_t, disp_t, pi_t = main_lasso.generator(xb)

            recon_loss = main_lasso.generator.zinb_loss(x=xrb, mean=mean_t, disp=disp_t, pi=pi_t, scale_factor=sfb)
            p = main_lasso.generator.target_distribution(q).detach()
            clust_loss = main_lasso.generator.cluster_loss(p, q)

            att_preds = main_lasso.discriminator(z, q)
            adv_loss = torch.tensor(0.0, device=device)
            for ap, bm in zip(att_preds, batch_masks):
                if torch.any(bm):
                    adv_loss = adv_loss + nn.BCELoss()(
                        torch.clamp(ap[bm], epsilon, 1 - epsilon),
                        torch.ones_like(ap[bm]),
                    )

            inner_l = torch.tensor(0.0, device=device)
            outer_l = torch.tensor(0.0, device=device)
            if lambda_feature > 0 and num_user_groups > 0:
                inner_l, outer_l = main_lasso.feature_separation_loss(z, batch_masks)

            ref_loss = torch.tensor(0.0, device=device)
            if lambda_ref > 0:
                ref_loss = nn.MSELoss()(z, zref_b.to(device))
            total_ref += ref_loss.item()

            plugin_batch_loss = _compute_batch_plugin_loss(
                plugin_states,
                epoch,
                indices,
                X_tensor.shape[0],
                z,
                q,
                mean_t,
            )
            total_plugin_batch += float(plugin_batch_loss.detach().item()) if plugin_batch_loss.requires_grad else 0.0

            gen_loss = (
                lambda_recon * recon_loss
                + lambda_cluster * clust_loss
                + lambda_attention * adv_loss
                + lambda_feature * (inner_l + outer_l)
                + lambda_ref * ref_loss
                + plugin_batch_loss
            )

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            total_g += gen_loss.item()
            n_batches += 1

        plugin_epoch_stats = _full_dataset_plugin_step(
            plugin_states,
            epoch,
            main_lasso.generator,
            gen_optimizer,
            X_tensor,
            batch_size,
        )

        gen_scheduler.step()
        disc_scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == gan_epoch - 1:
            avg_g = total_g / max(n_batches, 1)
            avg_ref = total_ref / max(n_batches, 1)
            avg_plugin_batch = total_plugin_batch / max(n_batches, 1)
            print(
                f"Epoch [{epoch + 1}/{gan_epoch}] Gen loss: {avg_g:.4f}  "
                f"Ref alignment loss (unweighted): {avg_ref:.4f}  "
                f"Batch plugin loss: {avg_plugin_batch:.4f}"
            )
            if plugin_epoch_stats:
                print(f"  Full-dataset plugin stats: {plugin_epoch_stats}")
            with torch.no_grad():
                main_lasso.generator.eval()
                lat_full = main_lasso.generator.encodeBatch(X_tensor)
                q_full = main_lasso.generator.soft_assign(lat_full)
                y_tmp = torch.argmax(q_full, dim=1).cpu().numpy()
                for gi, group in enumerate(user_selected_lists):
                    if group:
                        labels, counts = np.unique(y_tmp[group], return_counts=True)
                        print(f"  Group {gi + 1} distribution: {dict(zip(labels, counts))}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        main_lasso.generator.eval()
        latent_full = main_lasso.generator.encodeBatch(X_tensor)
        q_full = main_lasso.generator.soft_assign(latent_full)
        y_pred = torch.argmax(q_full, dim=1).cpu().numpy()

    adata.obs["LassoARE_clusters"] = y_pred.astype(np.int32)
    adata.obsm["LassoARE_latent"] = latent_full.cpu().numpy()

    if do_pp:
        leiden_adata = adata.copy()
        sc.pp.neighbors(leiden_adata, use_rep="LassoARE_latent", n_neighbors=100)
        sc.tl.umap(leiden_adata, min_dist=0.5, spread=0.8)
        sc.tl.leiden(leiden_adata, key_added="leiden_LassoARE", resolution=leiden_r)
        adata.obsm["X_umap_LassoARE"] = leiden_adata.obsm["X_umap"]
        adata.obs["leiden_LassoARE"] = leiden_adata.obs["leiden_LassoARE"]

    print("Reconstruction with plugins completed!")
    return adata


def reconstruction_with_lasso_are_plugins(
    adata,
    user_selected_lists,
    plugins: Optional[Sequence[Any]] = None,
    using_emb=None,
    n_clusters=None,
    enc_pretrain_epoch=300,
    disc_pretrain_epoch=100,
    gan_epoch=50,
    enc_layers=(256, 64),
    dec_layers=(64, 256),
    disc_layers=(128, 64),
    batch_size=256,
    device=torch.device("cpu"),
    lambda_attention=1.0,
    lambda_feature=1.0,
    lambda_consistency=1.0,
    lambda_cluster=1.0,
    lambda_recon=1.0,
    leiden_r=0.2,
    z_dim=32,
    is_pca=False,
    pca_dimension=500,
    do_pp=False,
):
    """
    Plugin-enabled reconstruction based on the original reconstruction_with_lasso_are pipeline.

    Unlike reconstruction_with_plugins, this variant does not require expression
    reference guidance and can operate directly on adata.X when using_emb is None.
    """
    del lambda_consistency  # Kept for API compatibility with existing entrypoints.

    device = torch.device(device)
    n_clusters = _auto_detect_clusters(adata, n_clusters)
    x, raw_X, size_factors = _prepare_lasso_are_inputs(adata, using_emb, is_pca, pca_dimension)

    user_selected_lists = _normalize_user_lists(user_selected_lists, adata.shape[0])
    num_user_groups = len(user_selected_lists)
    print(f"Number of user-selected groups: {num_user_groups}")

    selected_mask = np.zeros(adata.shape[0], dtype=bool)
    for group in user_selected_lists:
        selected_mask[group] = True
    adata.obs["LassoARE_selected"] = selected_mask

    plugin_states = coerce_plugins(plugins)
    if plugin_states:
        print(f"Using {len(plugin_states)} plugin(s): {[plugin.name for plugin in plugin_states]}")

    input_dim = x.shape[1]
    lasso_are = LassoARE(
        input_dim=input_dim,
        z_dim=z_dim,
        n_clusters=n_clusters,
        enc_layers=list(enc_layers),
        dec_layers=list(dec_layers),
        disc_layers=list(disc_layers),
        num_user_groups=num_user_groups,
        residual=True,
        device=device,
    )

    print("Pretraining autoencoder...")
    lasso_are.pretrain_generator(
        X=x,
        X_raw=raw_X,
        size_factor=size_factors,
        batch_size=batch_size,
        epochs=enc_pretrain_epoch,
    )

    print("Training adversarial clustering model with plugins...")
    X_check = check_matrix(x)
    X_raw_check = check_matrix(raw_X)
    X_tensor = torch.tensor(X_check, dtype=torch.float32)
    X_raw_tensor = torch.tensor(X_raw_check, dtype=torch.float32)
    sf_tensor = torch.tensor(size_factors, dtype=torch.float32)

    dataset = TensorDataset(
        torch.arange(len(X_tensor)),
        X_tensor,
        X_raw_tensor,
        sf_tensor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )

    if num_user_groups != lasso_are.num_user_groups:
        lasso_are.num_user_groups = num_user_groups
        lasso_are.discriminator = MultiDiscriminator(
            latent_dim=z_dim,
            n_clusters=n_clusters,
            num_groups=num_user_groups,
            hidden_dims=list(disc_layers),
        ).to(device)

    user_sets = [set(group) for group in user_selected_lists]
    _init_cluster_centers(lasso_are, X_tensor, user_selected_lists, n_clusters, device)

    gen_optimizer = optim.Adam(lasso_are.generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(lasso_are.discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=20, gamma=0.5)
    disc_scheduler = optim.lr_scheduler.StepLR(disc_optimizer, step_size=20, gamma=0.5)

    epsilon = 1e-3
    label_smoothing = 0.1

    if num_user_groups > 0 and disc_pretrain_epoch > 0:
        print("Pretraining discriminators...")
        for ep in range(disc_pretrain_epoch):
            for indices, xb, _, _ in dataloader:
                xb = xb.to(device, non_blocking=True)
                batch_masks = [
                    torch.tensor([idx.item() in user_set for idx in indices], dtype=torch.bool, device=device)
                    for user_set in user_sets
                ]
                with torch.no_grad():
                    z, q, _, _, _ = lasso_are.generator(xb)
                disc_optimizer.zero_grad()
                att_preds = lasso_are.discriminator(z.detach(), q.detach())
                dloss = torch.tensor(0.0, device=device)
                for ap, bm in zip(att_preds, batch_masks):
                    hi = torch.ones(xb.size(0), 1, device=device) * (1 - label_smoothing)
                    lo = torch.ones(xb.size(0), 1, device=device) * label_smoothing
                    target = torch.where(bm.unsqueeze(1), hi, lo)
                    dloss = dloss + nn.BCELoss()(torch.clamp(ap, epsilon, 1 - epsilon), target)
                dloss.backward()
                disc_optimizer.step()
            if (ep + 1) % 20 == 0 or ep == disc_pretrain_epoch - 1:
                print(f"  Disc pretrain [{ep + 1}/{disc_pretrain_epoch}], loss: {dloss.item():.4f}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    for epoch in range(gan_epoch):
        lasso_are.discriminator.train()
        lasso_are.generator.eval()
        for indices, xb, _, _ in dataloader:
            xb = xb.to(device, non_blocking=True)
            batch_masks = [
                torch.tensor([idx.item() in user_set for idx in indices], dtype=torch.bool, device=device)
                for user_set in user_sets
            ]
            with torch.no_grad():
                z, q, _, _, _ = lasso_are.generator(xb)
            disc_optimizer.zero_grad()
            att_preds = lasso_are.discriminator(z.detach(), q.detach())
            dloss = torch.tensor(0.0, device=device)
            for ap, bm in zip(att_preds, batch_masks):
                hi = torch.ones(xb.size(0), 1, device=device) * (1 - label_smoothing)
                lo = torch.ones(xb.size(0), 1, device=device) * label_smoothing
                target = torch.where(bm.unsqueeze(1), hi, lo)
                dloss = dloss + nn.BCELoss()(torch.clamp(ap, epsilon, 1 - epsilon), target)
            dloss.backward()
            disc_optimizer.step()

        lasso_are.generator.train()
        lasso_are.discriminator.eval()
        total_g = 0.0
        total_plugin_batch = 0.0
        total_recon = 0.0
        total_cluster = 0.0
        n_batches = 0

        for indices, xb, xrb, sfb in dataloader:
            xb = xb.to(device, non_blocking=True)
            xrb = xrb.to(device, non_blocking=True)
            sfb = sfb.to(device, non_blocking=True)

            batch_masks = [
                torch.tensor([idx.item() in user_set for idx in indices], dtype=torch.bool, device=device)
                for user_set in user_sets
            ]

            z, q, mean_t, disp_t, pi_t = lasso_are.generator(xb)

            recon_loss = lasso_are.generator.zinb_loss(x=xrb, mean=mean_t, disp=disp_t, pi=pi_t, scale_factor=sfb)
            p = lasso_are.generator.target_distribution(q).detach()
            clust_loss = lasso_are.generator.cluster_loss(p, q)
            total_recon += recon_loss.item()
            total_cluster += clust_loss.item()

            att_preds = lasso_are.discriminator(z, q)
            adv_loss = torch.tensor(0.0, device=device)
            for ap, bm in zip(att_preds, batch_masks):
                if torch.any(bm):
                    adv_loss = adv_loss + nn.BCELoss()(
                        torch.clamp(ap[bm], epsilon, 1 - epsilon),
                        torch.ones_like(ap[bm]),
                    )

            inner_l = torch.tensor(0.0, device=device)
            outer_l = torch.tensor(0.0, device=device)
            if lambda_feature > 0 and num_user_groups > 0:
                inner_l, outer_l = lasso_are.feature_separation_loss(z, batch_masks)

            plugin_batch_loss = _compute_batch_plugin_loss(
                plugin_states,
                epoch,
                indices,
                X_tensor.shape[0],
                z,
                q,
                mean_t,
            )
            total_plugin_batch += float(plugin_batch_loss.detach().item()) if plugin_batch_loss.requires_grad else 0.0

            gen_loss = (
                lambda_recon * recon_loss
                + lambda_cluster * clust_loss
                + lambda_attention * adv_loss
                + lambda_feature * (inner_l + outer_l)
                + plugin_batch_loss
            )

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            total_g += gen_loss.item()
            n_batches += 1

        plugin_epoch_stats = _full_dataset_plugin_step(
            plugin_states,
            epoch,
            lasso_are.generator,
            gen_optimizer,
            X_tensor,
            batch_size,
        )

        gen_scheduler.step()
        disc_scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == gan_epoch - 1:
            avg_g = total_g / max(n_batches, 1)
            avg_recon = total_recon / max(n_batches, 1)
            avg_cluster = total_cluster / max(n_batches, 1)
            avg_plugin_batch = total_plugin_batch / max(n_batches, 1)
            print(
                f"Epoch [{epoch + 1}/{gan_epoch}] Gen loss: {avg_g:.4f}  "
                f"Recon: {avg_recon:.4f}  Cluster: {avg_cluster:.4f}  "
                f"Batch plugin loss: {avg_plugin_batch:.4f}"
            )
            if plugin_epoch_stats:
                print(f"  Full-dataset plugin stats: {plugin_epoch_stats}")
            with torch.no_grad():
                lasso_are.generator.eval()
                lat_full = lasso_are.generator.encodeBatch(X_tensor)
                q_full = lasso_are.generator.soft_assign(lat_full)
                y_tmp = torch.argmax(q_full, dim=1).cpu().numpy()
                for gi, group in enumerate(user_selected_lists):
                    if group:
                        labels, counts = np.unique(y_tmp[group], return_counts=True)
                        print(f"  Group {gi + 1} distribution: {dict(zip(labels, counts))}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        lasso_are.generator.eval()
        latent_full = lasso_are.generator.encodeBatch(X_tensor)
        q_full = lasso_are.generator.soft_assign(latent_full)
        y_pred = torch.argmax(q_full, dim=1).cpu().numpy()

    adata.obs["LassoARE_clusters"] = y_pred.astype(np.int32)
    adata.obsm["LassoARE_latent"] = latent_full.cpu().numpy()

    if do_pp:
        leiden_adata = adata.copy()
        sc.pp.neighbors(leiden_adata, use_rep="LassoARE_latent", n_neighbors=100)
        sc.tl.umap(leiden_adata, min_dist=0.5, spread=0.8)
        sc.tl.leiden(leiden_adata, key_added="leiden_LassoARE", resolution=leiden_r)
        adata.obsm["X_umap_LassoARE"] = leiden_adata.obsm["X_umap"]
        adata.obs["leiden_LassoARE"] = leiden_adata.obs["leiden_LassoARE"]

    print("Reconstruction with LassoARE plugins completed!")
    return adata
