from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from backend.do_downsample import do_h5ad_downsample
from backend.do_lasso import do_lasso
from backend.LassoARE.reconstruction import reconstruction_with_lasso_are, reconstruction_with_ref


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def _sorted_unique(values: list[int]) -> list[int]:
    return sorted({int(item) for item in values})


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_color(adata: ad.AnnData, requested: str | None, fallback: str | None = None) -> str | None:
    if requested and requested in adata.obs.columns:
        return requested
    if fallback and fallback in adata.obs.columns:
        return fallback
    return adata.obs.columns[0] if len(adata.obs.columns) else None


def _int_list(value: Any, default: list[int]) -> list[int]:
    if isinstance(value, list) and value:
        return [int(item) for item in value]
    return default


def _ensure_lasso_graph(adata: ad.AnnData, spec: dict[str, Any]) -> None:
    if "connectivities" in adata.obsp:
        return

    embedding_key = spec.get("embedding_key")
    use_rep = None
    if embedding_key and embedding_key in adata.obsm:
        use_rep = embedding_key
    elif "X_pca" in adata.obsm:
        use_rep = "X_pca"
    elif adata.n_vars >= 2:
        n_comps = min(50, int(adata.n_vars), max(2, int(adata.n_obs) - 1))
        if n_comps >= 2:
            sc.pp.pca(adata, n_comps=n_comps)
            use_rep = "X_pca"

    sc.pp.neighbors(adata, use_rep=use_rep)


def _handle_lasso_view(adata: ad.AnnData, spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    selected_ids = _sorted_unique(spec.get("selected_ids", []))
    obs_col = spec.get("obs_col")
    if not obs_col or obs_col not in adata.obs.columns:
        raise ValueError("Lasso-View requires a valid obs_col.")
    _ensure_lasso_graph(adata, spec)

    expanded_ids = _sorted_unique(do_lasso(adata, selected_ids, obs_col=obs_col, vis=False, vis_key=obs_col, do_correct=bool(spec.get("do_correct", True))))

    seed_mask = np.zeros(adata.n_obs, dtype=bool)
    expanded_mask = np.zeros(adata.n_obs, dtype=bool)
    propagated_mask = np.zeros(adata.n_obs, dtype=bool)
    if selected_ids:
        seed_mask[selected_ids] = True
    if expanded_ids:
        expanded_mask[expanded_ids] = True
    propagated_only = sorted(set(expanded_ids) - set(selected_ids))
    if propagated_only:
        propagated_mask[propagated_only] = True

    status = np.full(adata.n_obs, "unselected", dtype=object)
    status[propagated_mask] = "propagated"
    status[seed_mask] = "seed"

    adata.obs["lasso_view_seed"] = pd.Categorical(seed_mask)
    adata.obs["lasso_view_selected"] = pd.Categorical(expanded_mask)
    adata.obs["lasso_view_propagated"] = pd.Categorical(propagated_mask)
    adata.obs["lasso_view_status"] = pd.Categorical(status, categories=["unselected", "propagated", "seed"])

    result_path = output_dir / "result.h5ad"
    adata.write_h5ad(result_path)
    return {
        "dataset_name": f"{spec['dataset_name']} (lasso view)",
        "analysis_type": "lasso_view",
        "result_h5ad": str(result_path),
        "preferred_embedding": spec.get("embedding_key"),
        "preferred_color_by": "lasso_view_status",
        "interactive_kind": None,
        "expanded_ids": expanded_ids,
        "seed_ids": selected_ids,
        "needs_postprocess": False,
    }


def _handle_downsample(adata: ad.AnnData, spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    embedding_key = spec.get("embedding_key")
    if not embedding_key:
        raise ValueError("Downsample requires embedding_key.")

    cluster_key = spec.get("cluster_key") or "leiden"
    sampled_adata, nearest_ids = do_h5ad_downsample(
        adata,
        sample_rate=float(spec.get("sample_rate", 0.1)),
        leiden_r=float(spec.get("leiden_resolution", 1.0)),
        uniform_rate=float(spec.get("uniform_rate", 0.5)),
        add_col="orig_idx",
        cluster_key=cluster_key,
        obsm_key=embedding_key,
    )
    sampled_adata.obs["orig_idx"] = sampled_adata.obs["orig_idx"].astype(int)

    result_path = output_dir / "result.h5ad"
    mapping_path = output_dir / "mapping.json"
    sampled_adata.write_h5ad(result_path)

    downsampled_orig_idx = sampled_adata.obs["orig_idx"].astype(int).tolist()
    mapping_payload = {
        "nearest_downsampled_local_id": [int(item) for item in np.asarray(nearest_ids).tolist()],
        "downsampled_orig_idx": downsampled_orig_idx,
        "nearest_downsampled_orig_idx": [int(downsampled_orig_idx[int(item)]) for item in np.asarray(nearest_ids).tolist()],
    }
    _write_json(mapping_path, mapping_payload)

    preferred_color = _default_color(sampled_adata, spec.get("color_by"), fallback=cluster_key)
    return {
        "dataset_name": f"{spec['dataset_name']} (downsample)",
        "analysis_type": "downsample",
        "result_h5ad": str(result_path),
        "mapping_path": str(mapping_path),
        "preferred_embedding": embedding_key,
        "preferred_color_by": preferred_color,
        "interactive_kind": "downsample",
        "needs_postprocess": False,
    }


def _common_reconstruction_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    enc_layers = _int_list(spec.get("enc_layers"), [256, 64])
    return {
        "user_selected_lists": spec.get("selected_groups", []),
        "n_clusters": spec.get("n_clusters"),
        "enc_pretrain_epoch": int(spec.get("enc_pretrain_epoch", 20)),
        "disc_pretrain_epoch": int(spec.get("disc_pretrain_epoch", 20)),
        "gan_epoch": int(spec.get("gan_epoch", 20)),
        "enc_layers": enc_layers,
        "dec_layers": list(reversed(enc_layers)),
        "disc_layers": _int_list(spec.get("disc_layers"), [256, 64]),
        "batch_size": int(spec.get("batch_size", 256)),
        "device": _device(),
        "lambda_attention": float(spec.get("lambda_attention", 0.1)),
        "leiden_r": float(spec.get("leiden_resolution", 1.0)),
        "z_dim": int(spec.get("z_dim", 32)),
        "is_pca": bool(spec.get("is_pca", True)),
        "pca_dimension": int(spec.get("pca_dimension", 500)),
        "do_pp": spec.get("execution_backend") == "scanpy",
    }


def _handle_lassoare(adata: ad.AnnData, spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    mode = spec.get("lassoare_mode", "generate")
    kwargs = _common_reconstruction_kwargs(spec)
    embedding_key = spec.get("embedding_key")
    if mode == "generate":
        result_adata = reconstruction_with_lasso_are(adata, using_emb=None, **kwargs)
        dataset_name = f"{spec['dataset_name']} (lassoare generate)"
    elif mode == "reconstruct_embedding":
        if not embedding_key:
            raise ValueError("Embedding reconstruction requires embedding_key.")
        result_adata = reconstruction_with_ref(
            adata,
            using_emb=embedding_key,
            ref_enc_layers=kwargs["enc_layers"],
            ref_pretrain_epoch=kwargs["enc_pretrain_epoch"],
            lambda_ref=float(spec.get("lambda_ref", 0.3)),
            **kwargs,
        )
        dataset_name = f"{spec['dataset_name']} (lassoare reconstruct {embedding_key})"
    else:
        raise ValueError(f"Unknown lassoare mode: {mode}")

    result_path = output_dir / "intermediate.h5ad"
    print("Writing LassoARE intermediate h5ad...", flush=True)
    result_adata.write_h5ad(result_path)
    return {
        "dataset_name": dataset_name,
        "analysis_type": "lassoare",
        "lassoare_mode": mode,
        "result_h5ad": str(result_path),
        "preferred_embedding": "X_umap_LassoARE",
        "preferred_color_by": "leiden_LassoARE",
        "interactive_kind": None,
        "needs_postprocess": not kwargs["do_pp"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_h5ad = Path(args.input_h5ad)
    spec_path = Path(args.spec)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = _read_json(spec_path)
    adata = ad.read_h5ad(input_h5ad)
    analysis_type = spec.get("analysis_type")

    if analysis_type == "lasso_view":
        result = _handle_lasso_view(adata, spec, output_dir)
    elif analysis_type == "downsample":
        result = _handle_downsample(adata, spec, output_dir)
    elif analysis_type in {"lassoare", "lassoare_x", "reconstruct_embedding"}:
        if analysis_type == "lassoare_x":
            spec["lassoare_mode"] = "generate"
        elif analysis_type == "reconstruct_embedding":
            spec["lassoare_mode"] = "reconstruct_embedding"
        result = _handle_lassoare(adata, spec, output_dir)
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

    _write_json(output_dir / "result.json", result)


if __name__ == "__main__":
    main()
