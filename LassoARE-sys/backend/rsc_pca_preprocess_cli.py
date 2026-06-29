from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import cupy as cp
import numpy as np
import rapids_singlecell as rsc
import scipy.sparse
from cuml.decomposition import PCA

RAW_PCA_KEY = "X_pca_raw_LassoARE"
EMBEDDING_PCA_KEY = "X_pca_lassoare_embedding"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        return np.asarray(value.get())
    return np.asarray(value)


def _move_to_gpu(adata: ad.AnnData) -> ad.AnnData:
    mover = getattr(getattr(rsc, "get", None), "anndata_to_GPU", None)
    if mover is None:
        return adata
    moved = mover(adata)
    return moved if moved is not None else adata


def _move_to_cpu(adata: ad.AnnData) -> ad.AnnData:
    mover = getattr(getattr(rsc, "get", None), "anndata_to_CPU", None)
    if mover is None:
        return adata
    moved = mover(adata)
    return moved if moved is not None else adata


def _matrix_to_cupy_float32(matrix: Any) -> cp.ndarray:
    if scipy.sparse.issparse(matrix):
        matrix = matrix.toarray()
    return cp.asarray(matrix, dtype=cp.float32)


def _run_pca(adata: ad.AnnData, n_comps: int) -> np.ndarray:
    if n_comps < 1:
        raise ValueError("PCA requires at least one component.")

    # rsc.pp.pca is version-sensitive in the current environment; use cuML directly
    # so PCA still runs on GPU without going through the broken wrapper path.
    print("Using cuML PCA backend...", flush=True)
    x_gpu = _matrix_to_cupy_float32(adata.X)
    try:
        pca = PCA(n_components=n_comps, svd_solver="auto", output_type="cupy")
        x_pca = pca.fit_transform(x_gpu)
        return cp.asnumpy(x_pca).astype(np.float32, copy=False)
    finally:
        del x_gpu
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def _effective_n_comps(adata: ad.AnnData, requested: int) -> int:
    return max(1, min(int(requested), int(adata.n_obs), int(adata.n_vars)))


def _preprocess_generate(adata: ad.AnnData, spec: dict[str, Any]) -> dict[str, Any]:
    requested = int(spec.get("pca_dimension", 500))
    n_comps = _effective_n_comps(adata, requested)
    print(f"Running RAPIDS PCA preprocessing on expression matrix ({adata.n_vars} -> {n_comps})...", flush=True)
    adata.obsm["X_pca"] = _run_pca(adata, n_comps)

    raw_source = getattr(adata, "raw", None)
    if raw_source is not None:
        print(f"Running RAPIDS PCA preprocessing on raw counts ({raw_source.n_vars} -> {n_comps})...", flush=True)
        try:
            raw_adata = ad.AnnData(X=raw_source.X, obs=adata.obs.copy())
            raw_n_comps = _effective_n_comps(raw_adata, n_comps)
            adata.obsm[RAW_PCA_KEY] = _run_pca(raw_adata, raw_n_comps)
            if raw_n_comps < n_comps:
                print("Raw PCA had fewer available components; using expression PCA for LassoARE raw target.", flush=True)
                adata.obsm[RAW_PCA_KEY] = adata.obsm["X_pca"].copy()
        except Exception as exc:
            print(f"RAPIDS raw PCA failed; reusing expression PCA for raw target: {exc}", flush=True)
            adata.obsm[RAW_PCA_KEY] = adata.obsm["X_pca"].copy()
    else:
        print("No adata.raw found; reusing expression PCA for raw target.", flush=True)
        adata.obsm[RAW_PCA_KEY] = adata.obsm["X_pca"].copy()

    return {
        "mode": "generate",
        "main_pca_key": "X_pca",
        "raw_pca_key": RAW_PCA_KEY,
        "n_comps": int(adata.obsm["X_pca"].shape[1]),
    }


def _preprocess_reconstruct_embedding(adata: ad.AnnData, spec: dict[str, Any]) -> dict[str, Any]:
    embedding_key = spec.get("embedding_key")
    if not embedding_key or embedding_key not in adata.obsm:
        raise ValueError("Embedding reconstruction PCA requires a valid embedding_key.")

    emb = adata.obsm[embedding_key]
    if scipy.sparse.issparse(emb):
        emb = emb.todense()
    emb = np.asarray(emb, dtype=np.float32)
    emb_adata = ad.AnnData(X=emb, obs=adata.obs.copy())
    requested = int(spec.get("pca_dimension", 500))
    n_comps = _effective_n_comps(emb_adata, requested)
    print(f"Running RAPIDS PCA preprocessing on embedding '{embedding_key}' ({emb.shape[1]} -> {n_comps})...", flush=True)
    adata.obsm[EMBEDDING_PCA_KEY] = _run_pca(emb_adata, n_comps)
    return {
        "mode": "reconstruct_embedding",
        "embedding_key": embedding_key,
        "embedding_pca_key": EMBEDDING_PCA_KEY,
        "n_comps": int(adata.obsm[EMBEDDING_PCA_KEY].shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_h5ad = Path(args.input_h5ad)
    spec = _read_json(Path(args.spec))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_h5ad)
    mode = spec.get("lassoare_mode", "generate")
    if mode == "reconstruct_embedding":
        metadata = _preprocess_reconstruct_embedding(adata, spec)
    else:
        metadata = _preprocess_generate(adata, spec)

    adata.uns["LassoARE_rsc_pca"] = metadata
    result_path = output_dir / "pca_input.h5ad"
    print("Writing RAPIDS PCA preprocessed h5ad...", flush=True)
    adata.write_h5ad(result_path)
    _write_json(
        output_dir / "pca_preprocess.json",
        {
            "result_h5ad": str(result_path),
            "metadata": metadata,
        },
    )


if __name__ == "__main__":
    main()
