"""Small, CPU-only I/O helpers for the supplementary plotting notebooks.

Large source matrices are read in blocks; source H5ADs are never modified.
Plotting and statistical choices remain visible in the individual notebooks.
"""
from pathlib import Path
import ast

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from anndata.io import read_elem
except ImportError:  # anndata 0.10
    from anndata.experimental import read_elem


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing input: {path}\nSee README.md for the source data/workflow. "
            "Edit DATA_ROOT or DATA_FILES in the first code cell and rerun from that cell. "
            "Large H5AD files are not included in the GitHub repository."
        )
    return path


def read_metadata(path, obsm_keys=("X_umap",)):
    """Read annotations and requested coordinates, without expression/raw/PCA."""
    with h5py.File(require_file(path), "r") as f:
        obs = read_elem(f["obs"])
        result = ad.AnnData(X=sparse.csr_matrix((len(obs), 0)), obs=obs)
        for key in obsm_keys:
            if key not in f["obsm"]:
                raise KeyError(f"{path}: missing obsm[{key!r}]")
            result.obsm[key] = read_elem(f["obsm"][key])
        if "uns" in f:
            for key in f["uns"]:
                if key.endswith("_colors"):
                    result.uns[key] = read_elem(f["uns"][key])
    return result


def read_expression(path, genes, *, row_mask=None, use_raw=False, chunk_size=2048):
    """Read only requested genes; return a dense cells × genes array.

    CSR inputs are streamed by rows, CSC inputs by selected columns. Gene symbols
    can reside in var_names, gene_name or feature_name. Ambiguous symbols fail.
    """
    with h5py.File(require_file(path), "r") as f:
        source = f["raw"] if use_raw else f
        var = read_elem(source["var"])
        positions = []
        for gene in genes:
            hits = np.flatnonzero(var.index.astype(str) == gene)
            for key in ("gene_name", "feature_name"):
                if len(hits) == 0 and key in var:
                    hits = np.flatnonzero(var[key].astype(str).to_numpy() == gene)
            if len(hits) != 1:
                raise ValueError(f"{path}: gene {gene!r} has {len(hits)} matches")
            positions.append(int(hits[0]))
        obs = f["obs"]
        n_rows = len(obs[obs.attrs["_index"]])
        mask = np.ones(n_rows, dtype=bool) if row_mask is None else np.asarray(row_mask)
        if mask.dtype != bool or mask.shape != (n_rows,):
            raise ValueError("row_mask must be boolean and aligned to the source observations")
        result = np.empty((int(mask.sum()), len(genes)), dtype=np.float32)
        node = source["X"]
        encoding = node.attrs.get("encoding-type", "array")
        if encoding == "csc_matrix":
            for j, pos in enumerate(positions):
                lo, hi = node["indptr"][pos:pos + 2]
                column = np.zeros(n_rows, dtype=np.float32)
                np.add.at(column, node["indices"][lo:hi], node["data"][lo:hi])
                result[:, j] = column[mask]
            return result
        offset = 0
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            local = mask[start:stop]
            count = int(local.sum())
            if not count:
                continue
            if encoding == "csr_matrix":
                indptr = node["indptr"][start:stop + 1]
                lo, hi = int(indptr[0]), int(indptr[-1])
                block = sparse.csr_matrix(
                    (node["data"][lo:hi], node["indices"][lo:hi], indptr - lo),
                    shape=(stop - start, len(var)),
                )[:, positions].toarray()
            elif isinstance(node, h5py.Dataset):
                order = np.argsort(positions)
                sorted_positions = np.asarray(positions)[order]
                block = node[start:stop, sorted_positions][:, np.argsort(order)]
            else:
                raise ValueError(f"Unsupported X encoding: {encoding}")
            result[offset:offset + count] = block[local]
            offset += count
        return result


def selection_mask(path, n_obs, line=0):
    lines = [s.strip() for s in require_file(path).read_text().splitlines() if s.strip()]
    values = np.asarray(ast.literal_eval(lines[line]))
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"{path}: expected a list of integer row positions")
    if len(np.unique(values)) != len(values) or np.any((values < 0) | (values >= n_obs)):
        raise ValueError(f"{path}: duplicate or out-of-range row positions")
    mask = np.zeros(n_obs, dtype=bool)
    mask[values] = True
    return mask


def set_selection(adata, mask):
    adata.obs["selection"] = pd.Categorical(
        np.where(mask, "Selected", "Others"), categories=["Others", "Selected"]
    )


def save_figure(fig, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_embedding(adata, color, *, basis="X_umap", title="", size=1.0, ax=None):
    import scanpy as sc

    if color not in adata.obs:
        raise KeyError(f"Missing obs[{color!r}]")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    kwargs = {}
    if color == "selection":
        kwargs["palette"] = {"Others": "#1f77b4", "Selected": "#ff7f0e"}
    sc.pl.embedding(adata, basis=basis, color=color, size=size,
                    ax=ax, show=False, title=title, **kwargs)
    for collection in ax.collections:
        collection.set_rasterized(True)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_frame_on(False)
    return ax
