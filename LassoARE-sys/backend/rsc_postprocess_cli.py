from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import rapids_singlecell as rsc


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


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
    if "LassoARE_latent" not in adata.obsm:
        raise ValueError("LassoARE_latent was not found in the intermediate result.")

    original_umap = np.asarray(adata.obsm["X_umap"]).copy() if "X_umap" in adata.obsm else None

    rsc.pp.neighbors(adata, use_rep="LassoARE_latent", n_neighbors=int(spec.get("n_neighbors", 30)))
    rsc.tl.leiden(adata, resolution=float(spec.get("leiden_resolution", 1.0)), key_added="leiden_LassoARE")
    rsc.tl.umap(adata)
    adata.obsm["X_umap_LassoARE"] = np.asarray(adata.obsm["X_umap"]).copy()
    if original_umap is not None:
        adata.obsm["X_umap"] = original_umap

    result_path = output_dir / "result.h5ad"
    adata.write_h5ad(result_path)
    _write_json(
        output_dir / "result.json",
        {
            "dataset_name": f"{spec['dataset_name']} ({spec['analysis_type']} rsc)",
            "analysis_type": spec["analysis_type"],
            "result_h5ad": str(result_path),
            "preferred_embedding": "X_umap_LassoARE",
            "preferred_color_by": "leiden_LassoARE",
            "interactive_kind": None,
            "needs_postprocess": False,
        },
    )


if __name__ == "__main__":
    main()
