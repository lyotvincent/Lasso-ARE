import importlib.util
import unittest

import anndata as ad
import numpy as np

from backend.scanpy_postprocess_cli import postprocess_with_scanpy


class ScanpyPostprocessTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("igraph"), "igraph is required")
    def test_creates_lassoare_umap_and_leiden_outputs(self) -> None:
        rng = np.random.default_rng(7)
        adata = ad.AnnData(X=rng.normal(size=(20, 4)).astype(np.float32))
        adata.obsm["LassoARE_latent"] = rng.normal(
            size=(20, 3)
        ).astype(np.float32)

        postprocess_with_scanpy(
            adata,
            {"n_neighbors": 5, "leiden_resolution": 0.5},
        )

        self.assertIn("X_umap_LassoARE", adata.obsm)
        self.assertEqual(adata.obsm["X_umap_LassoARE"].shape, (20, 2))
        self.assertIn("leiden_LassoARE", adata.obs)


if __name__ == "__main__":
    unittest.main()
