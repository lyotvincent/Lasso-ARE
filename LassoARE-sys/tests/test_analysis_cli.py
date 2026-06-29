import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np

try:
    from backend.analysis_cli import _common_reconstruction_kwargs, _handle_lassoare
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise unittest.SkipTest("torch is not installed in this test environment") from exc
    raise


class AnalysisCliTests(unittest.TestCase):
    def test_scanpy_backend_enables_in_process_postprocessing(self) -> None:
        kwargs = _common_reconstruction_kwargs(
            {"execution_backend": "scanpy", "is_pca": True}
        )

        self.assertTrue(kwargs["do_pp"])

    def test_rapids_backend_defers_postprocessing(self) -> None:
        kwargs = _common_reconstruction_kwargs(
            {"execution_backend": "rapids", "is_pca": True}
        )

        self.assertFalse(kwargs["do_pp"])

    def test_scanpy_backend_marks_result_as_fully_postprocessed(self) -> None:
        adata = ad.AnnData(X=np.ones((3, 2), dtype=np.float32))
        spec = {
            "dataset_name": "tiny",
            "execution_backend": "scanpy",
            "lassoare_mode": "generate",
            "selected_groups": [[0]],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "backend.analysis_cli.reconstruction_with_lasso_are",
                return_value=adata,
            ):
                result = _handle_lassoare(adata, spec, Path(temp_dir))

        self.assertFalse(result["needs_postprocess"])


if __name__ == "__main__":
    unittest.main()
