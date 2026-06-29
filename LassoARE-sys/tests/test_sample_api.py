import hashlib
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from app.samples import SampleDefinition, SampleManager


class SampleApiTests(unittest.TestCase):
    def test_lists_sample_status_and_reports_unavailable_load(self) -> None:
        content = b"sample"
        definition = SampleDefinition(
            name="tiny.h5ad",
            label="Tiny",
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            url_env="TEST_SAMPLE_URL",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            with patch.dict(
                os.environ,
                {
                    "LASSOARE_PROFILE": "cpu",
                    "LASSOARE_DATA_DIR": str(temp / "data"),
                },
                clear=False,
            ):
                import app.main

                main = importlib.reload(app.main)
            manager = SampleManager(
                temp / "samples",
                [definition],
                environ={},
            )

            with patch.object(main, "sample_manager", manager):
                payload = main.list_samples()
                with self.assertRaises(HTTPException) as raised:
                    main.load_sample("tiny.h5ad")

        self.assertEqual(payload["samples"][0]["action"], "unavailable")
        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
