import hashlib
import tempfile
import unittest
from pathlib import Path

from app.samples import (
    SampleDefinition,
    SampleIntegrityError,
    SampleManager,
    SampleUnavailableError,
)


class SampleManagerTests(unittest.TestCase):
    def definition_for(self, content: bytes) -> SampleDefinition:
        return SampleDefinition(
            name="tiny.h5ad",
            label="Tiny sample",
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            url_env="TEST_SAMPLE_URL",
        )

    def test_missing_sample_without_url_is_not_downloadable(self) -> None:
        content = b"valid sample"
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SampleManager(
                Path(temp_dir),
                [self.definition_for(content)],
                environ={},
            )

            status = manager.statuses()[0]
            with self.assertRaises(SampleUnavailableError):
                manager.prepare("tiny.h5ad")

        self.assertFalse(status["available"])
        self.assertFalse(status["download_configured"])

    def test_existing_valid_sample_is_available(self) -> None:
        content = b"valid sample"
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_dir = Path(temp_dir)
            (sample_dir / "tiny.h5ad").write_bytes(content)
            manager = SampleManager(
                sample_dir,
                [self.definition_for(content)],
                environ={},
            )

            prepared = manager.prepare("tiny.h5ad")

        self.assertEqual(prepared.name, "tiny.h5ad")

    def test_downloads_configured_sample_and_verifies_digest(self) -> None:
        content = b"downloaded sample"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.h5ad"
            source.write_bytes(content)
            sample_dir = root / "samples"
            manager = SampleManager(
                sample_dir,
                [self.definition_for(content)],
                environ={"TEST_SAMPLE_URL": source.as_uri()},
            )

            prepared = manager.prepare("tiny.h5ad")

            self.assertEqual(prepared.read_bytes(), content)
            self.assertFalse((sample_dir / "tiny.h5ad.part").exists())

    def test_bad_download_is_removed(self) -> None:
        expected = b"expected"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.h5ad"
            source.write_bytes(b"corrupt")
            sample_dir = root / "samples"
            manager = SampleManager(
                sample_dir,
                [self.definition_for(expected)],
                environ={"TEST_SAMPLE_URL": source.as_uri()},
            )

            with self.assertRaises(SampleIntegrityError):
                manager.prepare("tiny.h5ad")

            self.assertFalse((sample_dir / "tiny.h5ad").exists())
            self.assertFalse((sample_dir / "tiny.h5ad.part").exists())


    def test_corrupt_existing_sample_is_replaced_when_url_is_configured(self) -> None:
        expected = b"replacement"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.h5ad"
            source.write_bytes(expected)
            sample_dir = root / "samples"
            sample_dir.mkdir()
            (sample_dir / "tiny.h5ad").write_bytes(b"corrupt")
            manager = SampleManager(
                sample_dir,
                [self.definition_for(expected)],
                environ={"TEST_SAMPLE_URL": source.as_uri()},
            )

            prepared = manager.prepare("tiny.h5ad")

            self.assertEqual(prepared.read_bytes(), expected)

if __name__ == "__main__":
    unittest.main()
