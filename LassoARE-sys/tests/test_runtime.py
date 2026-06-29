import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.runtime import RuntimeConfigurationError, RuntimeSettings, module_command


class RuntimeSettingsTests(unittest.TestCase):
    def test_defaults_to_cpu_and_current_python(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = RuntimeSettings.from_env(project_root=Path("/srv/lassoare"))

        self.assertEqual(settings.profile, "cpu")
        self.assertEqual(settings.main_python, Path(sys.executable))
        self.assertIsNone(settings.rsc_python)
        self.assertEqual(
            settings.data_dir,
            Path.home() / ".local" / "share" / "lassoare",
        )

    def test_cuda_requires_rsc_python(self) -> None:
        with patch.dict(
            os.environ,
            {"LASSOARE_PROFILE": "cuda"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                RuntimeConfigurationError,
                "LASSOARE_RSC_PYTHON",
            ):
                RuntimeSettings.from_env(project_root=Path("/srv/lassoare"))

    def test_environment_overrides_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            main_python = temp / "main-python"
            rsc_python = temp / "rsc-python"
            main_python.touch()
            rsc_python.touch()
            data_dir = temp / "data"
            sample_dir = temp / "samples"
            with patch.dict(
                os.environ,
                {
                    "LASSOARE_PROFILE": "cuda",
                    "LASSOARE_MAIN_PYTHON": str(main_python),
                    "LASSOARE_RSC_PYTHON": str(rsc_python),
                    "LASSOARE_DATA_DIR": str(data_dir),
                    "LASSOARE_SAMPLE_DIR": str(sample_dir),
                },
                clear=True,
            ):
                settings = RuntimeSettings.from_env(project_root=temp)

        self.assertEqual(settings.profile, "cuda")
        self.assertEqual(settings.main_python, main_python)
        self.assertEqual(settings.rsc_python, rsc_python)
        self.assertEqual(settings.data_dir, data_dir)
        self.assertEqual(settings.sample_dir, sample_dir)


class ModuleCommandTests(unittest.TestCase):
    def test_builds_direct_python_module_command(self) -> None:
        command = module_command(
            Path("/envs/lassoare_main/bin/python"),
            "backend.analysis_cli",
            [
                "--input-h5ad",
                Path("/data/input file.h5ad"),
                "--spec",
                Path("/data/spec.json"),
            ],
        )

        self.assertEqual(
            command,
            [
                "/envs/lassoare_main/bin/python",
                "-u",
                "-m",
                "backend.analysis_cli",
                "--input-h5ad",
                "/data/input file.h5ad",
                "--spec",
                "/data/spec.json",
            ],
        )


if __name__ == "__main__":
    unittest.main()
