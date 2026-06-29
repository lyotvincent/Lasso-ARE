import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class AppRuntimeIntegrationTests(unittest.TestCase):
    def test_app_uses_configured_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "runtime-data"
            with patch.dict(
                os.environ,
                {
                    "LASSOARE_PROFILE": "cpu",
                    "LASSOARE_DATA_DIR": str(data_dir),
                },
                clear=False,
            ):
                import app.main

                main = importlib.reload(app.main)

            self.assertEqual(main.RUNTIME_SETTINGS.profile, "cpu")
            self.assertEqual(main.TMP_DIR, data_dir)
            self.assertEqual(main.DATASET_ROOT, data_dir / "datasets")
            self.assertEqual(main.JOB_ROOT, data_dir / "jobs")

    def test_module_runner_uses_direct_interpreter_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            stdout_path = temp / "stdout.log"
            stderr_path = temp / "stderr.log"
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

            with patch.object(main.subprocess, "run") as run:
                run.return_value.returncode = 0
                main._run_python_module(
                    python_executable=Path("/envs/lassoare_main/bin/python"),
                    module_name="backend.analysis_cli",
                    input_h5ad=Path("/data/input file.h5ad"),
                    spec_path=Path("/data/spec.json"),
                    output_dir=Path("/data/output"),
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )

            command = run.call_args.args[0]
            self.assertEqual(command[0], "/envs/lassoare_main/bin/python")
            self.assertEqual(command[1:4], ["-u", "-m", "backend.analysis_cli"])
            self.assertNotIn("bash", command)
            self.assertNotIn("conda", " ".join(command))


    def test_health_reports_runtime_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {
                    "LASSOARE_PROFILE": "cpu",
                    "LASSOARE_DATA_DIR": str(Path(temp_dir) / "data"),
                },
                clear=False,
            ):
                import app.main

                main = importlib.reload(app.main)

            payload = main.health()

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["profile"], "cpu")
        self.assertEqual(payload["torch_device"], "cpu")
        self.assertEqual(payload["rsc"], "disabled")
        self.assertFalse(payload["degraded"])

if __name__ == "__main__":
    unittest.main()
