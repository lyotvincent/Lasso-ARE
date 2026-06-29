import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EnvironmentManifestTests(unittest.TestCase):
    def test_environment_names_and_profiles_are_declared(self) -> None:
        cpu = (ROOT / "environments" / "lassoare-main-cpu.yml").read_text()
        cuda = (ROOT / "environments" / "lassoare-main-cuda.yml").read_text()
        rsc = (ROOT / "environments" / "lassoare-rsc.yml").read_text()

        self.assertIn("name: lassoare_main", cpu)
        self.assertIn("python=3.10", cpu)
        self.assertIn("name: lassoare_main", cuda)
        self.assertIn("python=3.10", cuda)
        self.assertIn("name: lassoare_rsc", rsc)
        self.assertIn("python=3.12", rsc)
        self.assertIn("cuda-version=12.8", rsc)
        self.assertNotIn("jupyter", (cpu + cuda + rsc).lower())
        self.assertNotIn("\n  - rapids=", rsc)

    def test_torch_requirements_select_cpu_and_cuda_indexes(self) -> None:
        cpu = (ROOT / "environments" / "torch-cpu.txt").read_text()
        cuda = (ROOT / "environments" / "torch-cu128.txt").read_text()

        self.assertIn("download.pytorch.org/whl/cpu", cpu)
        self.assertIn("torch==2.9.1", cpu)
        self.assertIn("download.pytorch.org/whl/cu128", cuda)
        self.assertIn("torch==2.9.1", cuda)


class InstallerContractTests(unittest.TestCase):
    def test_installer_help_is_side_effect_free(self) -> None:
        completed = subprocess.run(
            [str(ROOT / "install.sh"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--profile auto|cpu|cuda", completed.stdout)
        self.assertIn("--no-start", completed.stdout)

    def test_start_help_documents_localhost_default(self) -> None:
        completed = subprocess.run(
            [str(ROOT / "start.sh"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("127.0.0.1", completed.stdout)
        self.assertIn("15114", completed.stdout)


    def test_installer_persists_and_prepares_configured_sample_url(self) -> None:
        installer = (ROOT / "install.sh").read_text()

        self.assertIn("LASSOARE_SMALL_SAMPLE_URL", installer)
        self.assertIn("app.samples --prepare-configured", installer)

    def test_legacy_runner_delegates_without_deleting_data(self) -> None:
        runner = (ROOT / "run_app.sh").read_text()

        self.assertIn("start.sh", runner)
        self.assertNotIn("rm -r", runner)
        self.assertNotIn("conda activate lsys", runner)

    def test_installer_isolates_declared_channels_from_user_condarc(self) -> None:
        installer = (ROOT / "install.sh").read_text()

        self.assertIn("export CONDARC", installer)
        self.assertIn("channels: []", installer)

    def test_torch_install_streams_without_conda_run_capture(self) -> None:
        installer = (ROOT / "install.sh").read_text()

        self.assertIn("Installing PyTorch profile packages", installer)
        self.assertIn('"$MAIN_PYTHON" -m pip install', installer)
        self.assertNotIn('"$ENV_MANAGER" run -n lassoare_main', installer)

    def test_bare_sha256_files_are_supported(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            artifact = temp / "artifact"
            artifact.write_bytes(b"micromamba")
            digest = __import__("hashlib").sha256(b"micromamba").hexdigest()
            checksum = temp / "artifact.sha256"
            checksum.write_text(digest)
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    f"source {ROOT / 'scripts/common.sh'}; verify_bare_sha256 {artifact} {checksum}",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)

if __name__ == "__main__":
    unittest.main()
