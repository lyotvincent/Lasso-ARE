import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DockerDistributionTests(unittest.TestCase):
    def test_dockerignore_excludes_large_and_generated_content(self) -> None:
        content = (ROOT / ".dockerignore").read_text()

        for entry in (
            ".git",
            "tmp_data",
            "disabled",
            "*.h5ad",
            "*.zip",
            "frontend/node_modules",
        ):
            self.assertIn(entry, content)

    def test_dockerfile_has_cpu_and_cuda_targets(self) -> None:
        content = (ROOT / "Dockerfile").read_text()

        self.assertIn("AS cpu", content)
        self.assertIn("AS cuda", content)
        self.assertIn("lassoare-main-cpu.yml", content)
        self.assertIn("lassoare-main-cuda.yml", content)
        self.assertIn("lassoare-rsc.yml", content)

    def test_compose_binds_service_to_localhost(self) -> None:
        content = (ROOT / "compose.yaml").read_text()

        self.assertIn("127.0.0.1", content)
        self.assertIn("15114", content)
        self.assertIn("lassoare-data", content)

    def test_explicit_cuda_profile_still_checks_driver(self) -> None:
        launcher = (ROOT / "docker-start.sh").read_text()

        self.assertIn('[[ "$(detect_profile)" == "cuda" ]]', launcher)

    def test_docker_launcher_help_is_side_effect_free(self) -> None:
        completed = subprocess.run(
            [str(ROOT / "docker-start.sh"), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--profile auto|cpu|cuda", completed.stdout)


if __name__ == "__main__":
    unittest.main()
