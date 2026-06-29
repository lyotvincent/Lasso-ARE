import unittest

from app.execution import resolve_execution_policy


class ExecutionPolicyTests(unittest.TestCase):
    def test_cpu_reconstruction_uses_scanpy_only(self) -> None:
        policy = resolve_execution_policy(
            profile="cpu",
            is_reconstruction=True,
            is_pca=True,
        )

        self.assertFalse(policy.use_rsc_pca)
        self.assertEqual(policy.postprocess_backend, "scanpy")
        self.assertEqual(policy.slot_label, "analysis")

    def test_cuda_reconstruction_uses_rapids_stages(self) -> None:
        policy = resolve_execution_policy(
            profile="cuda",
            is_reconstruction=True,
            is_pca=True,
        )

        self.assertTrue(policy.use_rsc_pca)
        self.assertEqual(policy.postprocess_backend, "rapids")
        self.assertEqual(policy.slot_label, "GPU")

    def test_non_reconstruction_never_uses_rsc(self) -> None:
        policy = resolve_execution_policy(
            profile="cuda",
            is_reconstruction=False,
            is_pca=True,
        )

        self.assertFalse(policy.use_rsc_pca)
        self.assertEqual(policy.postprocess_backend, "none")

    def test_cuda_skips_rsc_pca_when_pca_disabled(self) -> None:
        policy = resolve_execution_policy(
            profile="cuda",
            is_reconstruction=True,
            is_pca=False,
        )

        self.assertFalse(policy.use_rsc_pca)
        self.assertEqual(policy.postprocess_backend, "rapids")


if __name__ == "__main__":
    unittest.main()
