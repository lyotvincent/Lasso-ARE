import importlib
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class JobRoutingTests(unittest.TestCase):
    def test_cuda_degradation_is_cleared_after_analysis_slot_is_acquired(self) -> None:
        import app.main

        source = inspect.getsource(app.main._run_analysis_job)
        self.assertGreater(
            source.index("_set_runtime_degraded(None)"),
            source.index("with analysis_job_lock"),
        )

    def test_cpu_reconstruction_skips_rsc_and_marks_scanpy_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            with patch.dict(
                os.environ,
                {
                    "LASSOARE_PROFILE": "cpu",
                    "LASSOARE_DATA_DIR": str(temp / "runtime"),
                },
                clear=False,
            ):
                import app.main

                main = importlib.reload(app.main)

            job_dir = temp / "job"
            job_dir.mkdir()
            main._write_json(
                job_dir / "spec.json",
                {
                    "analysis_type": "lassoare",
                    "is_pca": True,
                    "dataset_name": "tiny",
                },
            )
            job = main.AnalysisJob(
                job_id="job-1",
                dataset_id="dataset-1",
                analysis_type="lassoare",
                status="queued",
                message="queued",
                progress=0.0,
                job_dir=str(job_dir),
            )
            jobs = main.JobStore()
            jobs.create(job)
            calls: list[tuple[Path | None, str]] = []
            captured_specs: list[dict[str, object]] = []

            def fake_runner(**kwargs: object) -> None:
                calls.append(
                    (
                        kwargs["python_executable"],  # type: ignore[arg-type]
                        str(kwargs["module_name"]),
                    )
                )
                spec = main._read_json(Path(kwargs["spec_path"]))  # type: ignore[arg-type]
                captured_specs.append(spec)
                main._write_json(
                    job_dir / "result.json",
                    {
                        "result_h5ad": str(job_dir / "result.h5ad"),
                        "needs_postprocess": False,
                    },
                )

            with (
                patch.object(main, "job_store", jobs),
                patch.object(
                    main.store,
                    "get_source_path",
                    return_value=temp / "input.h5ad",
                ),
                patch.object(main, "_run_python_module", side_effect=fake_runner),
                patch.object(main, "_complete_job"),
            ):
                main._run_analysis_job("job-1")

            self.assertEqual(
                calls,
                [(main.RUNTIME_SETTINGS.main_python, "backend.analysis_cli")],
            )
            self.assertEqual(captured_specs[0]["execution_backend"], "scanpy")
            self.assertEqual(jobs.get("job-1").error, None)


    def test_cuda_postprocess_failure_falls_back_to_scanpy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            main_python = temp / "main-python"
            rsc_python = temp / "rsc-python"
            main_python.touch()
            rsc_python.touch()
            with patch.dict(
                os.environ,
                {
                    "LASSOARE_PROFILE": "cuda",
                    "LASSOARE_MAIN_PYTHON": str(main_python),
                    "LASSOARE_RSC_PYTHON": str(rsc_python),
                    "LASSOARE_DATA_DIR": str(temp / "runtime"),
                },
                clear=False,
            ):
                import app.main

                main = importlib.reload(app.main)

            job_dir = temp / "job"
            job_dir.mkdir()
            main._write_json(
                job_dir / "spec.json",
                {
                    "analysis_type": "lassoare",
                    "is_pca": True,
                    "dataset_name": "tiny",
                },
            )
            jobs = main.JobStore()
            jobs.create(
                main.AnalysisJob(
                    job_id="job-cuda",
                    dataset_id="dataset-1",
                    analysis_type="lassoare",
                    status="queued",
                    message="queued",
                    progress=0.0,
                    job_dir=str(job_dir),
                )
            )
            calls: list[str] = []

            def fake_runner(**kwargs: object) -> None:
                module = str(kwargs["module_name"])
                calls.append(module)
                if module == "backend.rsc_pca_preprocess_cli":
                    main._write_json(
                        job_dir / "pca_preprocess.json",
                        {"result_h5ad": str(job_dir / "pca.h5ad")},
                    )
                elif module == "backend.analysis_cli":
                    main._write_json(
                        job_dir / "result.json",
                        {
                            "result_h5ad": str(job_dir / "intermediate.h5ad"),
                            "needs_postprocess": True,
                        },
                    )
                elif module == "backend.rsc_postprocess_cli":
                    raise RuntimeError("simulated RAPIDS failure")
                elif module == "backend.scanpy_postprocess_cli":
                    main._write_json(
                        job_dir / "result.json",
                        {
                            "result_h5ad": str(job_dir / "result.h5ad"),
                            "needs_postprocess": False,
                            "postprocess_backend": "scanpy",
                        },
                    )

            with (
                patch.object(main, "job_store", jobs),
                patch.object(
                    main.store,
                    "get_source_path",
                    return_value=temp / "input.h5ad",
                ),
                patch.object(main, "_run_python_module", side_effect=fake_runner),
                patch.object(main, "_complete_job") as complete,
            ):
                main._run_analysis_job("job-cuda")

            self.assertEqual(
                calls,
                [
                    "backend.rsc_pca_preprocess_cli",
                    "backend.analysis_cli",
                    "backend.rsc_postprocess_cli",
                    "backend.scanpy_postprocess_cli",
                ],
            )
            self.assertEqual(
                complete.call_args.args[1]["postprocess_backend"],
                "scanpy",
            )
            self.assertEqual(
                main._read_json(job_dir / "spec.json")["execution_backend"],
                "scanpy-fallback",
            )
            self.assertIn("simulated RAPIDS failure", (job_dir / "stdout.log").read_text())
            self.assertTrue(main.health()["degraded"])
