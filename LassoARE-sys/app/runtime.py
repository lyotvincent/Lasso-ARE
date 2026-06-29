from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


RuntimeProfile = Literal["cpu", "cuda"]


class RuntimeConfigurationError(RuntimeError):
    pass


def _env_path(name: str, default: Path | None = None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    return default


@dataclass(frozen=True)
class RuntimeSettings:
    profile: RuntimeProfile
    main_python: Path
    rsc_python: Path | None
    data_dir: Path
    sample_dir: Path

    @classmethod
    def from_env(cls, *, project_root: Path) -> "RuntimeSettings":
        del project_root
        raw_profile = os.environ.get("LASSOARE_PROFILE", "cpu").strip().lower()
        if raw_profile not in {"cpu", "cuda"}:
            raise RuntimeConfigurationError(
                "LASSOARE_PROFILE must be either 'cpu' or 'cuda'."
            )

        main_python = _env_path(
            "LASSOARE_MAIN_PYTHON",
            Path(sys.executable),
        )
        if main_python is None or not main_python.is_file():
            raise RuntimeConfigurationError(
                f"LASSOARE_MAIN_PYTHON does not exist: {main_python}"
            )

        rsc_python = _env_path("LASSOARE_RSC_PYTHON")
        if raw_profile == "cuda":
            if rsc_python is None:
                raise RuntimeConfigurationError(
                    "LASSOARE_RSC_PYTHON is required for the cuda profile."
                )
            if not rsc_python.is_file():
                raise RuntimeConfigurationError(
                    f"LASSOARE_RSC_PYTHON does not exist: {rsc_python}"
                )
        else:
            rsc_python = None

        data_dir = _env_path(
            "LASSOARE_DATA_DIR",
            Path.home() / ".local" / "share" / "lassoare",
        )
        assert data_dir is not None
        sample_dir = _env_path("LASSOARE_SAMPLE_DIR", data_dir / "samples")
        assert sample_dir is not None

        return cls(
            profile=raw_profile,
            main_python=main_python,
            rsc_python=rsc_python,
            data_dir=data_dir,
            sample_dir=sample_dir,
        )


def module_command(
    python_executable: Path,
    module_name: str,
    arguments: Iterable[str | Path],
) -> list[str]:
    return [
        str(python_executable),
        "-u",
        "-m",
        module_name,
        *(str(argument) for argument in arguments),
    ]
