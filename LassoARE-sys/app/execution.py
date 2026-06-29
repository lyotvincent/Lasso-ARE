from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.runtime import RuntimeProfile


PostprocessBackend = Literal["none", "scanpy", "rapids"]


@dataclass(frozen=True)
class ExecutionPolicy:
    use_rsc_pca: bool
    postprocess_backend: PostprocessBackend
    slot_label: str


def resolve_execution_policy(
    *,
    profile: RuntimeProfile,
    is_reconstruction: bool,
    is_pca: bool,
) -> ExecutionPolicy:
    if not is_reconstruction:
        return ExecutionPolicy(
            use_rsc_pca=False,
            postprocess_backend="none",
            slot_label="analysis",
        )
    if profile == "cpu":
        return ExecutionPolicy(
            use_rsc_pca=False,
            postprocess_backend="scanpy",
            slot_label="analysis",
        )
    return ExecutionPolicy(
        use_rsc_pca=is_pca,
        postprocess_backend="rapids",
        slot_label="GPU",
    )
