from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


SMALL_SAMPLE_SIZE = 27_993_702
SMALL_SAMPLE_SHA256 = (
    "d614f5af06ab315c00300972e3014603755ade3c7d1cc098857e2afdc24f055b"
)


class SampleError(RuntimeError):
    pass


class SampleUnavailableError(SampleError):
    pass


class SampleIntegrityError(SampleError):
    pass


@dataclass(frozen=True)
class SampleDefinition:
    name: str
    label: str
    size_bytes: int | None
    sha256: str | None
    url_env: str | None


DEFAULT_SAMPLES = (
    SampleDefinition(
        name="sc_sampled.h5ad",
        label="Small sample (3,000 cells)",
        size_bytes=SMALL_SAMPLE_SIZE,
        sha256=SMALL_SAMPLE_SHA256,
        url_env="LASSOARE_SMALL_SAMPLE_URL",
    ),
    SampleDefinition(
        name="sc_sample_large.h5ad",
        label="Large sample",
        size_bytes=None,
        sha256=None,
        url_env=None,
    ),
)


class SampleManager:
    def __init__(
        self,
        sample_dir: Path,
        definitions: Sequence[SampleDefinition] = DEFAULT_SAMPLES,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.sample_dir = sample_dir
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.definitions = {definition.name: definition for definition in definitions}
        self.environ = os.environ if environ is None else environ

    def _definition(self, name: str) -> SampleDefinition:
        try:
            return self.definitions[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.definitions))
            raise SampleUnavailableError(
                f"Unknown sample '{name}'. Available samples: {available}."
            ) from exc

    def _validate(self, path: Path, definition: SampleDefinition) -> None:
        if definition.size_bytes is not None and path.stat().st_size != definition.size_bytes:
            raise SampleIntegrityError(
                f"{definition.name} has an unexpected size."
            )
        if definition.sha256 is not None:
            digest = hashlib.sha256()
            with path.open("rb") as sample_file:
                for chunk in iter(lambda: sample_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != definition.sha256:
                raise SampleIntegrityError(
                    f"{definition.name} failed SHA-256 verification."
                )

    def statuses(self) -> list[dict[str, object]]:
        statuses: list[dict[str, object]] = []
        for definition in self.definitions.values():
            path = self.sample_dir / definition.name
            available = False
            integrity_error: str | None = None
            if path.is_file():
                try:
                    self._validate(path, definition)
                    available = True
                except SampleIntegrityError as exc:
                    integrity_error = str(exc)
            url = (
                self.environ.get(definition.url_env, "").strip()
                if definition.url_env
                else ""
            )
            statuses.append(
                {
                    "name": definition.name,
                    "label": definition.label,
                    "size_bytes": definition.size_bytes,
                    "available": available,
                    "download_configured": bool(url),
                    "action": (
                        "load"
                        if available
                        else "download"
                        if url
                        else "unavailable"
                    ),
                    "integrity_error": integrity_error,
                }
            )
        return statuses

    def prepare(self, name: str) -> Path:
        definition = self._definition(name)
        target = self.sample_dir / definition.name
        url = (
            self.environ.get(definition.url_env, "").strip()
            if definition.url_env
            else ""
        )
        if target.is_file():
            try:
                self._validate(target, definition)
                return target
            except SampleIntegrityError:
                if not url:
                    raise
                target.unlink()

        if not url:
            raise SampleUnavailableError(
                f"Sample {definition.name} is not installed and has no download URL."
            )

        partial = target.with_suffix(target.suffix + ".part")
        partial.unlink(missing_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                with partial.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            self._validate(partial, definition)
            partial.replace(target)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-configured",
        action="store_true",
        help="Download configured samples that are not installed.",
    )
    args = parser.parse_args()

    from app.runtime import RuntimeSettings

    project_root = Path(__file__).resolve().parents[1]
    settings = RuntimeSettings.from_env(project_root=project_root)
    manager = SampleManager(settings.sample_dir)
    if args.prepare_configured:
        for status in manager.statuses():
            if not status["available"] and status["download_configured"]:
                path = manager.prepare(str(status["name"]))
                print(f"Prepared sample: {path}")
    else:
        for status in manager.statuses():
            print(status)


if __name__ == "__main__":
    main()
