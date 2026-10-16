"""Resolve data paths independently of the Jupyter kernel working directory."""
from pathlib import Path
import os


def repository_root(supp_dir):
    """Also support a standalone copy of the supplementary folder."""
    supp_dir = Path(supp_dir).resolve()
    return supp_dir.parent.parent if supp_dir.parent.name == "figures_res" else supp_dir


def resolve_data_paths(supp_dir, data_files, data_root=None):
    """Use an explicit root, or the nearest ancestor containing the input files.

    Search only the exact relative paths listed by the notebook. Prefer a root
    containing every input. If only some inputs are available, use the nearest
    matching root so that independent S1 sections can still run. Never combine
    relative inputs from different roots. Absolute per-file overrides are allowed.
    Relative root overrides are interpreted relative to the notebook folder.
    Bundled-table notebooks do not require or inspect an external data root.
    """
    supp_dir = Path(supp_dir).resolve()
    if not data_files:
        return None, {}
    files = {name: Path(value).expanduser() for name, value in data_files.items()}
    relative = [path for path in files.values() if not path.is_absolute()]
    override = data_root if data_root is not None else os.environ.get("LASSOARE_DATA_ROOT")
    if override is not None:
        root = Path(override).expanduser()
        if not root.is_absolute():
            root = supp_dir / root
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"Configured DATA_ROOT is not a directory: {root}\n"
                "Edit DATA_ROOT in the first code cell or unset LASSOARE_DATA_ROOT "
                "to enable automatic discovery."
            )
    else:
        candidates = [supp_dir, *supp_dir.parents]
        matches = [(base, sum((base / path).is_file() for path in relative))
                   for base in candidates]
        complete = [base for base, count in matches if relative and count == len(relative)]
        partial = [base for base, count in matches if count]
        root = next(iter(complete or partial), repository_root(supp_dir))
    paths = {name: (root / path).resolve() for name, path in files.items()}
    return root, paths


def report_paths(supp_dir, data_root, data_paths, output_dir):
    """Show exactly which inputs will be read before plotting starts."""
    print("Notebook folder:", supp_dir)
    print("Bundled inputs:", Path(supp_dir) / "inputs")
    if data_paths:
        print("Data root:", data_root)
        for name, path in data_paths.items():
            print(f"  {'OK' if path.is_file() else 'MISSING'} {name}: {path}")
        if any(not path.is_file() for path in data_paths.values()):
            print("Missing H5AD files are not bundled. Set DATA_ROOT or edit DATA_FILES "
                  "in the first code cell, then rerun from that cell. "
                  "Only panels using missing inputs are unavailable.")
    else:
        print("H5AD inputs: none (bundled result tables only)")
    print("Outputs:", output_dir)
