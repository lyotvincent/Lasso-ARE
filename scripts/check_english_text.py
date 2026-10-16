#!/usr/bin/env python3
"""Reject Han characters in Git-visible text and decoded notebook/JSON content.

Run from any directory: python scripts/check_english_text.py
Ignored dependencies, generated plots, binary files and Git internals are excluded.
Third-party locale strings in built JavaScript use ASCII escapes for compatibility;
this check inspects their distributed source bytes, not runtime locale semantics.
"""
from pathlib import Path
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f]")


def strings(value, location=""):
    if isinstance(value, str):
        yield location, value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from strings(item, f"{location}/{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield f"{location}/key", key
            yield from strings(item, f"{location}/{key}")


def main():
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, check=True, capture_output=True,
    )
    names = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
    failures = []
    checked = 0
    for name in names:
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        checked += 1
        if HAN.search(name):
            failures.append(f"{name}: filename")
        for number, line in enumerate(text.splitlines(), 1):
            if HAN.search(line):
                failures.append(f"{name}:{number}: Han characters in source text")
        if path.suffix in {".ipynb", ".json"}:
            for location, value in strings(json.loads(text)):
                if HAN.search(value):
                    failures.append(f"{name}:{location}: Han characters in decoded JSON")
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"PASS: {checked} Git-visible text files contain no Han characters.")


if __name__ == "__main__":
    main()
