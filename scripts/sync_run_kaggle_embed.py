"""Refresh embedded project snapshot inside run_kaggle.py.

Usage:
    python scripts/sync_run_kaggle_embed.py
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path


def _encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runner_path = root / "run_kaggle.py"
    if not runner_path.exists():
        raise FileNotFoundError(f"Missing file: {runner_path}")

    project_files: dict[str, str] = {
        "config.yaml": _encode_file(root / "config.yaml"),
        "main.py": _encode_file(root / "main.py"),
        "requirements.txt": _encode_file(root / "requirements.txt"),
    }

    src_root = root / "src"
    if not src_root.exists():
        raise FileNotFoundError(f"Missing folder: {src_root}")

    for file_path in sorted(src_root.rglob("*")):
        if not file_path.is_file():
            continue
        if "__pycache__" in file_path.parts or file_path.suffix == ".pyc":
            continue
        rel = file_path.relative_to(root).as_posix()
        project_files[rel] = _encode_file(file_path)

    text = runner_path.read_text(encoding="utf-8")
    replacement = "PROJECT_FILES = " + json.dumps(
        project_files,
        sort_keys=True,
        separators=(",", ":"),
    )
    updated, count = re.subn(
        r"PROJECT_FILES = \{.*?\}\n\n\ndef run",
        replacement + "\n\n\ndef run",
        text,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"Expected one PROJECT_FILES block, replaced {count}")

    runner_path.write_text(updated, encoding="utf-8")
    print(f"Updated run_kaggle.py with {len(project_files)} embedded files.")


if __name__ == "__main__":
    main()
