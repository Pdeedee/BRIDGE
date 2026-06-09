from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from nepactive import dlog


_ITER_DIR_RE = re.compile(r"^iter\.(\d+)$")


def _iter_sort_key(path: Path) -> tuple[int, str]:
    match = _ITER_DIR_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def _dataset_files(run_dir: Path, kind: str) -> list[Path]:
    if kind not in {"train", "test"}:
        raise ValueError(f"unsupported dataset kind: {kind}")

    name = f"iter_{kind}.xyz"
    files: list[Path] = []

    init_file = run_dir / "init" / name
    if init_file.is_file():
        files.append(init_file)

    iter_dirs = [
        path
        for path in run_dir.iterdir()
        if path.is_dir() and _ITER_DIR_RE.match(path.name)
    ]
    for iter_dir in sorted(iter_dirs, key=_iter_sort_key):
        label_file = iter_dir / "02.label" / name
        if label_file.is_file():
            files.append(label_file)

    return files


def _merge_files(files: list[Path], output_file: Path) -> None:
    with output_file.open("wb") as output:
        for file_path in files:
            with file_path.open("rb") as source:
                shutil.copyfileobj(source, output)


def build_dataset(run_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str] = ".") -> tuple[Path, Path, int, int]:
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"nepactive run directory not found: {run_path}")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    train_files = _dataset_files(run_path, "train")
    test_files = _dataset_files(run_path, "test")
    if not train_files:
        raise FileNotFoundError(
            f"no training dataset files found under {run_path}/init or {run_path}/iter.*/02.label"
        )

    train_output = output_path / "train.xyz"
    test_output = output_path / "test.xyz"
    _merge_files(train_files, train_output)
    _merge_files(test_files, test_output)

    dlog.info("merged %d train files into %s", len(train_files), train_output)
    dlog.info("merged %d test files into %s", len(test_files), test_output)
    return train_output, test_output, len(train_files), len(test_files)
