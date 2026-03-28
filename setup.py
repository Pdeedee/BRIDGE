from __future__ import annotations

import importlib.util
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
HELPER_PATH = ROOT / "src" / "nepactive" / "_native_build.py"
SPEC = importlib.util.spec_from_file_location("nepactive_native_build", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load native build helper: {HELPER_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    **MODULE.get_setup_kwargs(),
)
