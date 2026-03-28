from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

from setuptools import Extension
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent
GPU_SOURCES = [
    "nep_gpu.cu",
    "nep_desc.cu",
    "nep_parameters.cu",
    "main_nep/dataset.cu",
    "main_nep/nep.cu",
    "main_nep/nep_charge.cu",
    "main_nep/parameters.cu",
    "main_nep/structure.cu",
    "main_nep/tnep.cu",
    "utilities/cusolver_wrapper.cu",
    "utilities/error.cu",
    "utilities/main_common.cu",
    "utilities/read_file.cu",
]
VALID_BUILD_MODES = {"auto", "none", "cpu", "gpu", "all"}


def _parse_truthy_env(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _build_mode() -> str:
    mode = os.environ.get("NEP_NATIVE_BUILD", "auto").strip().lower() or "auto"
    if mode not in VALID_BUILD_MODES:
        valid = ", ".join(sorted(VALID_BUILD_MODES))
        raise ValueError(f"Invalid NEP_NATIVE_BUILD={mode!r}. Expected one of: {valid}.")
    return mode


def _common_compile_args() -> list[str]:
    args = ["-O3", "-std=c++14"]
    if os.name != "nt":
        args.append("-fopenmp")
    return args


def _common_link_args() -> list[str]:
    args = ["-O3", "-std=c++14"]
    if os.name != "nt":
        args.append("-fopenmp")
    return args


def _pybind11_include() -> str:
    try:
        import pybind11
    except ImportError as exc:
        raise RuntimeError(
            "pybind11 is required to build NEP native extensions. "
            "Install it in the build environment or run with NEP_NATIVE_BUILD=none."
        ) from exc
    return pybind11.get_include()


def _extension_suffixes() -> list[str]:
    return sorted(EXTENSION_SUFFIXES, key=len, reverse=True)


def _find_inplace_target(name: str) -> Path | None:
    for suffix in _extension_suffixes():
        candidate = ROOT / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _is_up_to_date(target: Path, dependencies: list[Path]) -> bool:
    if not target.exists():
        return False
    try:
        target_mtime = target.stat().st_mtime
    except OSError:
        return False

    for dependency in dependencies:
        try:
            if dependency.stat().st_mtime > target_mtime:
                return False
        except OSError:
            return False
    return True


def cuda_paths() -> tuple[Path, Path, Path] | None:
    cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_root:
        cuda_root_path = Path(cuda_root)
        nvcc = cuda_root_path / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
        include_dir = cuda_root_path / "include"
        lib_dir = cuda_root_path / ("lib/x64" if os.name == "nt" else "lib64")
        if nvcc.exists():
            return nvcc, include_dir, lib_dir

    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        return None
    nvcc = Path(nvcc_path)
    cuda_root_path = nvcc.parent.parent
    return nvcc, cuda_root_path / "include", cuda_root_path / "lib64"


def _cpu_extension() -> Extension:
    pybind11_include = _pybind11_include()
    return Extension(
        "nepactive.nep_cpu",
        [str(ROOT / "native_nep" / "nep_cpu" / "nep_cpu.cpp")],
        include_dirs=[
            pybind11_include,
            str(ROOT / "native_nep" / "nep_cpu"),
        ],
        extra_compile_args=_common_compile_args(),
        extra_link_args=_common_link_args(),
        language="c++",
    )


def _gpu_extension() -> Extension:
    pybind11_include = _pybind11_include()
    return Extension(
        "nepactive.nep_gpu",
        [str(ROOT / "native_nep" / "nep_gpu" / relpath) for relpath in GPU_SOURCES],
        include_dirs=[
            pybind11_include,
            str(ROOT / "native_nep" / "nep_gpu"),
            str(ROOT / "native_nep" / "nep_gpu" / "main_nep"),
        ],
        extra_compile_args=_common_compile_args(),
        extra_link_args=_common_link_args(),
        language="c++",
    )


def get_ext_modules() -> list[Extension]:
    mode = _build_mode()
    if mode == "none":
        return []

    cuda_available = cuda_paths() is not None
    if mode in {"gpu", "all"} and not cuda_available:
        raise RuntimeError(
            "CUDA toolchain not found. Set CUDA_HOME/CUDA_PATH or install nvcc, "
            "or use NEP_NATIVE_BUILD=cpu/none."
        )

    extensions: list[Extension] = []
    if mode in {"auto", "cpu", "all"}:
        extensions.append(_cpu_extension())
    if mode in {"gpu", "all"} or (mode == "auto" and cuda_available):
        extensions.append(_gpu_extension())
    return extensions


class SmartBuildExt(build_ext):
    def _maybe_skip(self, ext: Extension) -> bool:
        module_basename = ext.name.rsplit(".", 1)[-1]
        target = Path(self.get_ext_fullpath(ext.name))
        dependencies = [Path(src) for src in ext.sources]
        dependencies.extend([ROOT / "build_native_nep.py", ROOT / "_native_build.py"])
        if self.inplace:
            inplace_target = _find_inplace_target(module_basename)
            if inplace_target is not None:
                target = inplace_target
        if self.force or _parse_truthy_env("NEP_NATIVE_REBUILD"):
            return False
        if _is_up_to_date(target, dependencies):
            self.announce(f"skipping {ext.name} (up-to-date): {target}", level=2)
            return True
        return False

    def _build_gpu_extension(self, ext: Extension) -> None:
        cuda = cuda_paths()
        if cuda is None:
            raise FileNotFoundError("nvcc not found")
        nvcc, cuda_include, cuda_lib = cuda
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        py_paths = sysconfig.get_paths()
        include_flags: list[str] = []
        include_dirs = ext.include_dirs + [
            str(cuda_include),
            py_paths.get("include"),
            py_paths.get("platinclude"),
        ]
        for include_dir in include_dirs:
            if include_dir:
                include_flags.extend(["-I", str(include_dir)])

        nvcc_flags = ["-O3", "-std=c++14", "-Xcompiler", "-fPIC,-fopenmp"]
        gencode_env = os.environ.get("NEP_GPU_GENCODE", "").strip()
        if gencode_env:
            parts = shlex.split(gencode_env)
            if any(part == "-gencode" for part in parts):
                nvcc_flags.extend(parts)
            else:
                nvcc_flags += ["-gencode", gencode_env]
        else:
            nvcc_flags += ["-gencode", "arch=compute_75,code=sm_75"]
            nvcc_flags += ["-gencode", "arch=compute_80,code=sm_80"]
            nvcc_flags += ["-gencode", "arch=compute_86,code=sm_86"]
            nvcc_flags += ["-gencode", "arch=compute_89,code=sm_89"]
            nvcc_flags += ["-gencode", "arch=compute_90,code=sm_90"]
            nvcc_flags += ["-gencode", "arch=compute_90,code=compute_90"]

        objects: list[str] = []
        for src in ext.sources:
            src_path = Path(src)
            obj_path = build_temp / f"{src_path.stem}.o"
            cmd = [str(nvcc), "-c", str(src_path), "-o", str(obj_path), *nvcc_flags, *include_flags]
            subprocess.check_call(cmd)
            objects.append(str(obj_path))

        output_path = Path(self.get_ext_fullpath(ext.name))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.compiler.link_shared_object(
            objects,
            str(output_path),
            libraries=["cudart", "cublas", "cusolver", "curand"],
            library_dirs=[str(cuda_lib)],
            extra_postargs=ext.extra_link_args,
            target_lang="c++",
        )

    def build_extension(self, ext: Extension) -> None:
        if self._maybe_skip(ext):
            return

        is_optional = _build_mode() == "auto"
        try:
            if ext.name.endswith(".nep_gpu"):
                self._build_gpu_extension(ext)
            else:
                super().build_extension(ext)
        except Exception as exc:
            if is_optional:
                self.warn(f"Skipping optional native extension {ext.name}: {exc}")
                return
            raise


def get_setup_kwargs() -> dict[str, object]:
    return {
        "ext_modules": get_ext_modules(),
        "cmdclass": {"build_ext": SmartBuildExt},
        "zip_safe": False,
    }
