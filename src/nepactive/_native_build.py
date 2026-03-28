from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

from setuptools import Extension
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
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
DEFAULT_GPU_ARCHS = ("75", "80", "86", "89", "90")


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


def _normalize_cuda_arch(token: str) -> str | None:
    token = token.strip().lower()
    if not token:
        return None
    token = token.removeprefix("sm_").removeprefix("compute_")
    token = token.removesuffix("-real").removesuffix("-virtual")
    token = token.replace(".", "")
    if token.isdigit():
        return token
    match = re.search(r"(\d+)(?:\.(\d+))?", token)
    if match is None:
        return None
    major = match.group(1)
    minor = match.group(2) or ""
    normalized = f"{major}{minor}"
    return normalized if normalized.isdigit() else None


def _arch_list_to_gencode_flags(arches: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for arch in arches:
        cap = _normalize_cuda_arch(arch)
        if cap is None or cap in seen:
            continue
        seen.add(cap)
        normalized.append(cap)
    if not normalized:
        return []

    flags: list[str] = []
    for cap in normalized:
        flags += ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]
    max_cap = max(normalized, key=lambda item: int(item))
    flags += ["-gencode", f"arch=compute_{max_cap},code=compute_{max_cap}"]
    return flags


def _parse_cuda_arch_list(value: str) -> list[str]:
    arches: list[str] = []
    for token in re.split(r"[,\s;]+", value.strip()):
        cap = _normalize_cuda_arch(token)
        if cap is not None:
            arches.append(cap)
    return arches


def _detect_cuda_arches_from_nvidia_smi() -> list[str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return []
    try:
        output = subprocess.check_output(
            [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return []
    return _parse_cuda_arch_list(output)


def _gpu_gencode_flags() -> tuple[list[str], str]:
    gencode_env = os.environ.get("NEP_GPU_GENCODE", "").strip()
    if gencode_env:
        parts = shlex.split(gencode_env)
        if any(part == "-gencode" for part in parts):
            return parts, "NEP_GPU_GENCODE"
        env_arches = _parse_cuda_arch_list(gencode_env)
        flags = _arch_list_to_gencode_flags(env_arches)
        if flags:
            return flags, "NEP_GPU_GENCODE"

    cudaarchs_env = os.environ.get("CUDAARCHS", "").strip()
    if cudaarchs_env:
        env_arches = _parse_cuda_arch_list(cudaarchs_env)
        flags = _arch_list_to_gencode_flags(env_arches)
        if flags:
            return flags, "CUDAARCHS"

    detected_arches = _detect_cuda_arches_from_nvidia_smi()
    flags = _arch_list_to_gencode_flags(detected_arches)
    if flags:
        return flags, "nvidia-smi"

    return _arch_list_to_gencode_flags(list(DEFAULT_GPU_ARCHS)), "fallback"


def _gpu_nvcc_threads() -> str | None:
    value = os.environ.get("NEP_GPU_NVCC_THREADS", "").strip()
    if not value:
        return "0"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid NEP_GPU_NVCC_THREADS={value!r}. Expected a non-negative integer.") from exc
    if parsed < 0:
        raise ValueError(f"Invalid NEP_GPU_NVCC_THREADS={value!r}. Expected a non-negative integer.")
    return str(parsed)


def _describe_gencode_flags(flags: list[str]) -> str:
    descriptions: list[str] = []
    index = 0
    while index < len(flags):
        if flags[index] == "-gencode" and index + 1 < len(flags):
            descriptions.append(flags[index + 1])
            index += 2
            continue
        descriptions.append(flags[index])
        index += 1
    return ", ".join(descriptions)


def _emit_build_status(message: str) -> None:
    print(message, flush=True)


def _pybind11_include() -> str:
    try:
        import pybind11
    except ImportError as exc:
        raise RuntimeError(
            "pybind11 is required to build NEP native extensions. "
            "Install it in the build environment or run with NEP_NATIVE_BUILD=none."
        ) from exc
    return pybind11.get_include()


def _project_relpath(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _project_abspath(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _extension_suffixes() -> list[str]:
    return sorted(EXTENSION_SUFFIXES, key=len, reverse=True)


def _find_inplace_target(name: str) -> Path | None:
    for suffix in _extension_suffixes():
        candidate = ROOT / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _extension_dependencies(ext: Extension) -> list[Path]:
    dependencies = [_project_abspath(src) for src in ext.sources]
    module_basename = ext.name.rsplit(".", 1)[-1]
    if module_basename == "nep_cpu":
        native_root = ROOT / "native_nep" / "nep_cpu"
    elif module_basename == "nep_gpu":
        native_root = ROOT / "native_nep" / "nep_gpu"
    else:
        return dependencies

    include_suffixes = {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".h", ".hpp"}
    for path in native_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in include_suffixes:
            dependencies.append(path)
    return dependencies


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
    source_path = ROOT / "native_nep" / "nep_cpu" / "nep_cpu.cpp"
    include_dir = ROOT / "native_nep" / "nep_cpu"
    return Extension(
        "nepactive.nep_cpu",
        [_project_relpath(source_path)],
        include_dirs=[
            pybind11_include,
            _project_relpath(include_dir),
        ],
        extra_compile_args=_common_compile_args(),
        extra_link_args=_common_link_args(),
        language="c++",
    )


def _gpu_extension() -> Extension:
    pybind11_include = _pybind11_include()
    gpu_root = ROOT / "native_nep" / "nep_gpu"
    return Extension(
        "nepactive.nep_gpu",
        [_project_relpath(gpu_root / relpath) for relpath in GPU_SOURCES],
        include_dirs=[
            pybind11_include,
            _project_relpath(gpu_root),
            _project_relpath(gpu_root / "main_nep"),
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
        dependencies = _extension_dependencies(ext)
        inplace_target = _find_inplace_target(module_basename)
        preferred_target = target
        if self.inplace and inplace_target is not None:
            target = inplace_target
        if self.force or _parse_truthy_env("NEP_NATIVE_REBUILD"):
            return False
        if _is_up_to_date(target, dependencies):
            self.announce(f"skipping {ext.name} (up-to-date): {target}", level=2)
            return True
        if (
            not self.inplace
            and inplace_target is not None
            and inplace_target != preferred_target
            and _is_up_to_date(inplace_target, dependencies)
        ):
            preferred_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(inplace_target, preferred_target)
            self.announce(
                f"reusing prebuilt {ext.name}: {inplace_target} -> {preferred_target}",
                level=2,
            )
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
                include_flags.extend(["-I", str(_project_abspath(include_dir))])

        nvcc_flags = ["-O3", "-std=c++14", "-Xcompiler", "-fPIC,-fopenmp"]
        nvcc_threads = _gpu_nvcc_threads()
        if nvcc_threads is not None:
            nvcc_flags += ["--threads", nvcc_threads]

        gencode_flags, gencode_source = _gpu_gencode_flags()
        nvcc_flags.extend(gencode_flags)
        _emit_build_status(
            "building nep_gpu with "
            f"{len(ext.sources)} CUDA translation units; "
            f"gencode source={gencode_source}; "
            f"targets={_describe_gencode_flags(gencode_flags)}; "
            f"nvcc threads={nvcc_threads or 'disabled'}"
        )
        if gencode_source == "fallback":
            self.warn(
                "Unable to detect local GPU compute capability via CUDAARCHS or nvidia-smi; "
                "falling back to a broad CUDA arch set. "
                "Set CUDAARCHS or NEP_GPU_GENCODE to narrow the build and speed up compilation."
            )

        objects: list[str] = []
        for index, src in enumerate(ext.sources, start=1):
            src_path = _project_abspath(src)
            obj_path = build_temp / f"{src_path.stem}.o"
            _emit_build_status(f"[{index}/{len(ext.sources)}] nvcc compiling {src_path.relative_to(ROOT)}")
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
