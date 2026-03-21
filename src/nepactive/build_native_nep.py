from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sysconfig
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent
PYBIND11_INCLUDE = pybind11.get_include()
COMMON_COMPILE_ARGS = ["-O3", "-std=c++14"]
COMMON_LINK_ARGS = ["-O3", "-std=c++14"]

if os.name != "nt":
    COMMON_COMPILE_ARGS.append("-fopenmp")
    COMMON_LINK_ARGS.append("-fopenmp")


CPU_EXTENSION = Extension(
    "nep_cpu",
    [str(ROOT / "native_nep" / "nep_cpu" / "nep_cpu.cpp")],
    include_dirs=[
        PYBIND11_INCLUDE,
        str(ROOT / "native_nep" / "nep_cpu"),
    ],
    extra_compile_args=COMMON_COMPILE_ARGS,
    extra_link_args=COMMON_LINK_ARGS,
    language="c++",
)


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
GPU_EXTENSION = Extension(
    "nep_gpu",
    [str(ROOT / "native_nep" / "nep_gpu" / relpath) for relpath in GPU_SOURCES],
    include_dirs=[
        PYBIND11_INCLUDE,
        str(ROOT / "native_nep" / "nep_gpu"),
        str(ROOT / "native_nep" / "nep_gpu" / "main_nep"),
    ],
    extra_compile_args=COMMON_COMPILE_ARGS,
    extra_link_args=COMMON_LINK_ARGS,
    language="c++",
)


class BuildExtNvcc(build_ext):
    def _cuda_paths(self):
        cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_root:
            cuda_root = Path(cuda_root)
            nvcc = cuda_root / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
            include_dir = cuda_root / "include"
            lib_dir = cuda_root / ("lib/x64" if os.name == "nt" else "lib64")
            return nvcc, include_dir, lib_dir
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is None:
            raise FileNotFoundError("nvcc not found")
        nvcc = Path(nvcc_path)
        cuda_root = nvcc.parent.parent
        return nvcc, cuda_root / "include", cuda_root / "lib64"

    def build_extension(self, ext):
        if ext.name != "nep_gpu":
            return super().build_extension(ext)

        nvcc, cuda_include, cuda_lib = self._cuda_paths()
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        py_paths = sysconfig.get_paths()
        include_flags = []
        for include_dir in ext.include_dirs + [str(cuda_include), py_paths.get("include"), py_paths.get("platinclude")]:
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

        self.compiler.link_shared_object(
            objects,
            self.get_ext_fullpath(ext.name),
            libraries=["cudart", "cublas", "cusolver", "curand"],
            library_dirs=[str(cuda_lib)],
            extra_postargs=ext.extra_link_args,
            target_lang="c++",
        )


setup(
    name="nepactive-native-nep",
    ext_modules=[CPU_EXTENSION, GPU_EXTENSION],
    cmdclass={"build_ext": BuildExtNvcc},
    zip_safe=False,
)
