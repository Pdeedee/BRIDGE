from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms

from nepactive import dlog
from nepactive.nep_backend import NativeNepCalculator


def _normalize_backend(backend: str | None) -> str:
    resolved = str(backend or "auto").lower()
    if resolved not in {"auto", "native", "gpu", "cpu"}:
        raise ValueError(f"Unsupported NEP backend: {backend}")
    return resolved


def _run_native_locally(
    task: str,
    structures: Iterable[Atoms] | Atoms,
    model_file: str,
    backend: str,
):
    calculator = NativeNepCalculator(model_file=model_file, backend=backend)
    if task == "calculate":
        return calculator.calculate(structures, mean_virial=True)
    if task == "descriptor":
        return calculator.get_structures_descriptor(structures)
    raise ValueError(f"Unsupported native task: {task}")


def _worker_script() -> Path:
    return Path(__file__).resolve().with_name("_native_guard_worker.py")


def _summarize_subprocess_failure(process: subprocess.CompletedProcess[str]) -> str:
    parts: list[str] = [f"exit code {process.returncode}"]
    stderr = (process.stderr or "").strip()
    stdout = (process.stdout or "").strip()
    if stderr:
        parts.append(stderr.splitlines()[-1])
    elif stdout:
        parts.append(stdout.splitlines()[-1])
    return "; ".join(parts)


def _run_native_in_subprocess(
    task: str,
    structures: Iterable[Atoms] | Atoms,
    model_file: str,
    backend: str,
):
    payload = {
        "task": task,
        "structures": list(structures) if not isinstance(structures, Atoms) else [structures],
        "model_file": str(model_file),
        "backend": backend,
    }
    worker = _worker_script()
    with tempfile.TemporaryDirectory(prefix="nepactive-native-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        request_path = tmpdir_path / "request.pkl"
        response_path = tmpdir_path / "response.pkl"
        request_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        process = subprocess.run(
            [sys.executable, str(worker), str(request_path), str(response_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(_summarize_subprocess_failure(process))
        if not response_path.exists():
            raise RuntimeError("worker finished without producing a response file")
        return pickle.loads(response_path.read_bytes())


def calculate_with_gpu_guard(
    structures: Iterable[Atoms] | Atoms,
    model_file: str,
    backend: str = "auto",
) -> tuple[list[float], list[np.ndarray], list[np.ndarray], str]:
    resolved_backend = _normalize_backend(backend)
    if resolved_backend == "cpu":
        energies, forces, virials = _run_native_locally("calculate", structures, model_file, "cpu")
        return energies, forces, virials, "cpu"
    if resolved_backend == "gpu":
        energies, forces, virials = _run_native_in_subprocess("calculate", structures, model_file, "gpu")
        return energies, forces, virials, "gpu"

    try:
        energies, forces, virials = _run_native_in_subprocess("calculate", structures, model_file, "gpu")
        return energies, forces, virials, "gpu"
    except Exception as exc:
        dlog.warning("GPU NEP calculate failed for %s, fallback to CPU: %s", model_file, exc)
        energies, forces, virials = _run_native_locally("calculate", structures, model_file, "cpu")
        return energies, forces, virials, "cpu"


def get_structures_descriptor_with_gpu_guard(
    structures: Iterable[Atoms] | Atoms,
    model_file: str,
    backend: str = "auto",
) -> tuple[np.ndarray, str]:
    resolved_backend = _normalize_backend(backend)
    if resolved_backend == "cpu":
        return np.asarray(_run_native_locally("descriptor", structures, model_file, "cpu"), dtype=np.float32), "cpu"
    if resolved_backend == "gpu":
        return np.asarray(_run_native_in_subprocess("descriptor", structures, model_file, "gpu"), dtype=np.float32), "gpu"

    try:
        descriptor = _run_native_in_subprocess("descriptor", structures, model_file, "gpu")
        return np.asarray(descriptor, dtype=np.float32), "gpu"
    except Exception as exc:
        dlog.warning("GPU NEP descriptor failed for %s, fallback to CPU: %s", model_file, exc)
        descriptor = _run_native_locally("descriptor", structures, model_file, "cpu")
        return np.asarray(descriptor, dtype=np.float32), "cpu"
