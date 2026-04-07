import os
from typing import List

from ase import Atoms

from nepactive.native_guard import calculate_with_gpu_guard
from nepactive.nep_backend import NativeNepCalculator


def _get_force_list_native(atoms_list: List[Atoms], nep_file: str, backend: str = "auto"):
    calculator = NativeNepCalculator(model_file=nep_file, backend=backend)
    energies, forces, _ = calculator.calculate(atoms_list)
    if len(energies) != len(atoms_list) or len(forces) != len(atoms_list):
        raise RuntimeError(
            f"Native NEP returned inconsistent results for {nep_file}: "
            f"energies={len(energies)}, forces={len(forces)}, frames={len(atoms_list)}"
        )
    return list(zip(forces, energies))


def _get_force_list_guarded(atoms_list: List[Atoms], nep_file: str, backend: str = "auto"):
    energies, forces, _, _ = calculate_with_gpu_guard(atoms_list, nep_file, backend=backend)
    if len(energies) != len(atoms_list) or len(forces) != len(atoms_list):
        raise RuntimeError(
            f"Native NEP returned inconsistent results for {nep_file}: "
            f"energies={len(energies)}, forces={len(forces)}, frames={len(atoms_list)}"
        )
    return list(zip(forces, energies))


def get_force_list(atoms_list: List[Atoms], nep_file: str, backend: str = "auto"):
    """Backward-compatible public wrapper for multiprocessing pickling."""
    resolved_backend = str(backend or "auto").lower()
    if resolved_backend in {"auto", "native", "gpu"}:
        return _get_force_list_guarded(atoms_list, nep_file, backend=resolved_backend)
    return _get_force_list_native(atoms_list, nep_file, backend=resolved_backend)


def force_main(atoms_list: List[Atoms], pot_files: list[str], backend: str = "auto"):
    if not pot_files:
        raise ValueError("pot_files is empty, cannot compute model deviation force")

    backend = str(backend or "auto").lower()
    if backend not in {"auto", "native", "gpu", "cpu"}:
        raise ValueError(
            f"Unsupported deviation backend: {backend}. "
            "Use one of: auto, native, gpu, cpu."
        )

    resolved_backend = "auto" if backend in {"auto", "native"} else backend
    return [get_force_list(atoms_list, pf, backend=resolved_backend) for pf in pot_files]


if __name__ == "__main__":
    try:
        force_main(...)
    except KeyboardInterrupt:
        print("检测到中断，清理子进程...")
        os._exit(1)
