import os
from typing import List

from ase import Atoms
from ase.io import read

from nepactive.native_guard import calculate_with_gpu_guard
from nepactive.nep_backend import create_ase_calculator
from nepactive.sampling import select_structure_indices
from nepactive.write_extxyz import write_extxyz
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


def _ensure_atoms_list(images) -> List[Atoms]:
    if isinstance(images, list):
        return images
    return [images]


def build_force_dataset(
    structure_file: str,
    number: int,
    output: str = "out.xyz",
    index: str = ":",
    fps_descriptor: str = "structural",
):
    atoms_list = _ensure_atoms_list(read(structure_file, index=index))
    if not atoms_list:
        raise ValueError(f"No frames found in {structure_file} with index={index}")

    n_select = min(int(number), len(atoms_list))
    selected_indices = select_structure_indices(
        atoms_list,
        n_samples=n_select,
        method="fps",
        descriptor_mode=fps_descriptor,
    )
    selected_atoms = [atoms_list[ii].copy() for ii in selected_indices]

    calculator = create_ase_calculator(
        model_name="mattersim",
        model_file=None,
        device="cuda",
        nep_backend="gpu",
    )
    labeled_atoms: List[Atoms] = []
    for atoms in selected_atoms:
        atoms.calc = calculator
        atoms.get_potential_energy()
        labeled_atoms.append(atoms)

    write_extxyz(output, labeled_atoms)
    return output, len(labeled_atoms)


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Use MatterSim to label FPS-selected frames and write them to out.xyz",
    )
    parser.add_argument("structure_file", help="Input structure/trajectory file")
    parser.add_argument("number", type=int, help="Number of frames to keep after FPS")
    parser.add_argument("--output", default="out.xyz", help="Output extxyz filename (default: out.xyz)")
    parser.add_argument("--index", default=":", help="ASE index expression, default ':'")
    parser.add_argument(
        "--descriptor",
        default="structural",
        choices=["structural", "soap", "nep", "auto"],
        help="FPS descriptor mode (default: structural)",
    )
    args = parser.parse_args()

    output, count = build_force_dataset(
        structure_file=args.structure_file,
        number=args.number,
        output=args.output,
        index=args.index,
        fps_descriptor=args.descriptor,
    )
    print(f"Saved {count} labeled frames to {output}")


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        print("检测到中断，清理子进程...")
        os._exit(1)
