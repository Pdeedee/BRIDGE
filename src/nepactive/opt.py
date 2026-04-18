from __future__ import annotations

import argparse
from pathlib import Path

from ase import Atoms
from ase.filters import UnitCellFilter
from ase.io import read, write
from ase.optimize import LBFGS

from nepactive.nep_backend import create_ase_calculator


def _ensure_atoms_list(images) -> list[Atoms]:
    if isinstance(images, list):
        return images
    return [images]


def _default_output_path(structure: str) -> str:
    source = Path(structure)
    if source.suffix:
        return str(source.with_name(f"{source.stem}_opt.xyz"))
    return str(source.with_name(f"{source.name}_opt.xyz"))


def _infer_write_format(output_path: str, frame_count: int) -> str | None:
    suffix = Path(output_path).suffix.lower()
    if suffix == ".traj":
        return "traj"
    if suffix == ".xyz":
        return "extxyz"
    if suffix in {".vasp", ".poscar", ".contcar"}:
        if frame_count != 1:
            raise ValueError("VASP output supports only a single frame. Use --index to select one frame.")
        return "vasp"
    return None


def _build_auxiliary_path(path: str | None, frame_index: int, frame_count: int) -> str | None:
    if not path:
        return None
    if path == "-":
        return path
    if frame_count == 1:
        return path
    target = Path(path)
    suffix = target.suffix
    stem = target.stem if suffix else target.name
    parent = target.parent
    frame_name = f"{stem}.frame{frame_index:06d}"
    if suffix:
        frame_name += suffix
    return str(parent / frame_name)


def optimize_structures(
    structure: str,
    index: str = "0",
    output: str | None = None,
    fmax: float = 0.05,
    steps: int = 200,
    device: str = "cuda",
    optimize_cell: bool = False,
    hydrostatic_strain: bool = False,
    trajectory: str | None = None,
    logfile: str | None = None,
) -> tuple[str, list[dict[str, float | int | bool]]]:
    atoms_list = _ensure_atoms_list(read(structure, index=index))
    if not atoms_list:
        raise ValueError(f"No frames found in {structure} with index={index}")

    calculator = create_ase_calculator(
        model_name="mattersim",
        model_file=None,
        device=device,
        nep_backend="gpu",
    )

    optimized_atoms: list[Atoms] = []
    summaries: list[dict[str, float | int | bool]] = []
    frame_count = len(atoms_list)
    for frame_index, atoms in enumerate(atoms_list):
        optimized = atoms.copy()
        optimized.calc = calculator
        target = optimized
        if optimize_cell:
            target = UnitCellFilter(optimized, hydrostatic_strain=hydrostatic_strain)
        optimizer = LBFGS(
            target,
            trajectory=_build_auxiliary_path(trajectory, frame_index, frame_count),
            logfile=_build_auxiliary_path(logfile, frame_index, frame_count),
        )
        converged = bool(optimizer.run(fmax=float(fmax), steps=int(steps)))
        energy = float(optimized.get_potential_energy())
        max_force = float((optimized.get_forces() ** 2).sum(axis=1).max() ** 0.5)
        optimized.info["mattersim_energy"] = energy
        optimized.info["mattersim_fmax"] = max_force
        optimized.info["mattersim_converged"] = converged
        optimized_atoms.append(optimized)
        summaries.append(
            {
                "frame": frame_index,
                "energy": energy,
                "fmax": max_force,
                "converged": converged,
            }
        )

    output_path = output or _default_output_path(structure)
    write_format = _infer_write_format(output_path, len(optimized_atoms))
    if write_format is None:
        write(output_path, optimized_atoms)
    elif write_format == "vasp":
        write(output_path, optimized_atoms[0], format=write_format)
    else:
        write(output_path, optimized_atoms, format=write_format)
    return output_path, summaries


def cli(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(
        description="Optimize structures with MatterSim + ASE LBFGS.",
    )
    parser.add_argument("structure", help="Input structure/trajectory file")
    parser.add_argument(
        "--index",
        default="0",
        help="ASE index expression, default '0'. Use ':' to optimize all frames.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename, default '<input>_opt.xyz'",
    )
    parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence threshold in eV/A")
    parser.add_argument("--steps", type=int, default=200, help="Maximum optimizer steps")
    parser.add_argument("--device", default="cuda", help="MatterSim device, e.g. cuda or cpu")
    parser.add_argument(
        "--cell",
        action="store_true",
        help="Optimize the cell together with atomic positions using UnitCellFilter",
    )
    parser.add_argument(
        "--hydrostatic",
        action="store_true",
        help="Use hydrostatic strain when --cell is enabled",
    )
    parser.add_argument(
        "--traj",
        default=None,
        help="Optional ASE trajectory output path; multi-frame runs append .frameXXXXXX",
    )
    parser.add_argument(
        "--log",
        default="-",
        help="Optimizer log path; use '-' to print to stdout, empty to disable",
    )
    args = parser.parse_args(argv)

    logfile = args.log
    if logfile == "":
        logfile = None

    output_path, summaries = optimize_structures(
        structure=args.structure,
        index=args.index,
        output=args.output,
        fmax=args.fmax,
        steps=args.steps,
        device=args.device,
        optimize_cell=args.cell,
        hydrostatic_strain=args.hydrostatic,
        trajectory=args.traj,
        logfile=logfile,
    )
    print(f"Saved optimized structure(s) to {output_path}")
    for item in summaries:
        print(
            f"frame={item['frame']} converged={item['converged']} "
            f"energy={item['energy']:.8f} fmax={item['fmax']:.8f}"
        )
    return output_path


if __name__ == "__main__":
    cli()
