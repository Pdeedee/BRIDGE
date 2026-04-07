from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ase import Atoms
from ase.io import read, write


FORMAT_TO_SUFFIX = {
    "vasp": ".vasp",
    "xyz": ".xyz",
    "traj": ".traj",
}


def _normalize_argv(argv: list[str]) -> list[str]:
    token_map = {
        "index": "--index",
        "duplicate": "--duplicate",
        "output_filename": "--output",
        "output": "--output",
    }
    normalized: list[str] = []
    ii = 0
    while ii < len(argv):
        token = token_map.get(argv[ii], argv[ii])
        normalized.append(token)
        if token == "--duplicate" and ii + 1 < len(argv):
            next_token = argv[ii + 1].strip().strip("()[]")
            duplicate_parts = next_token.replace(",", " ").split()
            if len(duplicate_parts) == 3:
                normalized.extend(duplicate_parts)
                ii += 2
                continue
        ii += 1
    return normalized


def _parse_duplicate(values: Iterable[str] | None) -> tuple[int, int, int] | None:
    if not values:
        return None

    tokens = [value.strip() for value in values if value.strip()]
    if len(tokens) != 3:
        raise ValueError("duplicate must provide exactly three integers, e.g. 2,2,2")

    duplicate = tuple(int(token) for token in tokens)
    if any(multiplier <= 0 for multiplier in duplicate):
        raise ValueError("duplicate values must be positive integers")
    return duplicate


def _ensure_list(images) -> list[Atoms]:
    if isinstance(images, list):
        return images
    return [images]


def _sort_atoms(atoms: Atoms) -> Atoms:
    symbols = atoms.get_chemical_symbols()
    order = sorted(range(len(symbols)), key=lambda index: (symbols[index], index))
    return atoms[order]


def _default_output_filename(structure: str, output_format: str) -> str:
    source = Path(structure)
    return str(source.with_suffix(FORMAT_TO_SUFFIX[output_format]))


def convert_structure(
    structure: str,
    output_format: str,
    index: str = ":",
    duplicate: tuple[int, int, int] | None = None,
    output_filename: str | None = None,
) -> str:
    if output_format not in FORMAT_TO_SUFFIX:
        raise ValueError(f"Unsupported output format: {output_format}")

    images = _ensure_list(read(structure, index=index))
    converted: list[Atoms] = []
    for atoms in images:
        new_atoms = atoms.copy()
        if duplicate is not None:
            new_atoms = new_atoms.repeat(duplicate)
        converted.append(_sort_atoms(new_atoms))

    output_path = output_filename or _default_output_filename(structure, output_format)
    if output_format == "vasp":
        if len(converted) != 1:
            raise ValueError("VASP output supports only a single frame. Use --index to select one frame.")
        write(output_path, converted[0], format="vasp")
    elif output_format == "xyz":
        write(output_path, converted, format="extxyz")
    else:
        write(output_path, converted, format=output_format)
    return output_path


def cli(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(
        description="Slice structures/trajectories, optionally duplicate the cell, and convert format.",
    )
    parser.add_argument("structure", help="Input structure/trajectory file")
    parser.add_argument("format", choices=sorted(FORMAT_TO_SUFFIX), help="Output format")
    parser.add_argument(
        "--index",
        default=":",
        help="ASE index expression, e.g. ':', '0', '-1', '1:' (default: ':')",
    )
    parser.add_argument(
        "--duplicate",
        nargs=3,
        default=None,
        help="Cell repetition, e.g. '2 2 2' or '2,2,2'",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename (default: input prefix + format suffix)",
    )

    normalized_argv = _normalize_argv(list(argv) if argv is not None else [])
    parse_method = getattr(parser, "parse_intermixed_args", parser.parse_args)
    args = parse_method(normalized_argv or None)
    duplicate = _parse_duplicate(args.duplicate)
    output_path = convert_structure(
        structure=args.structure,
        output_format=args.format,
        index=args.index,
        duplicate=duplicate,
        output_filename=args.output,
    )
    print(f"Saved {output_path}")
    return output_path


if __name__ == "__main__":
    cli()
