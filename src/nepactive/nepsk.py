from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ase import Atoms
from ase.io import read, write
from ase.io.formats import ioformats


def _read_images(structure: str, index: str):
    try:
        return read(structure, index=index)
    except Exception as exc:
        suffix = Path(structure).suffix.lower()
        if suffix != ".xyz":
            raise
        try:
            return read(structure, index=index, format="xyz")
        except Exception:
            raise exc


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


def _resolve_output_format(output_format: str) -> str:
    normalized = str(output_format).strip().lower()
    matches = [
        name
        for name, fmt in ioformats.items()
        if normalized in {ext.lower() for ext in (getattr(fmt, "extensions", None) or [])}
    ]
    if len(matches) == 1:
        return matches[0]
    if normalized in ioformats:
        return normalized
    if not matches:
        raise ValueError(f"Unsupported ASE output format: {output_format}")
    raise ValueError(
        f"Ambiguous output format '{output_format}', matches ASE formats: {', '.join(sorted(matches))}"
    )


def _default_output_suffix(requested_format: str, resolved_format: str) -> str:
    normalized = str(requested_format).strip().lower()
    if normalized in {"xyz", "pdb", "vasp", "traj", "cif", "json"}:
        return f".{normalized}"
    fmt = ioformats.get(resolved_format)
    extensions = getattr(fmt, "extensions", None) or []
    if extensions:
        return f".{extensions[0]}"
    return f".{normalized}"


def _default_output_filename(structure: str, output_format: str, resolved_format: str) -> str:
    source = Path(structure)
    return str(source.with_suffix(_default_output_suffix(output_format, resolved_format)))


def convert_structure(
    structure: str,
    output_format: str,
    index: str = ":",
    duplicate: tuple[int, int, int] | None = None,
    output_filename: str | None = None,
) -> str:
    resolved_format = _resolve_output_format(output_format)

    images = _ensure_list(_read_images(structure, index=index))
    converted: list[Atoms] = []
    for atoms in images:
        new_atoms = atoms.copy()
        if duplicate is not None:
            new_atoms = new_atoms.repeat(duplicate)
        converted.append(_sort_atoms(new_atoms))

    output_path = output_filename or _default_output_filename(structure, output_format, resolved_format)
    if resolved_format == "vasp":
        if len(converted) != 1:
            raise ValueError("VASP output supports only a single frame. Use --index to select one frame.")
        write(output_path, converted[0], format="vasp")
    elif resolved_format == "extxyz":
        write(output_path, converted, format="extxyz")
    else:
        write(output_path, converted, format=resolved_format)
    return output_path


def cli(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(
        description="Slice structures/trajectories, optionally duplicate the cell, and convert format.",
    )
    parser.add_argument("structure", help="Input structure/trajectory file")
    parser.add_argument("format", help="ASE output format name or ASE-known extension, e.g. xyz, pdb, vasp, cif, traj")
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
