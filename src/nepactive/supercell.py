"""Prepare a bounded-size POSCAR before starting a new workflow."""

import os
import shutil
import tempfile
from math import ceil
from pathlib import Path

import numpy as np
from ase.io import read, write

from nepactive import dlog


def choose_repeats(atoms):
    """Choose integer repeats giving 100–200 atoms with balanced cell lengths."""
    count = len(atoms)
    if count == 0:
        raise ValueError("auto_supercell requires a non-empty structure")
    if count >= 100:
        return (1, 1, 1)
    lengths = atoms.cell.lengths()
    if not np.isfinite(atoms.cell.array).all() or atoms.cell.rank != 3 or atoms.get_volume() <= 0:
        raise ValueError("auto_supercell requires a finite, non-singular 3D cell")
    lower, upper = ceil(100 / count), 200 // count
    candidates = []
    for a in range(1, upper + 1):
        for b in range(1, upper // a + 1):
            for c in range(max(1, ceil(lower / (a * b))), upper // (a * b) + 1):
                expanded = lengths * (a, b, c)
                candidates.append((float(expanded.max() / expanded.min()), a * b * c, (a, b, c)))
    return min(candidates)[2]


def prepare_poscar(work_dir, enabled=True):
    """Back up POSCAR byte-for-byte to mv.vasp and atomically replace it.

    Missing POSCARs and restarted workflows are left alone. An existing backup
    is never overwritten. Return the replication factors (identity if skipped).
    """
    if not isinstance(enabled, bool):
        raise ValueError("auto_supercell must be a YAML boolean (true or false)")
    root = Path(work_dir)
    poscar = root / "POSCAR"
    if not enabled or (root / "record.nep").exists() or not poscar.exists():
        return (1, 1, 1)
    atoms = read(poscar, format="vasp")
    repeats = choose_repeats(atoms)
    if repeats == (1, 1, 1):
        if len(atoms) > 200:
            dlog.warning("auto_supercell: POSCAR has %d atoms (>200); keeping it unchanged", len(atoms))
        return repeats
    backup = root / "mv.vasp"
    if os.path.lexists(backup):
        raise FileExistsError(f"auto_supercell: backup already exists: {backup}; archive it or set auto_supercell: false")
    expanded = atoms.repeat(repeats)
    # Group species for a conventional POSCAR, retaining the original species
    # order so an existing POTCAR remains compatible.
    species = list(dict.fromkeys(atoms.get_chemical_symbols()))
    order = {symbol: index for index, symbol in enumerate(species)}
    expanded = expanded[sorted(range(len(expanded)), key=lambda i: order[expanded[i].symbol])]
    fd, temporary = tempfile.mkstemp(prefix=".POSCAR.supercell-", dir=root)
    os.close(fd)
    try:
        write(temporary, expanded, format="vasp", direct=True, vasp5=True)
        shutil.copymode(poscar, temporary)
        with backup.open("xb") as output, poscar.open("rb") as source:
            shutil.copyfileobj(source, output)
        os.replace(temporary, poscar)
    finally:
        Path(temporary).unlink(missing_ok=True)
    dlog.info("auto_supercell: %d -> %d atoms, repeats=%s; original saved to %s", len(atoms), len(expanded), repeats, backup)
    return repeats
