"""POSCAR preparation tests; no potentials or external jobs are executed."""

import importlib
import sys
import types

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write

from nepactive import supercell


def make_poscar(tmp_path, count=8):
    atoms = Atoms("H" * count, cell=[3, 4, 5], pbc=True)
    atoms.set_scaled_positions(np.arange(count * 3).reshape(count, 3) / (count * 3))
    path = tmp_path / "POSCAR"
    write(path, atoms, format="vasp", direct=True)
    return path, atoms


@pytest.mark.parametrize("count", range(1, 100))
def test_small_structures_always_reach_target(count):
    atoms = Atoms("H" * count, cell=[3, 5, 8], pbc=True)
    repeats = supercell.choose_repeats(atoms)
    assert all(isinstance(n, int) and n >= 1 for n in repeats)
    assert 100 <= count * np.prod(repeats) <= 200


def test_balanced_cell_prefers_short_axis():
    atoms = Atoms("H50", cell=[1, 2, 2], pbc=True)
    assert supercell.choose_repeats(atoms) == (2, 1, 1)


@pytest.mark.parametrize("count", [100, 150, 200, 201, 250])
def test_large_structures_are_not_rewritten(tmp_path, count):
    path, _ = make_poscar(tmp_path, count)
    original = path.read_bytes()
    assert supercell.prepare_poscar(tmp_path) == (1, 1, 1)
    assert path.read_bytes() == original
    assert not (tmp_path / "mv.vasp").exists()


def test_backup_and_repeat_are_lossless_and_idempotent(tmp_path):
    path, atoms = make_poscar(tmp_path)
    original = path.read_bytes()
    repeats = supercell.prepare_poscar(tmp_path)
    result = read(path, format="vasp")
    expected = atoms.repeat(repeats)
    assert 100 <= len(result) <= 200
    np.testing.assert_allclose(result.cell, expected.cell)
    np.testing.assert_allclose(result.positions, expected.positions, atol=1e-12)
    assert result.get_volume() / len(result) == pytest.approx(atoms.get_volume() / len(atoms))
    assert (tmp_path / "mv.vasp").read_bytes() == original
    expanded_bytes = path.read_bytes()
    supercell.prepare_poscar(tmp_path)
    assert path.read_bytes() == expanded_bytes
    assert (tmp_path / "mv.vasp").read_bytes() == original


def test_triclinic_species_order_and_constraints_survive(tmp_path):
    atoms = Atoms("OH", scaled_positions=[[0, 0, 0], [.2, .3, .4]],
                  cell=[[3, 0, 0], [1, 4, 0], [.5, .5, 5]], pbc=True)
    atoms.set_constraint(FixAtoms(indices=[0]))
    write(tmp_path / "POSCAR", atoms, format="vasp", direct=True)
    repeats = supercell.prepare_poscar(tmp_path)
    result = read(tmp_path / "POSCAR", format="vasp")
    copies = int(np.prod(repeats))
    assert result.get_chemical_symbols() == ["O"] * copies + ["H"] * copies
    np.testing.assert_allclose(result.cell, atoms.repeat(repeats).cell)
    np.testing.assert_array_equal(result.constraints[0].index, np.arange(copies))


@pytest.mark.parametrize("skip", ["disabled", "resume", "missing"])
def test_skip_conditions(tmp_path, skip):
    path, _ = make_poscar(tmp_path)
    original = path.read_bytes()
    if skip == "resume":
        (tmp_path / "record.nep").touch()
    if skip == "missing":
        path.unlink()
    assert supercell.prepare_poscar(tmp_path, enabled=skip != "disabled") == (1, 1, 1)
    assert not (tmp_path / "mv.vasp").exists()
    if skip != "missing":
        assert path.read_bytes() == original


@pytest.mark.parametrize("enabled", ["False", "True", None, 0, 1])
def test_invalid_config_rejected(tmp_path, enabled):
    with pytest.raises(ValueError, match="YAML boolean"):
        supercell.prepare_poscar(tmp_path, enabled)


@pytest.mark.parametrize("kind", ["empty", "zero", "singular", "nan"])
def test_invalid_structures_rejected(kind):
    atoms = Atoms("H", cell=[1, 1, 1])
    if kind == "empty":
        atoms = Atoms(cell=[1, 1, 1])
    elif kind == "zero":
        atoms.set_cell([0, 0, 0])
    elif kind == "singular":
        atoms.set_cell([[1, 0, 0], [2, 0, 0], [0, 0, 1]])
    else:
        atoms.set_cell([np.nan, 1, 1])
    with pytest.raises(ValueError, match="auto_supercell"):
        supercell.choose_repeats(atoms)


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_backup_is_never_overwritten(tmp_path, symlink):
    path, _ = make_poscar(tmp_path)
    original = path.read_bytes()
    backup = tmp_path / "mv.vasp"
    if symlink:
        backup.symlink_to(tmp_path / "missing-target")
    else:
        backup.write_bytes(b"prior original")
    with pytest.raises(FileExistsError, match="backup already exists"):
        supercell.prepare_poscar(tmp_path)
    assert path.read_bytes() == original
    assert backup.is_symlink() if symlink else backup.read_bytes() == b"prior original"


@pytest.mark.parametrize("stage", ["write", "replace"])
def test_io_failure_preserves_original_and_cleans_temporary(monkeypatch, tmp_path, stage):
    path, _ = make_poscar(tmp_path)
    original = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("simulated I/O failure")
    if stage == "write":
        monkeypatch.setattr(supercell, "write", fail)
    else:
        monkeypatch.setattr(supercell.os, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        supercell.prepare_poscar(tmp_path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".POSCAR.supercell-*"))
    if stage == "replace":
        assert (tmp_path / "mv.vasp").read_bytes() == original
    else:
        assert not (tmp_path / "mv.vasp").exists()


@pytest.mark.parametrize("shock_run", [False, True])
@pytest.mark.parametrize("enabled", [None, False, True])
def test_workflow_startup_uses_config(monkeypatch, tmp_path, shock_run, enabled):
    # Only model construction is optional for this startup integration check.
    if "mattersim.forcefield" not in sys.modules:
        module = types.ModuleType("mattersim.forcefield")
        module.MatterSimCalculator = type("MatterSimCalculator", (), {})
        monkeypatch.setitem(sys.modules, "mattersim.forcefield", module)
    train = importlib.import_module("nepactive.train")
    monkeypatch.chdir(tmp_path)
    path, _ = make_poscar(tmp_path)
    config = {"shock_run": shock_run, "structure_files": ["POSCAR"],
              "sampling": {"general": {"ensembles": ["nvt"]}}}
    if enabled is not None:
        config["auto_supercell"] = enabled
    train.Nepactive(config)
    result = read(path, format="vasp")
    if enabled is False:
        assert len(result) == 8
        assert not (tmp_path / "mv.vasp").exists()
    else:
        assert 100 <= len(result) <= 200
        assert (tmp_path / "mv.vasp").exists()
