from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms

from nepactive import dlog
from nepactive.nep_backend import NativeNepCalculator


def numpy_cdist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    squared_dist = np.sum(np.square(diff), axis=2)
    return np.sqrt(squared_dist).astype(np.float32, copy=False)


def farthest_point_sampling(
    points: np.ndarray,
    n_samples: int,
    min_dist: float = 0.0,
    selected_data: Optional[np.ndarray] = None,
) -> list[int]:
    if points.size == 0 or n_samples <= 0:
        return []

    n_points = int(points.shape[0])
    n_samples = min(int(n_samples), n_points)
    sampled_indices: list[int] = []

    if isinstance(selected_data, np.ndarray) and selected_data.size != 0:
        distances_to_samples = numpy_cdist(points, selected_data)
        min_distances = np.min(distances_to_samples, axis=1)
    else:
        first_index = 0
        sampled_indices.append(first_index)
        min_distances = np.linalg.norm(points - points[first_index], axis=1)

    while len(sampled_indices) < n_samples:
        current_index = int(np.argmax(min_distances))
        if float(min_distances[current_index]) < float(min_dist):
            break
        if current_index in sampled_indices:
            break
        sampled_indices.append(current_index)
        new_point = points[current_index]
        new_distances = np.linalg.norm(points - new_point, axis=1)
        min_distances = np.minimum(min_distances, new_distances)

    if len(sampled_indices) < n_samples:
        for index in range(n_points):
            if index not in sampled_indices:
                sampled_indices.append(index)
            if len(sampled_indices) >= n_samples:
                break
    return sampled_indices


def _guess_cutoff(atoms: Atoms) -> float:
    if atoms.cell.rank == 3 and np.any(atoms.pbc):
        lengths = atoms.cell.lengths()
        positive_lengths = [value for value in lengths if value > 1e-8]
        if positive_lengths:
            return max(2.0, min(8.0, min(positive_lengths) * 0.5))
    return 8.0


def structural_fingerprint(atoms: Atoms, bins: int = 64, cutoff: Optional[float] = None) -> np.ndarray:
    natoms = len(atoms)
    if natoms == 0:
        return np.zeros(bins + 12, dtype=np.float32)

    cutoff = float(cutoff or _guess_cutoff(atoms))
    distances = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))
    upper = distances[np.triu_indices(natoms, k=1)]
    hist, _ = np.histogram(upper, bins=bins, range=(0.0, cutoff), density=False)
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()

    volume = float(atoms.get_volume()) if atoms.cell.rank == 3 else 0.0
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.float32)
    masses = np.asarray(atoms.get_masses(), dtype=np.float32)
    cellpar = atoms.cell.cellpar() if atoms.cell.rank == 3 else np.zeros(6, dtype=np.float32)
    density = float(masses.sum() / volume) if volume > 1e-12 else 0.0

    extras = np.asarray(
        [
            natoms,
            volume / max(natoms, 1),
            density,
            float(np.mean(numbers)),
            float(np.std(numbers)),
            float(np.min(numbers)),
            float(np.max(numbers)),
            float(np.mean(upper)) if upper.size else 0.0,
            float(np.std(upper)) if upper.size else 0.0,
            float(cellpar[0]),
            float(cellpar[1]),
            float(cellpar[2]),
        ],
        dtype=np.float32,
    )
    return np.concatenate([hist, extras]).astype(np.float32, copy=False)


def soap_fingerprint(
    atoms_list: list[Atoms],
    r_cut: float = 5.0,
    n_max: int = 8,
    l_max: int = 6,
    sigma: float = 0.5,
) -> np.ndarray:
    try:
        from dscribe.descriptors import SOAP
    except Exception as exc:
        raise ModuleNotFoundError("SOAP sampling requires dscribe to be installed") from exc

    species = sorted({symbol for atoms in atoms_list for symbol in atoms.get_chemical_symbols()})
    if not species:
        return np.array([], dtype=np.float32)

    periodic = any(bool(np.any(atoms.pbc)) for atoms in atoms_list)
    soap = SOAP(
        species=species,
        r_cut=float(r_cut),
        n_max=int(n_max),
        l_max=int(l_max),
        sigma=float(sigma),
        periodic=periodic,
        sparse=False,
    )

    descriptors = []
    for atoms in atoms_list:
        atom_descriptor = soap.create(atoms)
        atom_descriptor = np.asarray(atom_descriptor, dtype=np.float32)
        if atom_descriptor.ndim == 1:
            descriptors.append(atom_descriptor)
        else:
            descriptors.append(np.mean(atom_descriptor, axis=0, dtype=np.float32))
    return np.asarray(descriptors, dtype=np.float32)


def _resolve_descriptor_model(descriptor_model: str | None) -> str | None:
    if not descriptor_model:
        return None
    path = Path(descriptor_model).expanduser()
    if path.exists():
        return str(path.resolve())
    return None


def compute_structure_descriptors(
    atoms_list: list[Atoms],
    descriptor_model: str | None = None,
    descriptor_mode: str = "auto",
    nep_backend: str = "auto",
) -> tuple[np.ndarray, str]:
    if not atoms_list:
        return np.array([], dtype=np.float32), "empty"

    resolved_model = _resolve_descriptor_model(descriptor_model)
    mode = str(descriptor_mode or "auto").lower()
    use_nep_descriptor = mode == "nep" or (mode == "auto" and resolved_model is not None)
    if use_nep_descriptor:
        if resolved_model is None:
            raise FileNotFoundError(
                "dataset_sampling_nep_file is required when dataset_sampling_descriptor=nep"
            )
        try:
            calculator = NativeNepCalculator(resolved_model, backend=nep_backend)
            descriptors = calculator.get_structures_descriptor(atoms_list)
            if descriptors.size == 0:
                raise RuntimeError(f"Failed to compute NEP descriptors from {resolved_model}")
            return np.asarray(descriptors, dtype=np.float32), "nep"
        except Exception:
            if mode == "nep":
                raise
            dlog.warning("NEP descriptor backend unavailable, fallback to structural fingerprint FPS")

    if mode == "soap":
        return soap_fingerprint(atoms_list), "soap"

    if mode not in {"auto", "structural"}:
        raise ValueError(f"Unsupported dataset sampling descriptor mode: {descriptor_mode}")

    fingerprints = [structural_fingerprint(atoms) for atoms in atoms_list]
    return np.asarray(fingerprints, dtype=np.float32), "structural"


def select_structure_indices(
    atoms_list: list[Atoms],
    n_samples: int,
    method: str = "fps",
    descriptor_model: str | None = None,
    descriptor_mode: str = "auto",
    min_dist: float = 0.0,
    nep_backend: str = "auto",
    reference_atoms_list: Optional[list[Atoms]] = None,
) -> list[int]:
    total = len(atoms_list)
    if n_samples <= 0 or total == 0:
        return []
    if n_samples >= total:
        return list(range(total))

    method = str(method or "fps").lower()
    if method == "random":
        return list(range(n_samples))
    if method != "fps":
        raise ValueError(f"Unsupported structure sampling method: {method}")

    reference_atoms = [atoms for atoms in (reference_atoms_list or []) if atoms is not None]
    if reference_atoms:
        combined_atoms = reference_atoms + list(atoms_list)
        combined_descriptors, source = compute_structure_descriptors(
            combined_atoms,
            descriptor_model=descriptor_model,
            descriptor_mode=descriptor_mode,
            nep_backend=nep_backend,
        )
        ref_count = len(reference_atoms)
        reference_descriptors = np.asarray(combined_descriptors[:ref_count], dtype=np.float32)
        descriptors = np.asarray(combined_descriptors[ref_count:], dtype=np.float32)
    else:
        descriptors, source = compute_structure_descriptors(
            atoms_list,
            descriptor_model=descriptor_model,
            descriptor_mode=descriptor_mode,
            nep_backend=nep_backend,
        )
        reference_descriptors = None

    indices = farthest_point_sampling(
        descriptors,
        n_samples=n_samples,
        min_dist=min_dist,
        selected_data=reference_descriptors,
    )
    dlog.info(
        "Selected %d/%d structures with %s FPS against %d reference structures",
        len(indices),
        total,
        source,
        len(reference_atoms),
    )
    return sorted(indices)


def split_train_test_structures(
    atoms_list: list[Atoms],
    training_ratio: float = 0.8,
    method: str = "fps",
    descriptor_model: str | None = None,
    descriptor_mode: str = "auto",
    min_dist: float = 0.0,
    nep_backend: str = "auto",
) -> tuple[list[Atoms], list[Atoms]]:
    total = len(atoms_list)
    if total == 0:
        return [], []

    if total == 1:
        return list(atoms_list), []

    n_train = int(round(total * float(training_ratio)))
    n_train = min(max(n_train, 1), total)
    n_test = total - n_train
    if n_test <= 0:
        return list(atoms_list), []

    test_indices = set(
        select_structure_indices(
            atoms_list,
            n_samples=n_test,
            method=method,
            descriptor_model=descriptor_model,
            descriptor_mode=descriptor_mode,
            min_dist=min_dist,
            nep_backend=nep_backend,
        )
    )
    train = [atoms for index, atoms in enumerate(atoms_list) if index not in test_indices]
    test = [atoms for index, atoms in enumerate(atoms_list) if index in test_indices]
    return train, test
