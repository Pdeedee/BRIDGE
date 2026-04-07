from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.io import read, write

from nepactive import dlog
from nepactive.native_guard import get_structures_descriptor_with_gpu_guard
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


def incremental_fps_with_r2(
    points: np.ndarray,
    n_samples: int,
    r2_threshold: float,
    min_dist: float = 0.0,
    selected_data: Optional[np.ndarray] = None,
) -> tuple[list[int], float]:
    if points.size == 0 or n_samples <= 0:
        return [], 0.0

    n_points = int(points.shape[0])
    n_samples = min(int(n_samples), n_points)
    sampled_indices: list[int] = []

    overall_mean = np.mean(points, axis=0)
    total_variance = float(np.sum((points - overall_mean) ** 2))

    if isinstance(selected_data, np.ndarray) and selected_data.size != 0:
        distances_to_samples = numpy_cdist(points, selected_data)
        min_distances = np.min(distances_to_samples, axis=1)
    else:
        first_index = 0
        sampled_indices.append(first_index)
        min_distances = np.linalg.norm(points - points[first_index], axis=1)

    def current_r2() -> float:
        if total_variance <= 0.0:
            return 1.0
        if not sampled_indices:
            return 0.0
        explained_variance = float(np.sum((points[sampled_indices] - overall_mean) ** 2))
        return explained_variance / total_variance

    r2 = current_r2()
    if r2 >= float(r2_threshold) or len(sampled_indices) >= n_samples:
        return sampled_indices, r2

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
        r2 = current_r2()
        if r2 >= float(r2_threshold):
            break

    return sampled_indices, r2


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


def _resolve_descriptor_model(descriptor_model: Optional[str]) -> Optional[str]:
    if not descriptor_model:
        return None
    path = Path(descriptor_model).expanduser()
    if path.exists():
        return str(path.resolve())
    return None


def _resolve_plot_subset(count: int, max_points: Optional[int]) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=int)
    if max_points is None:
        return np.arange(count, dtype=int)
    max_points = int(max_points)
    if max_points <= 0 or count <= max_points:
        return np.arange(count, dtype=int)
    return np.linspace(0, count - 1, num=max_points, dtype=int)


def _compute_pca_2d(descriptors: np.ndarray) -> np.ndarray:
    array = np.asarray(descriptors, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)

    centered = array - np.mean(array, axis=0, keepdims=True)
    coords = np.zeros((centered.shape[0], 2), dtype=np.float32)
    if centered.shape[1] == 0 or centered.shape[0] == 0:
        return coords
    if centered.shape[1] == 1:
        coords[:, 0] = centered[:, 0]
        return coords

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[: min(2, vt.shape[0])].T
    projected = centered @ basis
    coords[:, : projected.shape[1]] = projected
    return coords


def write_fps_pca_plot(
    descriptors: np.ndarray,
    selected_indices: list[int],
    output_path: str | Path,
    reference_descriptors: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    max_points: Optional[int] = None,
) -> str:
    candidate = np.asarray(descriptors, dtype=np.float32)
    if candidate.ndim == 1:
        candidate = candidate.reshape(1, -1)
    reference = None
    if isinstance(reference_descriptors, np.ndarray) and reference_descriptors.size != 0:
        reference = np.asarray(reference_descriptors, dtype=np.float32)
        if reference.ndim == 1:
            reference = reference.reshape(1, -1)

    combined = candidate if reference is None else np.vstack([reference, candidate])
    coords = _compute_pca_2d(combined)
    ref_count = 0 if reference is None else int(reference.shape[0])
    candidate_coords = coords[ref_count:]
    reference_coords = coords[:ref_count]
    selected = np.asarray(sorted({int(index) for index in selected_indices if 0 <= int(index) < len(candidate)}), dtype=int)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    if ref_count > 0:
        ref_plot_idx = _resolve_plot_subset(ref_count, max_points)
        ax.scatter(
            reference_coords[ref_plot_idx, 0],
            reference_coords[ref_plot_idx, 1],
            s=4,
            alpha=0.15,
            label=f"reference ({ref_count})",
        )

    candidate_plot_idx = _resolve_plot_subset(len(candidate_coords), max_points)
    if candidate_plot_idx.size:
        ax.scatter(
            candidate_coords[candidate_plot_idx, 0],
            candidate_coords[candidate_plot_idx, 1],
            s=8,
            alpha=0.35,
            label=f"candidate ({len(candidate_coords)})",
        )
    if selected.size:
        ax.scatter(
            candidate_coords[selected, 0],
            candidate_coords[selected, 1],
            s=14,
            alpha=0.9,
            label=f"selected ({len(selected)})",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title or "FPS Coverage PCA")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return str(output.resolve())


def compute_structure_descriptors(
    atoms_list: list[Atoms],
    descriptor_model: Optional[str] = None,
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
            if str(nep_backend or "auto").lower() in {"auto", "native", "gpu"}:
                descriptors, backend_used = get_structures_descriptor_with_gpu_guard(
                    atoms_list,
                    resolved_model,
                    backend=nep_backend,
                )
                dlog.info("Computed NEP descriptors with %s backend", backend_used)
            else:
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


def select_structure_indices_with_info(
    atoms_list: list[Atoms],
    n_samples: int,
    method: str = "fps",
    descriptor_model: Optional[str] = None,
    descriptor_mode: str = "auto",
    min_dist: float = 0.0,
    nep_backend: str = "auto",
    reference_atoms_list: Optional[list[Atoms]] = None,
    r2_threshold: Optional[float] = None,
    pca_plot_path: Optional[str] = None,
    pca_plot_title: Optional[str] = None,
    pca_plot_max_points: Optional[int] = None,
) -> tuple[list[int], dict]:
    total = len(atoms_list)
    info = {
        "source": "empty",
        "reference_count": 0,
        "r2": None,
        "r2_threshold": r2_threshold,
        "plot_path": None,
    }
    if n_samples <= 0 or total == 0:
        return [], info
    if n_samples >= total:
        return list(range(total)), info

    method = str(method or "fps").lower()
    if method == "random":
        info["source"] = "random"
        return list(range(n_samples)), info
    if method != "fps":
        raise ValueError(f"Unsupported structure sampling method: {method}")

    reference_atoms = [atoms for atoms in (reference_atoms_list or []) if atoms is not None]
    info["reference_count"] = len(reference_atoms)
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
    info["source"] = source

    if r2_threshold is not None:
        indices, r2 = incremental_fps_with_r2(
            descriptors,
            n_samples=n_samples,
            r2_threshold=float(r2_threshold),
            min_dist=min_dist,
            selected_data=reference_descriptors,
        )
        info["r2"] = float(r2)
        dlog.info(
            "Selected %d/%d structures with %s FPS against %d reference structures (R2=%.6f, threshold=%s)",
            len(indices),
            total,
            source,
            len(reference_atoms),
            float(r2),
            r2_threshold,
        )
    else:
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
    if pca_plot_path:
        try:
            info["plot_path"] = write_fps_pca_plot(
                descriptors,
                indices,
                output_path=pca_plot_path,
                reference_descriptors=reference_descriptors,
                title=pca_plot_title,
                max_points=pca_plot_max_points,
            )
        except Exception as exc:
            dlog.warning("Failed to write FPS PCA plot to %s: %s", pca_plot_path, exc)
    return sorted(indices), info


def select_structure_indices(
    atoms_list: list[Atoms],
    n_samples: int,
    method: str = "fps",
    descriptor_model: Optional[str] = None,
    descriptor_mode: str = "auto",
    min_dist: float = 0.0,
    nep_backend: str = "auto",
    reference_atoms_list: Optional[list[Atoms]] = None,
    r2_threshold: Optional[float] = None,
    pca_plot_path: Optional[str] = None,
    pca_plot_title: Optional[str] = None,
    pca_plot_max_points: Optional[int] = None,
) -> list[int]:
    indices, _ = select_structure_indices_with_info(
        atoms_list,
        n_samples=n_samples,
        method=method,
        descriptor_model=descriptor_model,
        descriptor_mode=descriptor_mode,
        min_dist=min_dist,
        nep_backend=nep_backend,
        reference_atoms_list=reference_atoms_list,
        r2_threshold=r2_threshold,
        pca_plot_path=pca_plot_path,
        pca_plot_title=pca_plot_title,
        pca_plot_max_points=pca_plot_max_points,
    )
    return indices


def split_train_test_structures(
    atoms_list: list[Atoms],
    training_ratio: float = 0.8,
    method: str = "fps",
    descriptor_model: Optional[str] = None,
    descriptor_mode: str = "auto",
    min_dist: float = 0.0,
    nep_backend: str = "auto",
    pca_plot_path: Optional[str] = None,
    pca_plot_title: Optional[str] = None,
    pca_plot_max_points: Optional[int] = None,
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
            pca_plot_path=pca_plot_path,
            pca_plot_title=pca_plot_title,
            pca_plot_max_points=pca_plot_max_points,
        )
    )
    train = [atoms for index, atoms in enumerate(atoms_list) if index not in test_indices]
    test = [atoms for index, atoms in enumerate(atoms_list) if index in test_indices]
    return train, test

def _load_atoms_from_file(structure_file: str, index: str = ":") -> list[Atoms]:
    atoms_obj = read(structure_file, index=index)
    if isinstance(atoms_obj, Atoms):
        return [atoms_obj]
    return list(atoms_obj)


def run_fps_cli(
    structure_file: str,
    output: str = "fps_selected.xyz",
    index: str = ":",
    number: Optional[int] = None,
    r2_threshold: Optional[float] = None,
    descriptor: str = "structural",
    descriptor_model: Optional[str] = None,
    backend: str = "auto",
    min_dist: float = 0.0,
    reference: Optional[str] = None,
    reference_index: str = ":",
    pca_plot: Optional[str] = None,
    pca_title: Optional[str] = None,
    pca_max_points: Optional[int] = None,
) -> dict:
    atoms_list = _load_atoms_from_file(structure_file, index=index)
    if not atoms_list:
        raise ValueError(f"No structures found in {structure_file} with index={index}")

    reference_atoms_list = None
    if reference is not None:
        reference_atoms_list = _load_atoms_from_file(reference, index=reference_index)

    total = len(atoms_list)
    if number is None:
        n_samples = total
    else:
        if int(number) <= 0:
            raise ValueError("--number must be a positive integer")
        n_samples = min(int(number), total)

    selected_indices, info = select_structure_indices_with_info(
        atoms_list,
        n_samples=n_samples,
        method="fps",
        descriptor_model=descriptor_model,
        descriptor_mode=descriptor,
        min_dist=min_dist,
        nep_backend=backend,
        reference_atoms_list=reference_atoms_list,
        r2_threshold=r2_threshold,
        pca_plot_path=pca_plot,
        pca_plot_title=pca_title,
        pca_plot_max_points=pca_max_points,
    )
    selected_atoms = [atoms_list[i] for i in selected_indices]
    write(output, selected_atoms)

    result = {
        "input": structure_file,
        "output": output,
        "index": index,
        "selected": len(selected_atoms),
        "total": total,
        "indices": selected_indices,
        "descriptor": info.get("source"),
        "reference_count": info.get("reference_count"),
        "r2": info.get("r2"),
        "r2_threshold": info.get("r2_threshold"),
        "plot_path": info.get("plot_path"),
    }
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Use FPS to select representative structures from a trajectory or multi-structure file."
    )
    parser.add_argument("structurefile", help="Input structure or trajectory file")
    parser.add_argument("-o", "--output", default="fps_selected.xyz", help="Output file for selected structures")
    parser.add_argument("--index", default=":", help="ASE frame index, e.g. ':' or '0:100:5'")
    parser.add_argument("-n", "--number", type=int, default=None, help="Maximum number of structures to keep")
    parser.add_argument("--r2", "--R2", dest="r2_threshold", type=float, default=None, help="Optional R2 coverage threshold for early stopping")
    parser.add_argument("--descriptor", default="structural", choices=["auto", "structural", "nep", "soap"], help="Descriptor used for FPS")
    parser.add_argument("--model", dest="descriptor_model", default=None, help="Descriptor model file when --descriptor nep")
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu", "native"], help="Backend used for NEP descriptors")
    parser.add_argument("--min-dist", type=float, default=0.0, help="Stop FPS when min distance falls below this threshold")
    parser.add_argument("--reference", default=None, help="Optional reference trajectory/file used as an already-covered dataset")
    parser.add_argument("--reference-index", default=":", help="ASE index for the reference file")
    parser.add_argument("--pca-plot", default=None, help="Optional output PNG path for a PCA coverage plot")
    parser.add_argument("--pca-title", default=None, help="Optional title used by --pca-plot")
    parser.add_argument("--pca-max-points", type=int, default=None, help="Optional per-series plotting cap used by --pca-plot")
    args = parser.parse_args()

    if args.number is None and args.r2_threshold is None:
        parser.error("One of --number or --r2/--R2 must be provided")

    result = run_fps_cli(
        structure_file=args.structurefile,
        output=args.output,
        index=args.index,
        number=args.number,
        r2_threshold=args.r2_threshold,
        descriptor=args.descriptor,
        descriptor_model=args.descriptor_model,
        backend="auto" if args.backend == "native" else args.backend,
        min_dist=args.min_dist,
        reference=args.reference,
        reference_index=args.reference_index,
        pca_plot=args.pca_plot,
        pca_title=args.pca_title,
        pca_max_points=args.pca_max_points,
    )

    print(f"Input: {result['input']}")
    print(f"Output: {result['output']}")
    print(f"Selected: {result['selected']}/{result['total']}")
    print(f"Descriptor: {result['descriptor']}")
    print(f"Reference count: {result['reference_count']}")
    print(f"R2: {result['r2']}")
    print(f"R2 threshold: {result['r2_threshold']}")
    print(f"PCA plot: {result['plot_path']}")
    print(f"Indices: {result['indices']}")


if __name__ == "__main__":
    cli()
