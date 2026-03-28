from __future__ import annotations

import contextlib
import importlib
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from nepactive import dlog


DEFAULT_NEP89_RESOURCE_NAME = "nep89_20250409.txt"
SUPPORTED_ASE_MODELS = ("mattersim", "nep89")


def normalize_ase_model_name(model_name: str | None) -> str:
    model = str(model_name or "mattersim").strip().lower()
    if model not in SUPPORTED_ASE_MODELS:
        raise ValueError(
            f"Unsupported ASE model '{model}'. Use one of: {', '.join(SUPPORTED_ASE_MODELS)}"
        )
    return model


def get_resources_dir() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "resources",
        here.parent / "resources",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_ase_model_path(model_name: str | None = None, model_file: str | Path | None = None) -> Path | None:
    model = normalize_ase_model_name(model_name)
    if model_file is not None:
        path = Path(model_file).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"ASE model file not found: {path}")
        return path.resolve()
    if model == "mattersim":
        return None
    path = get_resources_dir() / DEFAULT_NEP89_RESOURCE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Default nep89 resource not found: {path}. Set ase_model_file explicitly."
        )
    return path.resolve()


def get_ase_model_config(data: dict | None = None) -> dict:
    cfg = data or {}
    return {
        "model_name": normalize_ase_model_name(cfg.get("ase_model", "mattersim")),
        "model_file": cfg.get("ase_model_file", cfg.get("pot_file")),
        "nep_backend": str(cfg.get("ase_nep_backend", "gpu") or "gpu").lower(),
    }


def create_ase_calculator(
    model_name: str | None = None,
    model_file: str | Path | None = None,
    device: str = "cuda",
    nep_backend: str = "gpu",
    batch_size: int | None = None,
):
    model = normalize_ase_model_name(model_name)
    resolved_path = resolve_ase_model_path(model, model_file)
    if model == "mattersim":
        from mattersim.forcefield import MatterSimCalculator

        kwargs = {"device": device}
        if resolved_path is not None:
            kwargs["load_path"] = str(resolved_path)
        return MatterSimCalculator(**kwargs)

    return NepCalculator(
        model_file=str(resolved_path),
        backend=nep_backend,
        batch_size=batch_size,
    )


class NepBackend(str, Enum):
    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"


def split_by_natoms(array: np.ndarray, natoms_list: list[int]) -> list[np.ndarray]:
    if array.size == 0:
        return []
    counts = np.asarray(list(natoms_list), dtype=int)
    split_indices = np.cumsum(counts)[:-1]
    return np.split(array, split_indices)


def aggregate_per_atom_to_structure(
    array: np.ndarray,
    atoms_num_list: Iterable[int],
    map_func=np.mean,
    axis: int = 0,
) -> np.ndarray:
    split_arrays = split_by_natoms(array, list(atoms_num_list))
    if not split_arrays:
        return np.array([], dtype=np.float32)
    return np.asarray([map_func(item, axis=axis) for item in split_arrays], dtype=np.float32)


def _import_native_module(name: str):
    candidates = [f"nepactive.{name}", name]
    for module_name in candidates:
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    return None


class NativeNepCalculator:
    def __init__(
        self,
        model_file: str | Path = "nep.txt",
        backend: NepBackend | str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_path = Path(model_file)
        self.backend = NepBackend(backend or NepBackend.AUTO)
        self.batch_size = int(batch_size or 1000)
        self.initialized = False
        self.nep3 = None
        self.element_list: list[str] = []
        self.type_dict: dict[str, int] = {}

        if not self.model_path.exists():
            dlog.warning("NEP model file not found: %s", self.model_path)
            return

        self._load_backend()
        if self.nep3 is None:
            return

        self.element_list = list(self.nep3.get_element_list())
        self.type_dict = {element: index for index, element in enumerate(self.element_list)}
        self.initialized = True

    def _load_backend(self) -> None:
        if self.backend in (NepBackend.AUTO, NepBackend.GPU):
            gpu_mod = _import_native_module("nep_gpu")
            if gpu_mod is not None and hasattr(gpu_mod, "GpuNep"):
                try:
                    self.nep3 = gpu_mod.GpuNep(str(self.model_path))
                    if hasattr(self.nep3, "set_batch_size"):
                        self.nep3.set_batch_size(self.batch_size)
                    self.backend = NepBackend.GPU
                    return
                except Exception as exc:
                    dlog.warning("Failed to initialize GPU NEP backend: %s", exc)
                    if self.backend == NepBackend.GPU:
                        raise

        cpu_mod = _import_native_module("nep_cpu")
        if cpu_mod is not None and hasattr(cpu_mod, "CpuNep"):
            self.nep3 = cpu_mod.CpuNep(str(self.model_path))
            self.backend = NepBackend.CPU
            return

        if self.backend == NepBackend.AUTO:
            raise ModuleNotFoundError(
                "No local NEP native backend found. Build nep_cpu/nep_gpu first."
            )
        raise ModuleNotFoundError(f"Requested NEP backend '{self.backend.value}' is unavailable.")

    @staticmethod
    def _ensure_structure_list(structures: Iterable[Atoms] | Atoms) -> list[Atoms]:
        if isinstance(structures, Atoms):
            return [structures]
        if isinstance(structures, list):
            return structures
        return list(structures)

    def compose_structures(
        self,
        structures: Iterable[Atoms] | Atoms,
    ) -> tuple[list[list[int]], list[list[float]], list[list[float]], list[int]]:
        structure_list = self._ensure_structure_list(structures)
        group_sizes: list[int] = []
        atom_types: list[list[int]] = []
        boxes: list[list[float]] = []
        positions: list[list[float]] = []

        for atoms in structure_list:
            symbols = atoms.get_chemical_symbols()
            mapped_types = [self.type_dict[symbol] for symbol in symbols]
            box = atoms.cell.transpose(1, 0).reshape(-1).tolist()
            coords = atoms.positions.transpose(1, 0).reshape(-1).tolist()
            atom_types.append(mapped_types)
            boxes.append(box)
            positions.append(coords)
            group_sizes.append(len(mapped_types))

        return atom_types, boxes, positions, group_sizes

    def calculate(
        self,
        structures: Iterable[Atoms] | Atoms,
        mean_virial: bool = True,
    ) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
        structure_list = self._ensure_structure_list(structures)
        if not self.initialized or not structure_list:
            return [], [], []

        atom_types, boxes, positions, group_sizes = self.compose_structures(structure_list)
        if hasattr(self.nep3, "reset_cancel"):
            self.nep3.reset_cancel()

        with contextlib.nullcontext():
            potentials, forces, virials = self.nep3.calculate(atom_types, boxes, positions)

        potentials_arr = np.asarray(potentials, dtype=np.float32)
        forces_arr = np.asarray(forces, dtype=np.float32)
        virials_arr = np.asarray(virials, dtype=np.float32)
        if potentials_arr.size == 0:
            return [], [], []
        if forces_arr.ndim == 1:
            forces_arr = forces_arr.reshape(-1, 3)
        if virials_arr.ndim == 1:
            virials_arr = virials_arr.reshape(-1, 9)

        potentials_array = aggregate_per_atom_to_structure(
            potentials_arr,
            group_sizes,
            map_func=np.sum,
            axis=0,
        ).tolist()
        force_blocks = split_by_natoms(forces_arr, group_sizes)
        if mean_virial:
            virial_blocks = aggregate_per_atom_to_structure(
                virials_arr,
                group_sizes,
                map_func=np.mean,
                axis=0,
            )
            return potentials_array, force_blocks, list(virial_blocks)
        return potentials_array, force_blocks, split_by_natoms(virials_arr, group_sizes)

    def get_structures_descriptor(
        self,
        structures: Iterable[Atoms] | Atoms,
        mean_descriptor: bool = True,
    ) -> np.ndarray:
        structure_list = self._ensure_structure_list(structures)
        if not self.initialized or not structure_list:
            return np.array([], dtype=np.float32)

        atom_types, boxes, positions, group_sizes = self.compose_structures(structure_list)
        if hasattr(self.nep3, "reset_cancel"):
            self.nep3.reset_cancel()
        descriptor = self.nep3.get_structures_descriptor(atom_types, boxes, positions)
        descriptor = np.asarray(descriptor, dtype=np.float32)
        if descriptor.size == 0 or not mean_descriptor:
            return descriptor
        return aggregate_per_atom_to_structure(
            descriptor,
            group_sizes,
            map_func=np.mean,
            axis=0,
        )


def _virial9_to_ase_stress(virial9: np.ndarray, volume: float) -> np.ndarray:
    if volume <= 1e-12:
        return np.zeros(6, dtype=np.float32)
    virial_tensor = np.asarray(virial9, dtype=np.float32).reshape(3, 3)
    stress_tensor = -virial_tensor / float(volume)
    return np.asarray(
        [
            stress_tensor[0, 0],
            stress_tensor[1, 1],
            stress_tensor[2, 2],
            stress_tensor[1, 2],
            stress_tensor[0, 2],
            stress_tensor[0, 1],
        ],
        dtype=np.float32,
    )


class NepCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model_file: str | Path = "nep.txt",
        backend: NepBackend | str | None = None,
        batch_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.native = NativeNepCalculator(
            model_file=model_file,
            backend=backend,
            batch_size=batch_size,
        )
        self.model_path = self.native.model_path
        self.backend = self.native.backend

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("ASE NEP calculator requires an Atoms object")
        if not self.native.initialized:
            raise RuntimeError(f"Failed to initialize native NEP backend from {self.model_path}")

        energies, forces_blocks, virial_blocks = self.native.calculate([atoms], mean_virial=False)
        if len(energies) != 1 or len(forces_blocks) != 1 or len(virial_blocks) != 1:
            raise RuntimeError(
                f"Native NEP returned inconsistent single-structure results: "
                f"energies={len(energies)}, forces={len(forces_blocks)}, virials={len(virial_blocks)}"
            )

        energy = float(energies[0])
        forces = np.asarray(forces_blocks[0], dtype=np.float32)
        virial_tensor = np.asarray(virial_blocks[0], dtype=np.float32)
        total_virial = np.sum(virial_tensor, axis=0, dtype=np.float32)
        stress = _virial9_to_ase_stress(total_virial, atoms.get_volume())

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = stress


NEP = NepCalculator
AseNativeNepCalculator = NepCalculator


def has_native_nep_backend() -> bool:
    return _import_native_module("nep_gpu") is not None or _import_native_module("nep_cpu") is not None
