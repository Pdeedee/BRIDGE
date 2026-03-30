# Standard library imports
import copy
import itertools
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from math import ceil, floor
from typing import List, Optional

# Third-party imports
import numpy as np
import pandas as pd
import yaml
from ase import Atoms, units
from ase.io import read, write, Trajectory
from ase.optimize import BFGS
from torch.cuda import empty_cache
from tqdm import tqdm

# Local package imports
from nepactive import dlog, parse_yaml
from nepactive.extract import analyze_trajectory
from nepactive.force import force_main
from nepactive.packmol import make_structure
from nepactive.plt import ase_plt, gpumdplt, nep_plt
from nepactive.remote import Remotetask
from nepactive.nep_backend import create_ase_calculator, get_ase_model_config, resolve_ase_model_path
from nepactive.sampling import select_structure_indices, select_structure_indices_with_info, split_train_test_structures
from nepactive.stable import InitRun, ShockRun
from nepactive.template import (
    continue_pytemplate,
    model_devi_template,
    msst_template,
    nep_in_template,
    nphugo_mttk_pytemplate,
    nphugo_mttk_template,
    npt_scr_template,
    npt_template,
    nvt_pytemplate,
    nvt_template,
)
from nepactive.tools import (
    compute_volume_from_thermo,
    get_shortest_distance,
    run_gpumd_task,
    run_py_tasks,
)
from nepactive.write_extxyz import write_extxyz

class RestartSignal(Exception):
    def __init__(self, restart_total_time = None):
        super().__init__()
        self.restart_total_time = restart_total_time


class TrainingCompletedException(Exception):
    """Raised when training reaches convergence criteria."""
    def __init__(self, message, accuracy=None, error=None):
        super().__init__(message)
        self.accuracy = accuracy
        self.error = error

def process_id(value):
    # 检查是否为单一数字
    if re.match(r'^\d+$', value):
        return [int(value)]  # 如果是数字，返回列表形式的单个数字
    # 检查是否为有效的数字范围，如 "3-11"
    if re.match(r'^\d+-\d+$', value):
        start, end = value.split('-')
        start, end = int(start), int(end)
        if start > end:
            raise ValueError(f"范围的起始值不能大于结束值: {value}")
        return list(range(start, end + 1))  # 返回范围的数字列表
    # 如果不匹配数字或范围格式，抛出异常
    raise ValueError(f"无效的格式: {value}，必须是数字或者数字范围（如 '3-11'）")

def traj_write(atoms_list:Atoms, calculator):
    traj = Trajectory("out.traj", "w")
    for atoms in atoms_list:
        atoms._calc = calculator
        atoms.get_potential_energy()
        traj.write(atoms)    

def get_force(atoms:Atoms,calculator):
    atoms._calc=calculator
    return atoms.get_forces(),atoms.get_potential_energy()

# task = None
Maxlength = 70
def sepline(ch="-", sp="-"):
    r"""Seperate the output by '-'."""
    # if screen:
    #     print(ch.center(MaxLength, sp))
    # else:
    dlog.info(ch.center(Maxlength, sp))

def record_iter(record, ii, jj):
    with open(record, "a") as frec:
        frec.write("%d %d\n" % (ii, jj))

class Nepactive(object):
    @staticmethod
    def _sampling_config(data: Optional[dict]) -> dict:
        sampling = data.get("sampling")
        if sampling is None:
            raise KeyError("sampling")
        if not isinstance(sampling, dict):
            raise TypeError("sampling must be a mapping")
        return sampling

    @classmethod
    def _sampling_general_config(cls, data: Optional[dict]) -> dict:
        sampling_general = cls._sampling_config(data).get("general")
        if sampling_general is None:
            raise KeyError("sampling.general")
        if not sampling_general:
            raise ValueError("sampling.general is empty in config")
        if not isinstance(sampling_general, dict):
            raise TypeError("sampling.general must be a mapping")
        return sampling_general

    @staticmethod
    def _iter_existing_structure_files(data: dict, work_dir: Optional[str] = None):
        structure_prefix = data.get("structure_prefix", work_dir or os.getcwd())
        structure_files = data.get("structure_files", []) or []
        for struct_file in structure_files:
            struct_path = struct_file if os.path.isabs(struct_file) else os.path.join(structure_prefix, struct_file)
            if os.path.exists(struct_path):
                yield struct_path

    @classmethod
    def infer_nep_in_header(cls, data: dict, work_dir: Optional[str] = None) -> str:
        explicit_header = data.get("nep_in_header")
        if explicit_header:
            return explicit_header

        elements = []
        for struct_path in cls._iter_existing_structure_files(data, work_dir=work_dir):
            atoms = read(struct_path)
            for symbol in atoms.get_chemical_symbols():
                if symbol not in elements:
                    elements.append(symbol)

        if not elements:
            raise ValueError(
                "Cannot infer nep_in_header because no readable structure_files were found. "
                "Set nep_in_header explicitly in in.yaml."
            )
        return f"type {len(elements)} " + " ".join(elements)

    @staticmethod
    def _read_nep_in_header(nep_in_path: str) -> Optional[str]:
        if not os.path.exists(nep_in_path):
            return None
        with open(nep_in_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("type "):
                    return stripped
        return None

    def _resolve_nep_in_header(self, task_index: int, pot_inherit: bool) -> str:
        current_header = self.infer_nep_in_header(self.idata, work_dir=self.work_dir)
        explicit_header = self.idata.get("nep_in_header")
        if explicit_header:
            return explicit_header
        if pot_inherit and self.ii > 0:
            previous_nep_in = os.path.join(
                self.work_dir,
                f"iter.{self.ii-1:06d}",
                "00.nep",
                f"task.{task_index:06d}",
                "nep.in",
            )
            previous_header = self._read_nep_in_header(previous_nep_in)
            if previous_header and previous_header != current_header:
                raise ValueError(
                    "Auto-inferred nep_in_header changed while pot_inherit=True: "
                    f"previous='{previous_header}', current='{current_header}'. "
                    "Set nep_in_header explicitly in in.yaml to keep element order fixed."
                )
        return current_header

    def _resolve_user_nep_in_template_path(self) -> Optional[str]:
        explicit_template = self.idata.get("nep_template")
        candidates: list[str] = []
        if explicit_template:
            candidates.append(
                explicit_template if os.path.isabs(explicit_template) else os.path.join(self.work_dir, explicit_template)
            )
        candidates.append(os.path.join(self.work_dir, "nep.in"))

        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return None

    @staticmethod
    def _override_nep_generation(nep_in_text: str, train_steps: int) -> str:
        lines = nep_in_text.splitlines()
        generation_line = f"generation    {int(train_steps)}"
        replaced = False
        new_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("generation"):
                indent = line[: len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}{generation_line}")
                replaced = True
            else:
                new_lines.append(line)

        if not replaced:
            if new_lines and new_lines[-1].strip():
                new_lines.append("")
            new_lines.append(generation_line)

        return "\n".join(new_lines).rstrip() + "\n"

    def _build_nep_in_content(self, task_index: int, pot_inherit: bool, train_steps: int) -> str:
        user_template_path = self._resolve_user_nep_in_template_path()
        if user_template_path:
            with open(user_template_path, "r", encoding="utf-8") as f:
                user_nep_in = f.read()
            dlog.info("Using user-provided nep.in template: %s", user_template_path)
            return self._override_nep_generation(user_nep_in, train_steps)

        nep_in_header = self._resolve_nep_in_header(task_index, pot_inherit=pot_inherit)
        return nep_in_template.format(train_steps=train_steps, nep_in_header=nep_in_header)

    @staticmethod
    def _ensure_atoms_list(atoms_obj) -> list[Atoms]:
        if atoms_obj is None:
            return []
        if isinstance(atoms_obj, Atoms):
            return [atoms_obj]
        return list(atoms_obj)

    def _load_xyz_structures(self, file_path: str) -> list[Atoms]:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return []
        atoms_obj = read(file_path, index=":")
        return self._ensure_atoms_list(atoms_obj)

    def _filter_atoms_by_shortest_distance(
        self,
        atoms_list: list[Atoms],
        min_distance: float,
        context: str,
    ) -> tuple[list[Atoms], np.ndarray]:
        if not atoms_list:
            return [], np.array([], dtype=float)
        if min_distance is None or float(min_distance) <= 0:
            return list(atoms_list), np.array([], dtype=float)

        shortest_distances = np.asarray(
            [get_shortest_distance(atoms) for atoms in atoms_list],
            dtype=float,
        )
        keep_mask = shortest_distances >= float(min_distance)
        removed_count = int(np.count_nonzero(~keep_mask))
        if removed_count:
            dlog.warning(
                "%s removed %d/%d structures with shortest distance < %.3f",
                context,
                removed_count,
                len(atoms_list),
                float(min_distance),
            )
        filtered_atoms = [atoms for atoms, keep in zip(atoms_list, keep_mask) if keep]
        return filtered_atoms, shortest_distances

    def _truncate_atoms_before_shortest_distance_failure(
        self,
        atoms_list: list[Atoms],
        min_distance: float,
        context: str,
    ) -> tuple[list[Atoms], np.ndarray]:
        if not atoms_list:
            return [], np.array([], dtype=float)
        if min_distance is None or float(min_distance) <= 0:
            return list(atoms_list), np.array([], dtype=float)

        shortest_distances = np.asarray(
            [get_shortest_distance(atoms) for atoms in atoms_list],
            dtype=float,
        )
        bad_indices = np.where(shortest_distances < float(min_distance))[0]
        if bad_indices.size == 0:
            return list(atoms_list), shortest_distances

        first_bad = int(bad_indices[0])
        dlog.warning(
            "%s truncated at frame %d; discarded %d trailing structures with shortest distance < %.3f",
            context,
            first_bad,
            len(atoms_list) - first_bad,
            float(min_distance),
        )
        return list(atoms_list[:first_bad]), shortest_distances

    def _sampling_reference_cache_paths(self) -> tuple[str, str]:
        return (
            os.path.join(self.work_dir, ".sampling_reference_cache.xyz"),
            os.path.join(self.work_dir, ".sampling_reference_cache.yaml"),
        )

    def _sampling_reference_signature(self) -> dict:
        sampling_cfg = self._sampling_config(self.idata)
        descriptor_kwargs = self._sampling_descriptor_kwargs()
        descriptor_kwargs["method"] = "fps"
        return {
            "max_reference_points": int(sampling_cfg.get("max_reference_points", 10000) or 0),
            "shortest_d": float(self.idata.get("shortest_d", 0.0) or 0.0),
            "descriptor_kwargs": descriptor_kwargs,
        }

    @staticmethod
    def _load_sampling_cache_metadata(meta_path: str) -> Optional[dict]:
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return None
        return data

    def _write_sampling_reference_cache(
        self,
        atoms_list: list[Atoms],
        meta_path: str,
        xyz_path: str,
        signature: dict,
        last_reference_iter: int,
    ) -> None:
        with open(xyz_path, "w", encoding="utf-8") as f:
            if atoms_list:
                write_extxyz(f, atoms_list)
        metadata = {
            "version": 1,
            "last_reference_iter": int(last_reference_iter),
            "reference_count": len(atoms_list),
            "signature": signature,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(metadata, f, sort_keys=True)

    def _sampling_reference_files(
        self,
        start_iter: Optional[int] = None,
        end_iter: Optional[int] = None,
        include_init: bool = True,
    ) -> list[str]:
        reference_files: list[str] = []
        if include_init:
            reference_files.extend(
                [
                    os.path.join(self.work_dir, "init", "iter_train.xyz"),
                    os.path.join(self.work_dir, "init", "iter_test.xyz"),
                ]
            )
        if start_iter is None or end_iter is None:
            return reference_files
        for iii in range(max(start_iter, 0), end_iter + 1):
            reference_files.extend(
                [
                    os.path.join(self.work_dir, f"iter.{iii:06d}", "02.label", "iter_train.xyz"),
                    os.path.join(self.work_dir, f"iter.{iii:06d}", "02.label", "iter_test.xyz"),
                ]
            )
        return reference_files

    def _load_reference_atoms_from_files(self, file_paths: list[str]) -> list[Atoms]:
        reference_atoms: list[Atoms] = []
        for file_path in file_paths:
            reference_atoms.extend(self._load_xyz_structures(file_path))
        return reference_atoms

    def _compress_reference_atoms(
        self,
        reference_atoms: list[Atoms],
        max_reference_points: int,
        signature: dict,
        context: str,
    ) -> list[Atoms]:
        if max_reference_points <= 0 or len(reference_atoms) <= max_reference_points:
            return reference_atoms

        sampling_kwargs = dict(signature["descriptor_kwargs"])
        sampling_kwargs.update(
            self._sampling_pca_plot_kwargs(
                os.path.join(self.work_dir, ".sampling_reference_cache_pca.png"),
                "Sampling reference representative set",
            )
        )
        selected_indices, sampling_info = select_structure_indices_with_info(
            reference_atoms,
            n_samples=max_reference_points,
            **sampling_kwargs,
        )
        compressed_atoms = [reference_atoms[index] for index in selected_indices]
        dlog.info(
            "%s compressed %d -> %d structures with %s FPS representative set",
            context,
            len(reference_atoms),
            len(compressed_atoms),
            sampling_info.get("source"),
        )
        return compressed_atoms

    def _load_sampling_reference_structures(self) -> list[Atoms]:
        signature = self._sampling_reference_signature()
        max_reference_points = int(signature["max_reference_points"])
        target_last_iter = self.ii - 1
        xyz_path, meta_path = self._sampling_reference_cache_paths()

        if max_reference_points <= 0:
            reference_atoms = self._load_reference_atoms_from_files(
                self._sampling_reference_files(start_iter=0, end_iter=target_last_iter)
            )
            reference_atoms, _ = self._filter_atoms_by_shortest_distance(
                reference_atoms,
                signature["shortest_d"],
                "Sampling reference pool",
            )
            dlog.info("Loaded %d reference structures for dataset-aware FPS", len(reference_atoms))
            return reference_atoms

        meta = self._load_sampling_cache_metadata(meta_path)
        cache_valid = (
            meta is not None
            and meta.get("version") == 1
            and meta.get("signature") == signature
            and os.path.exists(xyz_path)
        )

        rebuild_required = (
            not cache_valid
            or int(meta.get("last_reference_iter", -10**9)) > target_last_iter
        )

        if rebuild_required:
            reference_atoms = self._load_reference_atoms_from_files(
                self._sampling_reference_files(start_iter=0, end_iter=target_last_iter)
            )
            reference_atoms, _ = self._filter_atoms_by_shortest_distance(
                reference_atoms,
                signature["shortest_d"],
                "Sampling reference pool",
            )
            original_count = len(reference_atoms)
            reference_atoms = self._compress_reference_atoms(
                reference_atoms,
                max_reference_points,
                signature,
                "Sampling reference pool rebuild",
            )
            self._write_sampling_reference_cache(
                reference_atoms,
                meta_path,
                xyz_path,
                signature,
                target_last_iter,
            )
            dlog.info(
                "Rebuilt sampling reference cache for iterations <= %d: %d -> %d structures",
                target_last_iter,
                original_count,
                len(reference_atoms),
            )
            return reference_atoms

        reference_atoms = self._load_xyz_structures(xyz_path)
        cached_last_iter = int(meta.get("last_reference_iter", -1))
        if cached_last_iter < target_last_iter:
            new_atoms = self._load_reference_atoms_from_files(
                self._sampling_reference_files(
                    start_iter=cached_last_iter + 1,
                    end_iter=target_last_iter,
                    include_init=False,
                )
            )
            new_atoms, _ = self._filter_atoms_by_shortest_distance(
                new_atoms,
                signature["shortest_d"],
                "Sampling reference pool increment",
            )
            combined_atoms = reference_atoms + new_atoms
            updated_atoms = self._compress_reference_atoms(
                combined_atoms,
                max_reference_points,
                signature,
                "Sampling reference pool increment",
            )
            self._write_sampling_reference_cache(
                updated_atoms,
                meta_path,
                xyz_path,
                signature,
                target_last_iter,
            )
            dlog.info(
                "Updated sampling reference cache from iterations <= %d to <= %d using %d new structures",
                cached_last_iter,
                target_last_iter,
                len(new_atoms),
            )
            reference_atoms = updated_atoms

        dlog.info("Loaded %d reference structures for dataset-aware FPS", len(reference_atoms))
        return reference_atoms

    def _sampling_descriptor_kwargs(self) -> dict:
        sampling_cfg = self._sampling_config(self.idata)
        dataset_descriptor = sampling_cfg.get("dataset_descriptor", "nep")
        return {
            "method": sampling_cfg.get("dataset_method", "fps"),
            "descriptor_model": self._resolve_sampling_nep_file(
                sampling_cfg.get("dataset_nep_file"),
                stage="dataset",
            ),
            "descriptor_mode": dataset_descriptor,
            "min_dist": sampling_cfg.get("dataset_min_dist", 0.0),
            "nep_backend": sampling_cfg.get("dataset_backend", "auto"),
        }

    def _resolve_sampling_nep_file(self, explicit_model: Optional[str] = None, stage: str = "dataset") -> Optional[str]:
        if explicit_model:
            return explicit_model
        legacy_model = self.idata.get("nep_file")
        if legacy_model:
            return legacy_model
        if str(stage).lower() == "init":
            resolved = resolve_ase_model_path("nep89", None)
            return None if resolved is None else str(resolved)

        sampling_cfg = self._sampling_config(self.idata)
        fps_pot = str(sampling_cfg.get("fps_pot", "nep89") or "nep89").lower()
        if fps_pot == "nep89":
            resolved = resolve_ase_model_path("nep89", None)
            return None if resolved is None else str(resolved)
        if fps_pot == "self":
            current_nep = os.path.join(self.iter_dir, "00.nep", "task.000000", "nep.txt")
            if not os.path.isfile(current_nep):
                raise FileNotFoundError(
                    f"sampling.fps_pot=self requires current iteration NEP model: {current_nep}"
                )
            return current_nep
        raise ValueError(f"Unsupported sampling.fps_pot: {fps_pot}. Use 'nep89' or 'self'.")

    def _sampling_pca_plot_kwargs(self, output_path: str, title: str) -> dict:
        sampling_cfg = self._sampling_config(self.idata)
        if not bool(sampling_cfg.get("fps_pca_plot", True)):
            return {}
        return {
            "pca_plot_path": output_path,
            "pca_plot_title": title,
            "pca_plot_max_points": sampling_cfg.get("fps_pca_max_points"),
        }

    def _final_candidate_sampling_kwargs(self) -> dict:
        sampling_cfg = self._sampling_config(self.idata)
        kwargs = self._sampling_descriptor_kwargs()
        kwargs["r2_threshold"] = sampling_cfg.get("final_r2_threshold")
        return kwargs

    def _ase_model_config(self) -> dict:
        return get_ase_model_config(self.idata)

    def _ase_template_kwargs(self) -> dict:
        cfg = self._ase_model_config()
        return {
            "ase_model_name": repr(cfg["model_name"]),
            "ase_model_file": repr(cfg["model_file"]),
            "ase_nep_backend": repr(cfg["nep_backend"]),
        }

    @staticmethod
    def _write_sampling_stats(stats_path: str, stats: dict) -> None:
        with open(stats_path, "w", encoding="utf-8") as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

    def _cap_run_steps(self, run_steps: int, context: str) -> int:
        capped_steps = int(run_steps)
        max_run_steps = int(self.idata.get("max_run_steps", 1200000) or 0)
        if max_run_steps > 0 and capped_steps > max_run_steps:
            dlog.info(
                "%s capped run_steps from %d to max_run_steps=%d",
                context,
                capped_steps,
                max_run_steps,
            )
            capped_steps = max_run_steps
        return capped_steps

    def _write_global_candidate_pool(
        self,
        candidate_records: list[dict],
        candidate_file: str,
        max_candidate: int,
        existing_atoms: Optional[list[Atoms]] = None,
    ) -> int:
        candidate_atoms: list[Atoms] = list(existing_atoms or [])
        for record in candidate_records:
            candidate_atoms.extend(record.get("atoms", []))

        candidate_atoms, _ = self._filter_atoms_by_shortest_distance(
            candidate_atoms,
            self.idata.get("shortest_d", 0.0),
            "Global candidate pool",
        )
        if not candidate_atoms:
            open(candidate_file, "w", encoding="utf-8").close()
            return 0

        stats_path = os.path.join(os.path.dirname(candidate_file), "candidate_fps_stats.txt")
        if len(candidate_atoms) > max_candidate:
            reference_atoms = self._load_sampling_reference_structures()
            selected_indices, sampling_info = select_structure_indices_with_info(
                candidate_atoms,
                n_samples=max_candidate,
                reference_atoms_list=reference_atoms,
                **self._final_candidate_sampling_kwargs(),
                **self._sampling_pca_plot_kwargs(
                    os.path.join(os.path.dirname(candidate_file), "candidate_fps_pca.png"),
                    "Global candidate FPS coverage",
                ),
            )
            candidate_atoms = [candidate_atoms[index] for index in selected_indices]
            total_candidates = len(existing_atoms or []) + sum(len(record.get("atoms", [])) for record in candidate_records)
            dlog.info(
                "Global candidate FPS kept %d/%d structures against %d reference structures",
                len(candidate_atoms),
                total_candidates,
                len(reference_atoms),
            )
            self._write_sampling_stats(
                stats_path,
                {
                    "mode": "global_candidate_pool",
                    "total_candidates": total_candidates,
                    "selected_candidates": len(candidate_atoms),
                    "reference_count": sampling_info.get("reference_count"),
                    "descriptor_source": sampling_info.get("source"),
                    "r2": sampling_info.get("r2"),
                    "r2_threshold": sampling_info.get("r2_threshold"),
                    "plot_path": sampling_info.get("plot_path"),
                },
            )
        else:
            dlog.info("Global candidate pool kept all %d structures", len(candidate_atoms))
            self._write_sampling_stats(
                stats_path,
                {
                    "mode": "global_candidate_pool",
                    "total_candidates": len(candidate_atoms),
                    "selected_candidates": len(candidate_atoms),
                    "reference_count": 0,
                    "descriptor_source": "none",
                    "r2": None,
                    "r2_threshold": self._final_candidate_sampling_kwargs().get("r2_threshold"),
                },
            )

        with open(candidate_file, "w", encoding="utf-8") as candidate_f:
            write_extxyz(candidate_f, candidate_atoms)
        return len(candidate_atoms)

    def __init__(self,idata:dict):
        self.idata:dict = idata
        self.work_dir = os.getcwd()
        self.make_gpumd_task_first = True
        self.gpu_available = self.idata.get("gpu_available")
        self.shock_run= self.idata.get("shock_run", True)
        self.structure_prefix = self.idata.get("structure_prefix", self.work_dir)
        if self.shock_run:
            self.check_inistruc()
            sampling_cfg = self._sampling_general_config(self.idata)
            if os.path.exists(f"{self.work_dir}/nostrucnum"):
                self.idata.setdefault("init", {})["struc_num"] = 0
                self.idata["structure_files"] = self.idata.get("structure_files",["POSCAR"])
                sampling_cfg["structure_id"] = sampling_cfg.get("structure_id", [[0]])
            else:
                self.idata.setdefault("init", {})["struc_num"] = 1
                self.idata["structure_files"] = ["POSCAR","init/struc.000/structure/POSCAR"]
                sampling_cfg["structure_id"] = [[0, 1]]
        else:
            self.idata.setdefault("init", {})["struc_num"] = 0
        # dlog.info(f"self.idata:{self.idata}")
        # print(f"structure_files: {self.idata['structure_files']}")
            
    def check_inistruc(self):
        os.chdir(self.work_dir)
        # atoms = read("POSCAR.p")
        atoms = read(f"{self.structure_prefix}/{self.idata.get('structure_files')[0]}")
        if os.path.exists("POSCAR"):
            return
        elif os.path.exists("molecule.xyz"):
            atoms = read("molecule.xyz")
            os.makedirs("init",exist_ok=True)
            os.chdir("init")
            if not os.path.exists(f"molecule.pdb"):
                atoms.calc = create_ase_calculator(**self._ase_model_config())
                opt = BFGS(atoms)
                opt.run(fmax=0.05,steps=100)
                write(f"molecule.pdb", atoms)
            else:
                atoms = read(f"molecule.pdb")
            nat = len(atoms)
            nat_r = floor(200/nat)
            # mass_per_molecule = atoms.get_masses().sum()
            # total_mass = nat_r * mass_per_molecule
            # density = 1.8e3 * units.g/units.m**3  # g/cm^3
            # length = (total_mass / density)**(1/3)
            molecule_dict = {"molecule": nat_r}
            make_structure(molecule_dict, name=f"{self.work_dir}/POSCAR", density=1.8e3)
            os.chdir(self.work_dir)
        elif self.idata.get("structure_files", None):
            atoms = read(f"{self.structure_prefix}/{self.idata.get('structure_files')[0]}")
            write(f"{self.work_dir}/POSCAR", atoms)
        else:
            raise ValueError("No initial structure found, please provide POSCAR or molecule.xyz")

    def _sampling_dir(self, ii: Optional[int] = None, prefer_existing: bool = True) -> str:
        idx = self.ii if ii is None else ii
        new_dir = os.path.join(self.work_dir, f"iter.{idx:06d}", "01.sampling")
        old_dir = os.path.join(self.work_dir, f"iter.{idx:06d}", "01.gpumd")
        if prefer_existing:
            if os.path.exists(new_dir):
                return new_dir
            if os.path.exists(old_dir):
                return old_dir
        return new_dir

    def _sampling_state_path(self, filename: str) -> str:
        new_path = os.path.join(self.work_dir, filename.replace("model_devi", "sampling"))
        old_path = os.path.join(self.work_dir, filename)
        if os.path.exists(new_path):
            return new_path
        if os.path.exists(old_path):
            return old_path
        return new_path

    def run(self):
        '''
        using different engines to make initial training data
        in.yaml need:
        init_engine: the engine to use for initial training data
        init_template_files: the template files to use for initial training data
        python_interpreter: the python interpreter to use for initial training data
        training_ratio: the ratio of training data to total data
        '''
        #change working directory

        engine = self.idata.get("ini_engine","ase")
        # label_engine = idata.get("label_engine","mattersim")
        work_dir = self.work_dir
        os.chdir(work_dir)
        os.makedirs("init",exist_ok=True)
        record = "record.nep"
        #init engine choice
        dlog.info(f"Initializing engine and make initial training data using {engine}")
        if not os.path.isfile(record):
            self.calculate_properties()
            self.make_init_ase_run() 
            dlog.info("Extracting data from initial runs")
            self.make_data_extraction()
        self.make_loop_train()

    def calculate_properties(self):
        os.chdir(self.work_dir)
        os.makedirs("init",exist_ok=True)
        if os.path.exists("POSCAR"):
            shutil.copy("POSCAR","init")
        work_dir = f"{self.work_dir}/init"
        os.chdir(work_dir)
        init_data:dict = self.idata.get("init", {})
        init_data["python_interpreter"] = self.idata.get("python_interpreter", "python")
        init_data["tfreq"] = self.idata.get("tfreq", None)
        init_data["pfreq"] = self.idata.get("pfreq", None)
        init_run = InitRun(self.idata, init_data)
        if not self.shock_run:
            dlog.info(f"shock_run is False, skip calculate properties")
            format = '%12.5f'
            property = [0,0,0,0,0]
            np.savetxt("properties.txt",np.array(property).reshape(1, -1),fmt=format,encoding="utf-8")
            # np.savetxt('frame_properties.txt', data, fmt='%12.2f', header="shortest_d, molecule_num, molecule_density")
        else:
            init_run.calculate_properties()


    def make_init_ase_run(self):
        '''
        For the template file, the file name must be fixed form.
        Assumed that the working directory is already the correct directory.
        '''
        # if_stable_run = self.idata.get("if_stable_run",False)

        work_dir = f"{self.work_dir}/init"
        # struc_dirs = []
        os.chdir(work_dir)
        # if if_stable_run:
        rho = self.idata.get("rho", None)
        init_data:dict = self.idata.get("init", {})
        init_data["python_interpreter"] = self.idata.get("python_interpreter", "python")
        if rho:
            dlog.info(f"rho is {rho}, will run stable run for rho={rho}")
            init_data["rho"] = rho
        init_run = InitRun(self.idata, init_data)
        init_run.calculate_properties()
        if self.shock_run:
            if not os.path.exists(f"{self.work_dir}/nostrucnum"):
                sampling_cfg = self._sampling_general_config(self.idata)
                self.idata["structure_files"]=["POSCAR","init/struc.000/structure/POSCAR"]
                sampling_cfg["structure_id"] = sampling_cfg.get("structure_id", [[0, 1], [0, 1]])
                sampling_cfg["temperature"] = sampling_cfg.get("temperature", [3000])
                sampling_cfg["ensembles"] = sampling_cfg.get("ensembles", ["nphugo_scr", "nvt"])
                try:
                    for ii in range(init_run.struc_num):
                        os.chdir(work_dir)
                        os.makedirs(f"struc.{ii:03d}",exist_ok=True)
                        struc_dir = os.path.abspath(f"struc.{ii:03d}")
                        # struc_dirs.append(struc_dir)
                        os.chdir(struc_dir)
                        init_run.make_preparations()
                except ValueError as e:
                    dlog.warning(f"Error in make preparations: {e}")
                    os.system(f"touch {self.work_dir}/nostrucnum")
                    if self.shock_run:
                        sampling_cfg = self._sampling_general_config(self.idata)
                        self.idata.setdefault("init", {})["struc_num"] = 0
                        self.idata["structure_files"] = self.idata.get("structure_files",["POSCAR"])
                        sampling_cfg["structure_id"] = sampling_cfg.get("structure_id", [[0], [0]])
                        sampling_cfg["ensembles"] = sampling_cfg.get("ensembles", ["nphugo_scr", "nvt"])
                        sampling_cfg["temperature"] = sampling_cfg.get("temperature", [3000])
                        dlog.warning("Exception occurred, set struc_num to 0 and continue")

        # 处理用户在 in.yaml 中额外提供的结构文件（跳过自动生成的 init/struc.000/structure/POSCAR）
        structure_files = self.idata.get("structure_files", ["POSCAR"])
        for ii, struct_file in enumerate(structure_files):
            # 跳过 POSCAR 和自动生成的 init/struc.*/structure/POSCAR
            if ii == 0 or struct_file.startswith("init/struc."):
                continue
            os.chdir(work_dir)
            os.makedirs(f"struc.{ii:03d}",exist_ok=True)
            struc_dir = os.path.abspath(f"struc.{ii:03d}")
            os.chdir(struc_dir)
            os.makedirs("structure",exist_ok=True)
            dlog.info(f"Processing additional structure file: {struct_file}")
            atoms = read(f"{self.structure_prefix}/{struct_file}")
            elements = atoms.get_chemical_symbols()
            sorted_indices = sorted(range(len(elements)), key=lambda x: elements[x])
            sorted_atoms = atoms[sorted_indices]
            write("structure/POSCAR", sorted_atoms)
            write("structure/stable.pdb", sorted_atoms)
            init_run.make_preparations()

        if True:
            os.chdir(work_dir)
            os.makedirs("original",exist_ok=True)
            struc_dir = os.path.abspath("original")
            # struc_dirs.append(struc_dir)
            os.chdir(struc_dir)
            atoms = read(f"{self.work_dir}/POSCAR")
            elements = atoms.get_chemical_symbols()
            sorted_indices = sorted(range(len(elements)), key=lambda x: elements[x])
            sorted_atoms = atoms[sorted_indices]
            os.makedirs("structure",exist_ok=True)
            write("structure/POSCAR", sorted_atoms)
            write("structure/stable.pdb", sorted_atoms)
            init_run.make_preparations()

        ase_ensemble_files = self.idata.get("ini_ase_ensemble_files")
        if ase_ensemble_files:
            ase_ensemble_files:list[str] = [os.path.abspath(path) for path in ase_ensemble_files]

        python_interpreter:str = self.idata.get("python_interpreter")
        processes = []
        ase_model_file = self._ase_model_config()["model_file"]
        self.gpu_available = self.idata.get("gpu_available")
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        # Make initial training task directories
        if ase_ensemble_files:
            for index, file in enumerate(ase_ensemble_files):
                task_name = f"task.{index:06d}"
                task_dir = os.path.join(work_dir, task_name)
                os.makedirs(task_dir, exist_ok=True)
                if ase_model_file:
                    model_path = os.path.abspath(str(ase_model_file))
                    os.system(f"ln -snf {model_path} {task_dir}/{os.path.basename(model_path)}")
                os.system(f"ln -snf {file} {task_dir}")
        
        os.chdir(work_dir)

        self.run_py_tasks()
        task_dirs = glob("./**/task.*", recursive=True)
        if task_dirs:
            task_dirs = [os.path.abspath(path) for path in task_dirs]
        for task_dir in task_dirs:
            os.chdir(task_dir)
            ase_plt()
            atoms = read("out.traj", index=-1)
            write_extxyz("final.xyz", atoms)
            dlog.info(f"Process completed successfully. Log saved at: {task_dir}/log")

        dlog.info("Initial training data generated")
    
    def run_py_tasks(self):
        dlog.info(f"start run py tasks")
        gpu_available = self.idata.get("gpu_available",[0,1,2,3])
        self.task_per_gpu = self.idata.get("task_per_gpu",1)
        run_py_tasks(gpu_available=gpu_available, task_per_gpu=self.task_per_gpu)

    def make_data_extraction(self):
        '''
        extract data from initial runs, turn it into the gpumd format
        '''
        train:List[Atoms]=[]
        test:List[Atoms]=[]
        training_ratio:float = self.idata.get("training_ratio", 0.8)
        init_frames:int = self.idata.get("ini_frames", 100)
        # work_dir = f"{work_dir}/init"
        os.chdir(self.work_dir)

        ####检查
        fs = glob("init/**/task.*/*.traj",recursive=True)
        dlog.info(f"Found {len(fs)} files to extract frames from.")
        if not fs:
            raise ValueError("No files found to extract frames from.")
        # may report error due to the format not matching
        atoms = []
        
        # average sampling
        fnumber= len(fs)
        needed_frames = init_frames/fnumber
        
        for f in fs:
            atom = read(f,index=":")
            if len(atom) < needed_frames:
                dlog.warning(f"Not enough frames in {f} to extract {needed_frames} frames.")
                # raise ValueError(f"Not enough frames in {f} to extract {needed_frames} frames.")
            interval = max(1, floor(len(atom) / needed_frames))  # 确保 interval 至少为 1
            atom = atom[::interval]
            atoms.extend(atom)

        assert atoms is not None

        if len(atoms) < init_frames:
            dlog.warning(f"Not enough frames to extract {init_frames} frames.")
        elif len(atoms) > init_frames:
            # dlog.warning(f"Too many frames to extract {init_frames} frames.")
            # raise ValueError(f"Not enough frames to extract {init_frames} frames.")
            atoms = atoms[:init_frames]
        else:
            dlog.info(f"Extracted {init_frames} frames from {fnumber} files.")

        write_extxyz("init/init.xyz", atoms)
        sampling_cfg = self._sampling_config(self.idata)
        init_descriptor = sampling_cfg.get("init_descriptor", sampling_cfg.get("dataset_descriptor", "nep"))
        train, test = split_train_test_structures(
            atoms,
            training_ratio=training_ratio,
            method=sampling_cfg.get("init_method", sampling_cfg.get("dataset_method", "fps")),
            descriptor_model=self._resolve_sampling_nep_file(
                sampling_cfg.get("init_nep_file", sampling_cfg.get("dataset_nep_file")),
                stage="init",
            ),
            descriptor_mode=init_descriptor,
            min_dist=sampling_cfg.get("init_min_dist", sampling_cfg.get("dataset_min_dist", 0.0)),
            nep_backend=sampling_cfg.get("init_backend", sampling_cfg.get("dataset_backend", "auto")),
            **self._sampling_pca_plot_kwargs(
                os.path.join(self.work_dir, "init", "init_fps_pca.png"),
                "Init train/test FPS coverage",
            ),
        )
        write_extxyz("init/iter_train.xyz", train)
        write_extxyz("init/iter_test.xyz", test)
        dlog.info("Initial training data extracted")

    def parse_yaml(self,file):
        with open(file, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        from nepactive import _migrate_stable_config
        data = _migrate_stable_config(data)
        if self.shock_run:
            sampling_cfg = self._sampling_general_config(data)
            current_sampling_cfg = self._sampling_general_config(self.idata)
            if os.path.exists(f"{os.path.dirname(file)}/nostrucnum"):
                data.setdefault("init", {})["struc_num"] = 0
                data["structure_files"] = self.idata.get("structure_files",["POSCAR"])
                sampling_cfg["structure_id"] = current_sampling_cfg.get("structure_id", [[0]])
                sampling_cfg["ensembles"] = current_sampling_cfg.get("ensembles", sampling_cfg.get("ensembles", ["nphugo_scr", "nvt"]))
                sampling_cfg["temperature"] = current_sampling_cfg.get("temperature", sampling_cfg.get("temperature", [3000]))
                sampling_cfg["structure_id"] = [[0],[0]]
            else:
                data.setdefault("init", {})["struc_num"] = 1
                data["structure_files"] = ["POSCAR","init/struc.000/structure/POSCAR"]
                sampling_cfg["ensembles"] = current_sampling_cfg.get("ensembles", sampling_cfg.get("ensembles", ["nphugo_scr", "nvt"]))
                sampling_cfg["temperature"] = current_sampling_cfg.get("temperature", sampling_cfg.get("temperature", [3000]))
                sampling_cfg["structure_id"] = [[0,1],[0,1]]
        else:
            self.idata.setdefault("init", {})["struc_num"] = 0
        return data

    def make_loop_train(self):
        '''
        make loop training task
        '''

        os.chdir(self.work_dir)

        record = "record.nep"
        iter_rec = [0, -1]
        if os.path.isfile(record):
            with open(record) as frec:
                for line in frec:
                    iter_rec = [int(x) for x in line.split()]
        cont = True
        self.ii = -1
        numb_task = 9
        max_tasks = 10000
        self.restart_total_time = None

        try:
            while True:
                self.ii += 1
                if self.ii < iter_rec[0]:
                    continue
                self.iter_dir = os.path.abspath(os.path.join(self.work_dir,f"iter.{self.ii:06d}")) #the work path has been changed
                os.makedirs(self.iter_dir, exist_ok=True)
                iter_name = f"iter.{self.ii:06d}"
                self.restart_gpumd = False

                for jj in range(numb_task):
                    if not self.restart_gpumd:
                        self.jj = jj
                    yaml_synchro = self.idata.get("yaml_synchro", False)
                    if yaml_synchro:
                        dlog.info(f"yaml_synchro is True, reread the in.yaml from {self.work_dir}/in.yaml")
                        self.idata:dict = self.parse_yaml(f"{self.work_dir}/in.yaml")
                    if self.ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1] and not self.restart_gpumd:
                        continue
                    task_name = "task %02d" % jj
                    sepline(f"{iter_name} {task_name}", "-")
                    if jj == 0:
                        self.make_nep_train()
                    elif jj == 1:
                        self.run_nep_train()
                    elif jj == 2:
                        self.post_nep_train()
                    elif jj == 3:
                        self.make_sampling()
                    elif jj == 4:
                        self.run_model_devi()
                    elif jj == 5:
                        self.post_sampling_run()
                    elif jj == 6:
                        self.make_label_task()
                    elif jj == 7:
                        self.run_label_task()
                    elif jj == 8:
                        self.post_label_task()
                    else:
                        raise RuntimeError("unknown task %d, something wrong" % jj)

                    os.chdir(self.work_dir)
                    record_iter(record, self.ii, jj)

        except TrainingCompletedException as e:
            dlog.info(f"Training completed: {e}")
            if e.accuracy and e.error:
                dlog.info(f"Target accuracy: {e.accuracy}%, achieved error: {e.error:.2f}%")
            return

        except KeyboardInterrupt:
            dlog.warning("Training interrupted by user")
            raise

    def shock(self):
        work_dir = os.getcwd()
        shock_raw = self.idata.get("shock", None)
        if shock_raw is None:
            raise ValueError("shock data is None, please check your in.yaml")
        shock_data = copy.deepcopy(shock_raw)
        rhos = self.idata.get("rhos", None)
        if rhos is None:
            rhos = shock_data.get("rhos", [1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

        # `nepactive shock` 使用 in.yaml 的 shock.pot；仅在 pot=nep 时补默认 nep 路径
        pot = shock_data.get("pot", "nep")
        shock_data["pot"] = pot
        if pot == "nep" and not shock_data.get("nep", None):
            shock_data["nep"] = os.path.join(os.path.abspath(os.getcwd()), "nep.txt")
        struture_files = os.path.join(os.path.abspath(os.getcwd()), "POSCAR")
        shock_data["structure_files"] = [struture_files]

        shock_data["python_interpreter"] = self.idata.get("python_interpreter", "python")
        shock_data["task_per_gpu"] = self.idata.get("task_per_gpu", 1)
        shock_data["gpu_available"] = self.idata.get("gpu_available", [0,1,2,3])

        if not os.path.isfile("properties.txt"):
            raise ValueError("properties.txt is not found, please check your in.yaml")

        for rho in rhos:
            os.chdir(self.work_dir)
            os.makedirs(f"{rho}",exist_ok=True)
            os.chdir(f"{rho}")
            shock_data["rho"] = rho
            dlog.info(f"Running shock velocity test for rho={rho}")
            os.system(f"ln -snf {self.work_dir}/POSCAR POSCAR")
            os.system(f"ln -snf {self.work_dir}/properties.txt properties.txt")
            shock_task = ShockRun(self.idata, shock_data)
            shock_task.run()
            dlog.info(f"Shock velocity test for rho={rho} completed")
            with open(f"{self.work_dir}/shock_vel.txt", "a") as f:
                np.savetxt(f, np.array(shock_task.shock_vels), fmt='%.3f', header='Shock velocities (km/s) for each rho')


    def shock_vel_test(self):

        work_dir = os.path.abspath(os.path.join(self.iter_dir, "03.shock"))
        os.makedirs(work_dir, exist_ok=True)
        atoms = read(f"{self.work_dir}/POSCAR")
        write(f"{work_dir}/POSCAR", atoms)
        os.system(f"ln -snf {self.work_dir}/init/properties.txt {work_dir}/properties.txt")
        os.chdir(work_dir)
        shock_data = self.idata.get("shock", None)
        assert shock_data is not None, "shock data is None"
        nep_file = os.path.join(self.iter_dir, "00.nep/task.000000/nep.txt")
        shock_data["nep"] = nep_file
        shock_data["pot"] = "nep"
        original_make = shock_data.get("original_make", False)

        shock_data["python_interpreter"] = self.idata.get("python_interpreter", "python")
        shock_data["task_per_gpu"] = self.idata.get("task_per_gpu", 1)
        shock_data["gpu_available"] = self.idata.get("gpu_available", [0,1,2,3])

        if original_make:
            structure_files = [os.path.abspath(self.idata.get("structure_files")[0])]
        else:
            structure_files = []
        sampling_dir = self._sampling_dir(self.ii, prefer_existing=True)
        final_xyzs = glob(f"{sampling_dir}/task.[0-9][0-9][0-9][0-9][0-9][0-9]/final.xyz")
        final_xyzs.sort()
        if not final_xyzs:
            raise FileNotFoundError(f"No final.xyz found under {sampling_dir}/task.*")
        final_xyz = final_xyzs[-1]
        structure_files.append(final_xyz)
        shock_data["structure_files"] = structure_files
        rho = self.idata.get("rho", None)
        if rho:
            dlog.info(f"rho is {rho}, will run shock velocity test for rho={rho}")
            shock_data["rho"] = rho

        shock_task = ShockRun(self.idata, shock_data)
        shock_task.run()

        # 计算爆热
        from nepactive.hod import calculate_heat_of_detonation
        dlog.info("Calculating heat of detonation...")
        try:
            gpu_id = shock_data["gpu_available"][0] if shock_data["gpu_available"] else 0
            job_system = self.idata.get("job_system", None)
            Q_release = calculate_heat_of_detonation(work_dir, nep_file, gpu_id, job_system)
            dlog.info(f"Heat of detonation: {Q_release:.2f} kJ/kg")
        except Exception as e:
            dlog.error(f"Failed to calculate heat of detonation: {e}")
            Q_release = None

        # 保存结果到 CSV 格式
        import csv
        shock_vel_csv = f"{self.work_dir}/shock_vel.csv"

        # shock_vels 形状 (N, 4): D_v, V_CJ, P_CJ, rho
        sv = np.array(shock_task.shock_vels)
        if sv.ndim == 1:
            sv = sv.reshape(1, -1)
        dv_mean = np.mean(sv[:, 0]) if len(sv) > 0 else 0
        vcj_mean = np.mean(sv[:, 1]) if sv.shape[1] > 1 else 0
        pcj_mean = np.mean(sv[:, 2]) if sv.shape[1] > 2 else 0
        rho_mean = np.mean(sv[:, 3]) if sv.shape[1] > 3 else 0

        file_exists = os.path.exists(shock_vel_csv)
        with open(shock_vel_csv, "a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Iteration", "Dv_km/s", "V_CJ", "P_CJ_GPa", "rho_g/cm3", "HOD_kJ/kg"])
            hod_csv = f"{Q_release:.2f}" if Q_release else "N/A"
            writer.writerow([f"iter.{self.ii:06d}", f"{dv_mean:.3f}", f"{vcj_mean:.4f}",
                             f"{pcj_mean:.2f}", f"{rho_mean:.3f}", hod_csv])

        # 保存到 txt 格式（格式化对齐，方便 vi 查看）
        shock_vel_txt = f"{self.work_dir}/shock_vel.txt"
        if not os.path.exists(shock_vel_txt):
            with open(shock_vel_txt, "w") as f:
                f.write("="*100 + "\n")
                f.write("Shock Velocity and Detonation Results\n")
                f.write("="*100 + "\n")
                f.write(f"{'Iteration':<15} {'Dv(km/s)':<14} {'V_CJ':<14} {'P_CJ(GPa)':<14} {'rho(g/cm3)':<14} {'HOD(kJ/kg)':<14}\n")
                f.write("-"*100 + "\n")

        with open(shock_vel_txt, "a") as f:
            hod_str = f"{Q_release:.2f}" if Q_release else "N/A"
            iter_tag = f"iter.{self.ii:06d}"
            f.write(f"{iter_tag:<15} {dv_mean:<14.3f} {vcj_mean:<14.4f} {pcj_mean:<14.2f} {rho_mean:<14.3f} {hod_str:<14}\n")

        dlog.info(f"Shock velocity test completed, results: {shock_task.shock_vels} km/s")
        if Q_release:
            dlog.info(f"Heat of detonation: {Q_release:.2f} kJ/kg")

    def run_nep_train(self):
        '''
        run nep training
        '''
        train_steps = self.idata.get("train_steps", 10000)
        work_dir = os.path.abspath(os.path.join(self.iter_dir, "00.nep"))
        pot_num = self.idata.get("pot_num", self.idata.get("pot_number", 4))
        pot_inherit: bool = self.idata.get("pot_inherit", True)
        self.gpu_available = self.idata.get("gpu_available")
        if not self.gpu_available:
            raise ValueError("gpu_available is missing or empty in in.yaml")
        processes = []
        log_files = []  # ✅ 添加log_files列表
        
        if not pot_inherit:
            dlog.info(f"{os.getcwd()}")
            dlog.info(f"pot_inherit is false, will remove old task files {work_dir}/task*")
            os.system(f"rm -r {work_dir}/task.*")
        
        try:
            for jj in range(pot_num):
                # ensure the work_dir is the absolute path
                task_dir = os.path.join(work_dir, f"task.{jj:06d}")
                os.makedirs(task_dir, exist_ok=True)
                os.chdir(task_dir)
                
                # preparation files
                if not os.path.isfile("train.xyz"):
                    os.symlink("../dataset/train.xyz", "train.xyz")
                if not os.path.isfile("test.xyz"):
                    os.symlink("../dataset/test.xyz", "test.xyz")
                if self.ii == 0:
                    ini_train_steps = self.idata.get("ini_train_steps", 10000)
                    nep_in = self._build_nep_in_content(
                        jj,
                        pot_inherit=pot_inherit,
                        train_steps=ini_train_steps,
                    )
                else:
                    nep_in = self._build_nep_in_content(
                        jj,
                        pot_inherit=pot_inherit,
                        train_steps=train_steps,
                    )
                with open("nep.in", "w", encoding="utf-8") as f:
                    f.write(nep_in)
                
                if pot_inherit and self.ii > 0:
                    nep_restart = f"{self.work_dir}/iter.{self.ii-1:06d}/00.nep/task.{jj:06d}/nep.restart"
                    dlog.info(f"pot_inherit is true, will copy nep.restart from {nep_restart}")
                    shutil.copy(nep_restart, "nep.restart")
                
                log_file_path = os.path.join(task_dir, 'log')
                env = os.environ.copy()
                gpu_id = self.gpu_available[jj % len(self.gpu_available)]
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                
                # ✅ 不使用with,手动打开
                log = open(log_file_path, 'w')
                log_files.append(log)  # ✅ 保存文件对象
                
                process = subprocess.Popen(
                    ["nep"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env
                )
                processes.append((process, log_file_path))  # ✅ 保存进程和文件路径
                dlog.info(f"submitted task.{jj:06d} to GPU:{gpu_id}")
                
                self.gpu_numbers = len(self.gpu_available)  # ✅ 修正:使用self.gpu_available
                
                # ✅ 每提交gpu_numbers个任务后,等待这一批完成
                if (jj + 1) % self.gpu_numbers == 0:
                    dlog.info(f"jobs submitted, checking status of batch ending at task.{jj:06d}")
                    for process, log_file_path in processes:
                        process.wait()
                        if process.returncode != 0:
                            dlog.error(f"Process failed. Check the log at: {log_file_path}")
                            raise RuntimeError(f"One or more processes failed. Check the log file:({log_file_path}) for details.")
                        else:
                            dlog.info(f"Process completed successfully. Log saved at: {log_file_path}")
                    
                    # ✅ 这一批完成后清空列表,准备下一批
                    processes = []
            
            # ✅ 处理最后不满一批的任务
            if processes:
                dlog.info(f"checking remaining jobs")
                for process, log_file_path in processes:
                    process.wait()
                    if process.returncode != 0:
                        dlog.error(f"Process failed. Check the log at: {log_file_path}")
                        raise RuntimeError(f"One or more processes failed. Check the log file:({log_file_path}) for details.")
                    else:
                        dlog.info(f"Process completed successfully. Log saved at: {log_file_path}")

        finally:
            # Close all log files with safety checks
            for log in log_files:
                if log and not log.closed:
                    try:
                        log.flush()  # Ensure buffer is written
                        log.close()
                    except Exception as e:
                        dlog.error(f"Failed to close log file: {e}")
    # def run_nep_train(self):
    #     '''
    #     run nep training
    #     '''
    #     train_steps = self.idata.get("train_steps", 10000)
    #     work_dir = os.path.abspath(os.path.join(self.iter_dir, "00.nep"))
    #     pot_num = self.idata.get("pot_num", 4)
    #     pot_inherit:bool = self.idata.get("pot_inherit", True)
    #     # nep_template = os.path.abspath(self.idata.get("nep_template"))
    #     processes = []
    #     if not pot_inherit:
    #         dlog.info(f"{os.getcwd()}")
    #         dlog.info(f"pot_inherit is false, will remove old task files {work_dir}/task*")
    #         os.system(f"rm -r {work_dir}/task.*")
    #     for jj in range(pot_num):
    #         #ensure the work_dir is the absolute path
    #         task_dir = os.path.join(work_dir, f"task.{jj:06d}")
    #         os.makedirs(task_dir,exist_ok=True)
    #         #     absworkdir/iter.000000/00.nep/task.000000/
    #         os.chdir(task_dir)
    #         #preparation files
    #         if not os.path.isfile("train.xyz"):
    #             os.symlink("../dataset/train.xyz","train.xyz")
    #         if not os.path.isfile("test.xyz"):
    #             os.symlink("../dataset/test.xyz","test.xyz")
    #         if not os.path.isfile("nep.in"):
    #             nep_in_header = self.idata.get("nep_in_header", "type 4 H C N O")
    #             if self.ii == 0:
    #                 ini_train_steps = self.idata.get("ini_train_steps", 10000)
    #                 nep_in = nep_in_template.format(train_steps=ini_train_steps,nep_in_header=nep_in_header)
    #             else:
    #                 nep_in = nep_in_template.format(train_steps=train_steps,nep_in_header=nep_in_header)
    #             with open("nep.in", "w") as f:
    #                 f.write(nep_in)
    #             # os.symlink(nep_template, "nep.in")
    #         if pot_inherit and self.ii > 0:
    #             nep_restart = f"{self.work_dir}/iter.{self.ii-1:06d}/00.nep/task.{jj:06d}/nep.restart"
    #             dlog.info(f"pot_inherit is true, will copy nep.restart from {nep_restart}")
    #             shutil.copy(nep_restart, "nep.restart")
    #             # exit()
    #         log_file = os.path.join(task_dir, 'log')  # Log file path
    #         env = os.environ.copy()
    #         gpu_id = self.gpu_available[jj%len(self.gpu_available)]
    #         env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    #         # env['CUDA_VISIBLE_DEVICES'] = str(jj)
    #         with open(log_file, 'w') as log:
    #             process = subprocess.Popen(
    #                 ["nep"],  # 程序名
    #                 stdout=log,
    #                 stderr=subprocess.STDOUT,
    #                 env=env  # 使用修改后的环境
    #             )
    #             processes.append((process, log_file))
            
    #         self.gpu_numbers = len(self.idata.get("gpu_available"))            
    #         if (jj+1)%self.gpu_numbers == 0:
    #             dlog.info(f"jobs submitted, checking status of {self.jj}")
    #             for process, log_file in processes:
    #                 process.wait()  # Wait for all processes to complete
    #                 # Check for errors using the return code
    #                 if process.returncode != 0:
    #                     dlog.error(f"Process failed. Check the log at: {log_file}")
    #                     raise RuntimeError(f"One or more processes failed. Check the log file:({log_file}) for details.")
    #                 else:
    #                     dlog.info(f"Process completed successfully. Log saved at: {log_file}")

    def post_label_task(self):
        '''
        Post-process label task and check convergence.
        '''
        if not self.shock_run:
            dlog.info("shock_run is False, skip shock velocity test")
            return
        test_interval = self.idata.get("shock_test_interval", 1)
        test_begin_step = self.idata.get("shock_test_begin_step", 400000)

        self.run_steps = int(np.loadtxt(f"{self.work_dir}/steps.txt",ndmin=1,encoding="utf-8")[-1])
        if (self.run_steps > test_begin_step) and (self.ii%test_interval == 0):
            dlog.info(f"run_steps is {self.run_steps}, will run shock velocity test")
            self.shock_vel_test()

        shock_vel_csv = f"{self.work_dir}/shock_vel.csv"
        if os.path.exists(shock_vel_csv):
            import csv
            with open(shock_vel_csv, "r") as _f:
                reader = csv.reader(_f)
                header = next(reader, None)
                rows = []
                for row in reader:
                    try:
                        # CSV: Iteration, Dv, V_CJ, P_CJ, rho, HOD
                        rows.append([float(row[1])])
                    except (ValueError, IndexError):
                        continue
            shock_data = np.array(rows) if rows else np.empty((0, 1))
            if len(shock_data) > 3:
                shock_values = shock_data[-3:, 0]
                shock_mean = shock_values.mean()

                # Guard against division by zero
                if shock_mean == 0:
                    dlog.warning(f"Mean shock velocity is zero, skipping error calculation. Values: {shock_values}")
                    return

                Dv_error = np.abs(shock_values.max() - shock_values.min()) / shock_mean * 100
                dlog.info(f"Shock velocity error is {Dv_error:.2f}%")
                accuracy = self.idata.get("accuracy", 1)
                if Dv_error < accuracy:
                    dlog.info(f"Reach accuracy {accuracy}%, job finished successfully, will stop training process")
                    fs = glob(f"{self.work_dir}/iter.{self.ii:06d}/03.shock/**/shock_vel.png",recursive=True)
                    if fs:
                        fs.sort()
                        shutil.copy2(fs[-1], f"{self.work_dir}/shock_vel.png")
                    # Raise exception for clean shutdown instead of exit()
                    raise TrainingCompletedException(
                        "Training completed successfully",
                        accuracy=accuracy,
                        error=Dv_error
                    )

    def post_nep_train(self):
        '''
        post nep training
        '''
        os.chdir(f"{self.work_dir}/iter.{self.ii:06d}/00.nep")
        tasks = glob("task.*")
        nep_plot = self.idata.get("nep_plot", True)
        for task in tasks:
            if nep_plot:
                os.chdir(f"{self.work_dir}/iter.{self.ii:06d}/00.nep")
                os.chdir(task)
                if not os.path.exists("loss.png"):
                    nep_plt()

        os.chdir(f"{self.work_dir}/iter.{self.ii:06d}/00.nep")
        # nep_plt(testplt=False)
        os.system("rm dataset/*xyz */*train.out */*test.out")

    def make_nep_train(self):
        '''
        Train nep. 
        '''
        #ensure the work_dir is the absolute pathk
        global_work_dir = os.path.abspath(self.work_dir)
        work_dir = os.path.abspath(os.path.join(self.iter_dir, "00.nep"))
        pot_num = self.idata.get("pot_num", self.idata.get("pot_number", 4))
        use_init_data:bool = self.idata.get("use_init_data", True)

        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        #     absworkdir/iter.000000/00.nep/


        #preparation files
        if pot_num > 4:
            raise ValueError(f"pot_num should be no bigger than 4, and it is now {pot_num}")
        
        #make training data preparation
        os.makedirs("dataset", exist_ok=True)
        def merge_files(input_files, output_file):
            command = ['cat'] + input_files  # 适用于类Unix系统
            with open(output_file, 'wb') as outfile:
                subprocess.run(command, stdout=outfile)
        files = []
        testfiles = []
        # 注意每一代有没有划分训练集和测试集
        if use_init_data == True:
            print(f"{os.path.isfile(os.path.join(global_work_dir, 'init/iter_train.xyz'))}")
            # 直接调用 extend 方法，不要尝试将其结果赋值
            files.extend(glob(os.path.join(global_work_dir, "init/iter_train.xyz")))
            testfiles.extend(glob(os.path.join(global_work_dir, "init/iter_test.xyz")))

            # 检查文件列表是否为空
            # if not files:
            #     raise ValueError("No files found to merge.")
        """等于之后反而出错了"""

        if self.ii > 0:
            for iii in range(self.ii):
                newtrainfile = glob(f"../../iter.{iii:06d}/02.label/iter_train.xyz")
                files.extend(newtrainfile)
                newtestfile = glob(f"../../iter.{iii:06d}/02.label/iter_test.xyz")
                testfiles.extend(newtestfile)
                if (not newtrainfile) or (not newtestfile):
                    dlog.warning(f"iter.{iii:06d} has no training or test data")
        if not files:
            raise ValueError("No files found to merge.")
            # dlog.error("No files found to merge.")
        merge_files(files, "dataset/train.xyz")
        merge_files(testfiles, "dataset/test.xyz")
        self.gpu_available = self.idata.get("gpu_available")



        # work_dir = os.path.abspath(f"{work_dir}/task.{ii:06d}")

    def make_model_devi(self):
        '''
        run gpumd, this function is referenced by make_loop_train
        '''

        model_devi = self.get_model_devi()

        if self.make_gpumd_task_first:
            # 日志记录
            dlog.info(f"make_gpumd_task_first is true, will backup old task directory {self.work_dir}/iter.{self.ii:06d}/01.gpumd")

            # 查找已有的备份文件夹
            bak_files = glob(f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd.bak.*")
            if bak_files:
                # 提取最大后缀数字
                suffixes = [int(os.path.basename(f).split('.')[-1]) for f in bak_files]
                new_suffix = max(suffixes) + 1
            else:
                # 如果没有备份文件夹，默认后缀为 0
                new_suffix = 0

            # 构造新备份路径
            src_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
            dst_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd.bak.{new_suffix}"

            # 重命名文件夹
            if os.path.exists(src_dir):
                shutil.move(src_dir, dst_dir)
                dlog.info(f"Backup completed: {src_dir} -> {dst_dir}")
            else:
                dlog.warning(f"Source directory does not exist: {src_dir}")


        work_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
        structure_files:list = self.idata.get("structure_files")
        structure_prefix = self.idata.get("structure_prefix",self.work_dir)
        sampling_cfg = self._sampling_config(self.idata)
        time_step_general = sampling_cfg.get("time_step", None)
        structure_files = [os.path.join(structure_prefix,structure_file) for structure_file in structure_files]
        nep_file = self.idata.get("nep_file","../../00.nep/task.000000/nep.txt")
        needed_frames = sampling_cfg.get("needed_frames",10000)
        self.run_steps_factor = self.idata.get("run_steps_factor",1.5)
        if self.ii == 0:
            run_steps = self.idata.get("ini_run_steps",100000)
        else:
            all_run_steps = np.loadtxt(f"{self.work_dir}/steps.txt",ndmin=1,encoding="utf-8")
            old_run_steps = all_run_steps[-1]
            run_steps = int(self.run_steps_factor*int(old_run_steps))
            if len(all_run_steps) > 1:
                if all_run_steps[-2] > all_run_steps[-1]:
                    dlog.warning(f"The older run_steps is {old_run_steps}, the new run_steps is {all_run_steps[-1]},and the older one is bigger")
        run_steps = self._cap_run_steps(run_steps, f"iter.{self.ii:06d} sampling setup")
        self.run_steps = run_steps
        dlog.info(f"run_steps is {run_steps}")
        # if self.run_steps < 10000:

        if os.path.exists(f"{self.work_dir}/init/properties.txt"):
            property = np.loadtxt(f"{self.work_dir}/init/properties.txt",encoding="utf-8")
            o_rho = property[0]
            v0 = [property[3]]
            e0 = [property[1]]
            p0 = [property[2]]
            real_p0 = self.idata.get("real_p0",False)
            if not real_p0:
                dlog.info(f"real_p0 is {real_p0}, will set p0 to 0")
                p0 = [0]
            rho = self.idata.get("rho")
            if rho is None:
                rho = o_rho
                dlog.info(f"rho is not set, keep original density {o_rho}")
            elif rho != o_rho:
                dlog.info(f"rho is {rho}, o_rho is {o_rho}, will scale v0")
                v0 = [property[3] * o_rho / rho]  # scale v0 according to rho
        else:
            e0 = [0]
            p0 = [0]
            v0 = [0]

        pperiod = model_devi.get("pperiod", 2000)
        self.total_time, self.model_devi_task_numbers = Nepactive.make_gpumd_task(model_devi=model_devi, structure_files=structure_files, needed_frames=needed_frames,time_step_general=time_step_general,
                                                                                   work_dir=work_dir,nep_file=nep_file, run_steps=run_steps,e0=e0, p0=p0,v0 =v0, pperiod=pperiod)

    def make_sampling(self):
        return self.make_model_devi()

    @classmethod
    def make_gpumd_task(cls, model_devi:dict, structure_files, needed_frames=10000, time_step_general=0.2, work_dir:str=None, 
                        nep_file:str=None, run_steps:int=20000, e0=[0], p0=[0], v0=[0],pperiod = 1000):
        if not work_dir:
            work_dir = os.getcwd()
        if not nep_file:
            nep_file = f"{work_dir}/nep.txt"


        if not (e0 and p0 and v0):
            e0 = model_devi.get("e0",None)
            p0 = model_devi.get("p0",None)
            v0 = model_devi.get("v0",None)

        assert e0 and p0 and v0, "e0, p0, v0 are not empty set"

        task_dicts = []
        ensembles = model_devi.get("ensembles",["nphugo_scr"])
        replicate_cell = model_devi.get("replicate_cell","1 1 1")
        nums = list(map(int, replicate_cell.split()))
        mult_power = np.prod(nums)
        for ensemble_index,ensemble in enumerate(ensembles):

            structure_id = model_devi.get("structure_id",[[0]])[ensemble_index]
            all_dict = {}
            assert structure_files is not None
            structure = [structure_files[ii] for ii in structure_id] #####################################################
            all_dict["structure"] = structure
            time_step = model_devi.get("time_step")
            # print("ensembles =", ensembles)
            # print("structure_id raw =", model_devi.get("structure_id", [[0]]))
            if not time_step:
                time_step = time_step_general

            assert all([structure,time_step,run_steps]), "有变量为空"
            # task_dict = {}
            all_dict["pperiod"] = [pperiod]
            if ensemble in ["nvt", "npt", "npt_scr", "nphugo_mttk", "nphugo_scr"]:
                temperature = model_devi.get("temperature") #
                if ensemble in ["npt", "npt_scr", "nvt"]:
                    assert temperature is not None
                    all_dict["temperature"] = temperature
                if ensemble in ["npt", "npt_scr", "nphugo_mttk", "nphugo_scr"]:
                    pressure = model_devi.get("pressure") #
                    all_dict["pressure"] = pressure
                    assert pressure is not None
                if ensemble == "npt_scr":
                    all_dict["tau_t"] = [model_devi.get("tau_t", 100)]
                    all_dict["tau_p"] = [pperiod]
                    all_dict["elastic_modulus"] = [model_devi.get("elastic_modulus", 15.0)]
                if ensemble in ["nphugo_mttk", "nphugo_scr"]:
                    all_dict["e0"] = e0 * mult_power
                    all_dict["p0"] = p0
                    all_dict["v0"] = v0 * mult_power
            elif ensemble == "msst":
                v_shock:list = model_devi.get("v_shock",[9.0])       #
                qmass:list = model_devi.get("qmass",[100000])           #
                viscosity:list = model_devi.get("viscosity",[10])   #
                shock_direction = model_devi.get("shock_direction","y")
                all_dict["v_shock"] = v_shock
                all_dict["qmass"] = qmass
                all_dict["viscosity"] = viscosity
                all_dict["shock_direction"] = shock_direction
            else:
                raise NotImplementedError
            dlog.info(f"all_dict:{all_dict}")
            assert all(v not in [None, '', [], {}, set()] for v in all_dict.values())
            # task_para = []
            # combo_numbers = 0
            for combo in itertools.product(*all_dict.values()):
                # 将每一组合生成字典，字典的键是列表的变量名，值是组合中的对应元素
                combo_dict = {keys: combo[index] for index, keys in enumerate(all_dict.keys())}
                task_dicts.append((ensemble,combo_dict))
            dlog.info(f"{all_dict.keys()} generate {len(task_dicts)} tasks")
            frames_pertask = needed_frames/len(task_dicts)
            dump_freq = max(1,floor(run_steps/frames_pertask))

            model_devi_task_numbers = len(task_dicts)

            # nep_file = self.idata.get("nep_file","../../00.nep/task.000000/nep.txt")

        index = 0
        for ensemble,task in task_dicts:
            if ensemble == "msst":
                assert run_steps > 20000
                text = msst_template.format(time_step = time_step,run_steps = run_steps-20000,dump_freq = dump_freq, replicate_cell = replicate_cell, **task)
            elif ensemble == "nvt":
                text = nvt_template.format(time_step = time_step,run_steps = run_steps,dump_freq = dump_freq, replicate_cell = replicate_cell, **task)
            elif ensemble == "npt":
                text = npt_template.format(time_step = time_step,run_steps = run_steps,dump_freq = dump_freq, replicate_cell = replicate_cell, **task)
            elif ensemble == "npt_scr":
                text = npt_scr_template.format(time_step = time_step, run_steps = run_steps, dump_freq = dump_freq, replicate_cell = replicate_cell, **task)
            elif ensemble in ["nphugo_mttk", "nphugo_scr"]:
                assert run_steps > 20000
                text = nphugo_mttk_template.format(
                    time_step = time_step,
                    run_steps = run_steps-20000,
                    dump_freq = dump_freq,
                    replicate_cell = replicate_cell,
                    **task
                )
                # dlog.info(f"nphugo task:{task},npugo text:{text}")
            else:
                raise NotImplementedError(f"The ensemble {ensemble} is not supported")
            task_dir = f"{work_dir}/task.{index:06d}"
            file = f"{task_dir}/run.in"
            os.makedirs(task_dir,exist_ok=True)
            os.chdir(task_dir)
            structure:str = task["structure"]
            if not structure.endswith("xyz"):
                atom = read(structure)
                atom = write("POSCAR",atom)
                atom = read("POSCAR")
                write_extxyz(f"model.xyz",atom)####################
            else:
                shutil.copy(structure,f"model.xyz")
            with open(file,mode='w') as f:
                f.write(text)
            if not os.path.isfile("nep.txt"):
                os.symlink(nep_file, "nep.txt")
            index += 1
        total_time = run_steps*time_step

        # dlog.info("generate gpumd task done")
        return total_time, model_devi_task_numbers

    def run_model_devi(self):
        # dlog.info("entering the run_gpumd_task")
        try:
            self.dump_freq
        except AttributeError:
            # 捕获变量不存在时的错误
            self.make_gpumd_task_first = False
            self.make_model_devi()
            self.make_gpumd_task_first = True
            dlog.info("remake the gpumd task")
        model_devi_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
        os.chdir(model_devi_dir)
        gpu_available = self.idata.get("gpu_available")
        if not gpu_available:
            raise ValueError("gpu_available is missing or empty in in.yaml")
        self.task_per_gpu = self.idata.get("task_per_gpu", 1)

        Nepactive.run_gpumd_task(work_dir=model_devi_dir, gpu_available=gpu_available, task_per_gpu=self.task_per_gpu)
        # self.write_steps()

    @classmethod
    def run_gpumd_task(cls,work_dir:str=None,gpu_available:List[int]=None,task_per_gpu:int=1):
        run_gpumd_task(work_dir=work_dir, gpu_available=gpu_available, task_per_gpu=task_per_gpu)

        
    def get_model_devi(self):
        '''
        get the model deviation from the gpumd run
        '''
        model_devi = self._sampling_general_config(self.idata)
        if not model_devi:
            raise ValueError("sampling.general is empty in config")
        return model_devi

    def post_gpumd_run(self):
        '''
        优化版本：专注于循环和I/O瓶颈优化
        '''
        try:
            self.total_time
        except AttributeError:
            self.make_gpumd_task_first = False
            self.make_model_devi()
            self.make_gpumd_task_first = True
            dlog.info("remake the gpumd task")

        # 预先获取所有配置参数，避免重复调用
        config = self._get_cached_config()
        nep_dir = os.path.join(self.iter_dir, "00.nep")
        
        # 绘图处理（如果需要，可以考虑跳过或异步处理）
        if config['plot']:
            self._handle_plotting_optimized(config)

        self.gpumd_dir = os.path.join(self.iter_dir, "01.gpumd")
        
        # 预先获取和排序所有任务目录，避免多次调用
        task_dirs = self._get_cached_task_dirs()
        
        dlog.info(f"-----start analysis the trajectory-----"
                f"Processing {len(task_dirs)} tasks with config: {config['summary']}")

        # 核心优化：减少循环内的重复计算和I/O
        results = self._optimized_task_processing(task_dirs, nep_dir, config)
        
        # 快速检查失败任务
        if self._quick_failure_check(results, config):
            return
        
        # 批量输出处理
        self._batch_output_processing(results, config)
        
        # 更新步数和状态
        self._finalize_iteration(config)

    def post_sampling_run(self):
        return self.post_gpumd_run()

    def _get_cached_config(self):
        """一次性获取所有配置，避免重复查询"""
        model_devi = self.get_model_devi()
        sampling_cfg = self._sampling_config(self.idata)
        threshold = model_devi.get("uncertainty_threshold") or sampling_cfg.get("uncertainty_threshold", [0.3, 1])
        energy_threshold = sampling_cfg.get("energy_threshold", float("inf"))
        if energy_threshold is None:
            energy_threshold = float("inf")
        
        config = {
            'plot': self.idata.get("gpumd_plt", True),
            'threshold': threshold,
            'energy_threshold': energy_threshold,
            'mode': sampling_cfg.get("uncertainty_mode", "mean"),
            'level': sampling_cfg.get("uncertainty_level", 1),
            'deviation_backend': sampling_cfg.get("deviation_backend", "auto"),
            'enable_molecule_analysis': sampling_cfg.get("enable_molecule_analysis", False),
            'molecule_analysis_backend': sampling_cfg.get("molecule_analysis_backend", "ase"),
            'sample_method': sampling_cfg.get("method", "relative"),
            'continue_from_old': self.idata.get("continue_from_old", False),
            'max_candidate': sampling_cfg.get("max_candidate", 1000),
            'max_temp': self.idata.get("max_temp", 10000),
            'shortest_d': self.idata.get("shortest_d", 0.5),
            'analyze_range': self.idata.get("analyze_range", [0.5, 1.0]),
            'max_run_steps': self.idata.get("max_run_steps", 1200000),
            'max_iter': self.idata.get("max_iter", 40),
            'time_step': self.idata.get("time_step")
        }
        
        # 预计算一些常用值
        task_count = len(self._get_cached_task_dirs())
        if task_count == 0:
            raise ValueError(f"No GPUMD task dirs found under {self.work_dir}/iter.{self.ii:06d}/01.gpumd")
        config['max_candidate_per_task'] = max(1, min(config['max_candidate'], 2 * ceil(config['max_candidate'] / task_count)))
        config['summary'] = f"threshold:{threshold}, energy_threshold:{energy_threshold}, max_temp:{config['max_temp']}"
        dlog.info(f"Config summary: {config['summary']}")
        return config

    def _get_cached_task_dirs(self):
        """缓存任务目录列表"""
        model_devi_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
        self._cached_task_dirs = sorted([
            os.path.join(model_devi_dir, task) 
            for task in glob(f"{model_devi_dir}/task.[0-9][0-9][0-9][0-9][0-9][0-9]")
        ])
        return self._cached_task_dirs

    def _handle_plotting_optimized(self, config):
        """优化绘图处理，可选择跳过或异步"""
        # 如果绘图不重要，可以考虑跳过以节省时间
            
        model_devi_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
        task_dirs = self._get_cached_task_dirs()
        
        current_dir = os.getcwd()
        try:
            dlog.info(f"plotting {len(task_dirs)} tasks")
            for task_dir in task_dirs:
                os.chdir(task_dir)
                try:
                    gpumdplt(self.total_time, config['time_step'])
                except Exception as e:
                    dlog.error(f"plotting {task_dir} failed: {e}")
                    # 可以选择继续而不是抛出异常
                    continue
        finally:
            os.chdir(current_dir)

    def _optimized_task_processing(self, task_dirs, nep_dir, config):
        """优化的任务处理循环"""
        results = {
            'failed_indices': [],
            'thermo_averages': [],
            'all_candidates': [],
            'statistics': {'accurate': 0, 'candidate': 0, 'total': 0},
        }

        label_dir = os.path.join(self.iter_dir, "02.label")
        os.makedirs(label_dir, exist_ok=True)
        candidate_file = os.path.join(label_dir, "candidate.xyz")
        if os.path.exists(candidate_file):
            os.remove(candidate_file)

        def make_candidate_condition(frame_prop):
            return (
                (
                    ((frame_prop[:, 1] >= config['threshold'][0]) & (frame_prop[:, 1] <= config['threshold'][1]))
                    | (frame_prop[:, 2] > config['energy_threshold'])
                )
                & (frame_prop[:, 3] > config['shortest_d'])
                & (frame_prop[:, 4] < config['max_temp'])
            )

        def make_fps_condition(frame_prop):
            return (frame_prop[:, 3] > config['shortest_d']) & (frame_prop[:, 4] < config['max_temp'])

        def make_accurate_condition(frame_prop):
            return (frame_prop[:, 1] < config['threshold'][0]) & (frame_prop[:, 2] < config['energy_threshold'])

        fmt = "%14d" + "%12.2f" * 9 + "%12.4f"
        header = f"{'indices':>14} {'time':^14} {'relative_error':^14} {'energy_error':^14} {'shortest_d':^14} {'temperature':^14} {'potential':^14} {'pressure':^14} {'volume':^14} {'molecule_num':^14} {'molecule_density':^14}"

        current_dir = os.getcwd()
        analyze_start = config['analyze_range'][0]
        analyze_end = config['analyze_range'][1]

        candidate_records: list[dict] = []
        open(candidate_file, 'w', encoding='utf-8').close()
        for ii, task_dir in enumerate(task_dirs):
            os.chdir(task_dir)
            dlog.info(f"processing task {ii}")
            try:
                atoms_list, frame_property, failed_row_index = Nepactive.relative_force_error(
                    total_time=self.total_time,
                    nep_dir=nep_dir,
                    mode=config['mode'],
                    level=config['level'],
                    backend=config['deviation_backend'],
                    enable_molecule_analysis=config['enable_molecule_analysis'],
                    molecule_analysis_backend=config['molecule_analysis_backend'],
                    allowed_max_temp=config['max_temp'],
                    allowed_shortest_distance=config['shortest_d'],
                )

                write_extxyz(os.path.join(task_dir, "final.xyz"), atoms_list[-1])

                prop_len = len(frame_property)
                start_idx = int(analyze_start * prop_len)
                end_idx = int(analyze_end * prop_len)
                thermo_avg = np.mean(frame_property[start_idx:end_idx, 1:], axis=0, keepdims=True)

                p_rmse = np.sqrt(np.mean((frame_property[start_idx:end_idx, 6] - thermo_avg[0, 5]) ** 2))
                p_mae = np.mean(np.abs(frame_property[start_idx:end_idx, 6] - thermo_avg[0, 5]))
                thermo_avg = np.hstack((thermo_avg, np.array([[p_rmse, p_mae]])))
                results['thermo_averages'].append(thermo_avg)

                if config['sample_method'] == "fps":
                    candidate_condition = make_fps_condition(frame_property)
                    accurate_count = 0
                else:
                    candidate_condition = make_candidate_condition(frame_property)
                    accurate_condition = make_accurate_condition(frame_property)
                    accurate_count = int(np.sum(accurate_condition))

                candidate_indices = np.where(candidate_condition)[0]
                results['statistics']['accurate'] += accurate_count
                results['statistics']['candidate'] += len(candidate_indices)
                results['statistics']['total'] += prop_len
                results['failed_indices'].append(failed_row_index)

                if len(candidate_indices) == 0:
                    continue

                filtered_rows = frame_property[candidate_indices]
                indices_with_data = np.column_stack((candidate_indices, filtered_rows))
                dlog.info(f"Task {ii} has {len(indices_with_data)} candidates before failed-frame trimming")
                indices_with_data = indices_with_data[indices_with_data[:, 0] < failed_row_index]
                dlog.info(f"Task {ii} keeps {len(indices_with_data)} candidates before local reduction")

                if len(indices_with_data) == 0:
                    continue

                if config['sample_method'] == "fps" and len(indices_with_data) > config['max_candidate_per_task']:
                    candidate_atoms = [atoms_list[idx] for idx in indices_with_data[:, 0].astype(int)]
                    selected_local_indices = select_structure_indices(
                        candidate_atoms,
                        n_samples=config['max_candidate_per_task'],
                        **self._sampling_descriptor_kwargs(),
                    )
                    selected_data = indices_with_data[np.asarray(selected_local_indices, dtype=int)]
                else:
                    selected_data = indices_with_data

                if len(selected_data) == 0:
                    continue

                final_data = selected_data[selected_data[:, 0].argsort()]
                selected_indices = final_data[:, 0].astype(int)
                candidate_atoms = [atoms_list[idx] for idx in selected_indices]
                candidate_records.append(
                    {
                        'task_index': ii,
                        'data': final_data,
                        'atoms': candidate_atoms,
                    }
                )
                np.savetxt(f"candidate_{ii}.txt", final_data, fmt=fmt, header=header, comments=f"_{ii}_")
            finally:
                os.chdir(current_dir)

        results['final_candidate_count'] = self._write_global_candidate_pool(
            candidate_records,
            candidate_file,
            config['max_candidate'],
        )
        return results

    def _quick_failure_check(self, results, config):
        """快速检查失败任务"""
        failed_indices = np.array(results['failed_indices'])
        if len(failed_indices) == 0:
            return False
            
        # 使用第一个任务的长度作为参考
        frame_len = results['statistics']['total'] // len(failed_indices) if len(failed_indices) > 0 else 0
        failed_threshold = int(0.8 * frame_len)
        
        early_failures = failed_indices < failed_threshold
        if np.any(early_failures):
            early_failed_indices = failed_indices[early_failures]
            early_failed_tasks = np.array(self._get_cached_task_dirs())[early_failures]
            min_failed = int(np.min(early_failed_indices))

            dlog.info(f"Early failures detected at indices {early_failed_indices}")

            hard_fail_mask = early_failed_indices < 3
            if np.any(hard_fail_mask):
                hard_fail_pairs = [
                    f"{task}(failed_index={int(idx)})"
                    for task, idx in zip(early_failed_tasks[hard_fail_mask], early_failed_indices[hard_fail_mask])
                ]
                raise ValueError(
                    "Some sampling tasks failed within the first 3 frames, which usually indicates an invalid setup "
                    "(for example wrong nep_in_header / element order, unstable initial structure, or broken model). "
                    f"Offending tasks: {', '.join(hard_fail_pairs)}"
                )

            self.handle_bad_job(
                failed_row_indices=early_failed_indices,
                failed_task_dirs=early_failed_tasks,
            )
            self.run_steps = self._cap_run_steps(
                max(22000, int(self.run_steps*min_failed/frame_len), int(self.run_steps/self.run_steps_factor)),
                f"iter.{self.ii:06d} failed-job recovery",
            )
            dlog.info(f"Adjusted run_steps to {self.run_steps} for next iteration")
            self.write_steps()
            return True
        
        return False

    def _batch_output_processing(self, results, config):
        """批量处理输出"""
        # 处理热力学数据
        if results['thermo_averages']:
            thermo_data = np.vstack(results['thermo_averages'])
            thermo_file = f'{self.work_dir}/thermo.txt'
            
            with open(thermo_file, 'a') as f:
                np.savetxt(f, thermo_data, 
                        fmt="%24.2f"+"%14.2f"*6+"%14.4f"*2+"%14.2f"*2,
                        header=f"{'relative_error':^14} {'energy_error':^14} {'shortest_d':^14} {'temperature':^14} {'potential':^14} {'pressure':^14} {'volume':^14} {'molecule_num':^14} {'molecule_density':^14} {'P_RMSE':^14} {'P_MAE':^14}",
                        comments=f"#iter.{self.ii:06d}")
        
        # 计算和记录比例
        stats = results['statistics']
        if stats['total'] > 0:
            accurate_ratio = stats['accurate'] / stats['total']
            candidate_ratio = stats['candidate'] / stats['total']
            failed_ratio = 1 - accurate_ratio - candidate_ratio
            dlog.info(f"Ratios - failed: {failed_ratio:.4f}, candidate: {candidate_ratio:.4f}, accurate: {accurate_ratio:.4f}")

    def _finalize_iteration(self, config):
        """完成迭代处理"""
        self.run_steps = self._cap_run_steps(self.run_steps, f"iter.{self.ii:06d} finalize")
        self.write_steps()
        dlog.info("All frames processed successfully")
    
    def handle_bad_job(self,failed_row_indices=None,failed_task_dirs=None,allowed_shortest_distance=0.5, run_temp = 1500):
        """
        rerun the failed task
        """
        work_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd/mattersim"
        if not os.path.exists(work_dir):
            os.makedirs(work_dir,exist_ok=True)
        os.chdir(work_dir)
        dlog.info(f"the failed job will be rerun in {os.getcwd()}")
        task_dirs = []
        os.system(f"rm -rf task.*")
        for ii,failed_row_index in enumerate(failed_row_indices):
            if failed_row_index < 3:
                raise ValueError(
                    f"The first 3 frames already failed in {failed_task_dirs[ii]} (failed_index={int(failed_row_index)}). "
                    "This usually indicates an invalid setup such as wrong nep_in_header / element order."
                )
            task_dir = os.path.join(work_dir, f"task.{ii:06d}")
            task_dirs.append(task_dir)
            os.makedirs(task_dir, exist_ok=True)
            os.chdir(task_dir)

            dlog.info(f"the {failed_row_index-2} frames of {failed_task_dirs[ii]} will be rerun")
            atoms = read(os.path.join(failed_task_dirs[ii], "dump.xyz"), index = failed_row_index-2)
            struc_file = os.path.abspath(os.path.join(task_dir, "POSCAR"))
            write(struc_file, atoms)
            py_file = continue_pytemplate.format(structure = struc_file, temperature = run_temp, steps = 20000, **self._ase_template_kwargs())
            with open(os.path.join(task_dir, "ensemble.py"), "w",encoding="utf-8") as f:
                f.write(py_file)

        os.chdir(work_dir)
        task_dirs = [os.path.abspath(task_dir) for task_dir in glob("task.*")]
        
        task_dirs.sort()

        sorted_task_dirs = copy.deepcopy(task_dirs)

        for task_dir in task_dirs:
            os.chdir(task_dir)
            if os.path.exists("task_finished"):
                sorted_task_dirs.remove(task_dir)
                dlog.info(f"the task {task_dir} has been finished")

        os.chdir(work_dir)

        self.run_pytasks(sorted_task_dirs)
        
        os.chdir(work_dir)
        trajs = []
        shortest_distance_records = []
        for task_dir in task_dirs:
            traj_file = os.path.join(task_dir, "out.traj")
            traj = self._load_xyz_structures(traj_file)
            kept_traj, shortest_distances = self._truncate_atoms_before_shortest_distance_failure(
                traj,
                allowed_shortest_distance,
                f"Rerun task {os.path.basename(task_dir)}",
            )
            shortest_distance_records.extend(shortest_distances.tolist())
            trajs.extend(kept_traj)
            
        sampling_cfg = self._sampling_config(self.idata)
        self.max_candidate = sampling_cfg.get("max_candidate", 1000)
        label_dir = os.path.join(self.iter_dir, "02.label")
        os.makedirs(label_dir, exist_ok=True)
        existing_candidates = self._load_xyz_structures(os.path.join(label_dir, "candidate.xyz"))

        candidate_pool = existing_candidates + trajs
        candidate_pool, _ = self._filter_atoms_by_shortest_distance(
            candidate_pool,
            allowed_shortest_distance,
            "Bad-job rerun candidate pool",
        )
        np.savetxt("shortest_distance.txt", np.asarray(shortest_distance_records, dtype=float), fmt="%12.4f")
        if not candidate_pool:
            raise ValueError(
                f"All rerun candidates have shortest distance smaller than {allowed_shortest_distance}"
            )
        if len(candidate_pool) > self.max_candidate:
            reference_atoms = self._load_sampling_reference_structures()
            selected_indices, sampling_info = select_structure_indices_with_info(
                candidate_pool,
                n_samples=self.max_candidate,
                reference_atoms_list=reference_atoms,
                **self._final_candidate_sampling_kwargs(),
                **self._sampling_pca_plot_kwargs(
                    os.path.join(label_dir, "candidate_fps_pca.png"),
                    "Bad-job rerun candidate FPS coverage",
                ),
            )
            candidate_pool = [candidate_pool[i] for i in selected_indices]
            self._write_sampling_stats(
                os.path.join(label_dir, "candidate_fps_stats.txt"),
                {
                    "mode": "bad_job_rerun_pool",
                    "total_candidates": len(existing_candidates) + len(trajs),
                    "selected_candidates": len(candidate_pool),
                    "reference_count": sampling_info.get("reference_count"),
                    "descriptor_source": sampling_info.get("source"),
                    "r2": sampling_info.get("r2"),
                    "r2_threshold": sampling_info.get("r2_threshold"),
                    "plot_path": sampling_info.get("plot_path"),
                },
            )

        os.chdir(label_dir)
        with open("candidate.xyz", "w", encoding="utf-8") as f:
            write_extxyz(f, candidate_pool)

    def run_pytasks(self, task_dirs):
        """
        运行生成python任务
        """
        self.gpu_available = self.idata.get("gpu_available", [0, 1, 2, 3])
        self.gpu_per_task = self.idata.get("gpu_per_task", 1)
        python_interpreter = self.idata.get("python_interpreter", "python")
        processes = []
        
        # 限制同时运行的任务数量
        max_concurrent_tasks = len(self.gpu_available)  # 或者其他合理的数量
        
        for i in range(0, len(task_dirs), max_concurrent_tasks):
            batch_dirs = task_dirs[i:i + max_concurrent_tasks]
            batch_processes = []
            
            for task_dir in batch_dirs:
                os.chdir(task_dir)
                if os.path.exists("task_finished"):
                    dlog.warning(f"{task_dir} has already been finished, skip it")
                    continue
                    
                basename = "ensemble.py"
                gpu_id = self.gpu_available[len(batch_processes) % len(self.gpu_available)]
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                
                log_file = os.path.join(task_dir, 'log')
                try:
                    with open(log_file, 'w') as log:
                        process = subprocess.Popen(
                            [python_interpreter, basename],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=env
                        )
                        batch_processes.append((process, task_dir))
                except Exception as e:
                    dlog.error(f"Failed to start process in {task_dir}: {str(e)}")
                    raise Exception(f"Failed to start process in {task_dir}: {str(e)}")

            
            # 添加超时机制和健康检查
            timeout = 3600  # 设置合理的超时时间（秒）
            start_time = time.time()
            
            while batch_processes and time.time() - start_time < timeout:
                for i, (process, task_dir) in enumerate(batch_processes[:]):
                    ret = process.poll()
                    if ret is not None:  # 进程已结束
                        if ret != 0:
                            dlog.error(f"Process failed with return code {ret}. Check log at: {task_dir}/log")
                        else:
                            try:
                                os.chdir(task_dir)
                                ase_plt()
                                os.system(f"touch {task_dir}/task_finished")
                                dlog.info(f"Process completed successfully. Log at: {task_dir}/log")
                            except Exception as e:
                                dlog.error(f"Post-processing failed for {task_dir}: {str(e)}")
                                raise Exception(f"Post-processing failed for {task_dir}: {str(e)}")
                        
                        batch_processes.pop(i)
                
                # 防止 CPU 空转
                if batch_processes:
                    time.sleep(1)
            
            # 处理超时的进程
            for process, task_dir in batch_processes:
                if process.poll() is None:  # 进程仍在运行
                    dlog.warning(f"Process in {task_dir} timed out, terminating...")
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        dlog.error(f"Process in {task_dir} still running after termination, killing...")
                        process.kill()  # 强制终止
                    dlog.error(f"Process in {task_dir} terminated due to timeout")
        
    @classmethod
    def relative_force_error(
        cls,
        total_time,
        nep_dir=None,
        mode: str = "mean",
        level=1,
        backend: str = "auto",
        enable_molecule_analysis: bool = False,
        molecule_analysis_backend: str = "ase",
        allowed_shortest_distance=0.5,
        allowed_max_temp=10000,
    ):
        """
        return frame_index dict for accurate, candidate, failed
        the candidate_index may be more than the needed, need to resample
        """

        if os.path.exists("frame_property.txt"):
            frame_property = np.loadtxt("frame_property.txt")

            if os.path.exists("failed_index.txt"):
                with open("failed_index.txt","r") as f:
                    failed_index = int(f.read())
            else :
                failed_index = len(frame_property)

            atoms_list = read(f"dump.xyz", index = ":")
            return  atoms_list, frame_property, failed_index

        if not nep_dir:
            nep_dir = os.getcwd()
        if os.path.isfile(nep_dir):
            calculator_fs = [nep_dir]
        else:
            direct_nep = os.path.join(nep_dir, "nep.txt")
            if os.path.isfile(direct_nep):
                calculator_fs = [direct_nep]
            else:
                calculator_fs = glob(f"{nep_dir}/**/nep*.txt", recursive=True)
        if not calculator_fs:
            raise FileNotFoundError(f"No NEP model found under: {nep_dir}")
        atoms_list = read(f"dump.xyz", index = ":")
        


        f_lists = force_main(atoms_list, calculator_fs, backend=backend)
        property_list = []
        time_list = np.linspace(0, total_time, len(atoms_list), endpoint=False)/1000
        for ii,atoms in tqdm(enumerate(atoms_list)):
            f_list = [item[ii] for item in f_lists]
            energy_list = [item[1] for item in f_list]
            f_list = [item[0] for item in f_list]
            f_avg = np.average(f_list,axis=0)
            df_sqr = np.sqrt(np.mean(np.square([(f_list[i] - f_avg) for i in range(len(f_list))]).sum(axis=2),axis=0))
            abs_f_avg = np.mean(np.sqrt(np.square(f_list).sum(axis=2)),axis=0)
            relative_error = (df_sqr/(abs_f_avg+level))
            if mode == "mean":
                relative_error = np.mean(relative_error)
            elif mode == "max":
                relative_error = np.max(relative_error)
            energy_error = np.sqrt(np.mean(np.power(np.array(energy_list)-np.mean(energy_list),2)))
            shortest_distance = get_shortest_distance(atoms)
            property_list.append([time_list[ii], relative_error, energy_error, shortest_distance])
        # property_list
        property_list_np = np.array(property_list)
        thermo = np.loadtxt("thermo.out")
        thermo_new = compute_volume_from_thermo(thermo)[:,[0,2,3,-1]]
        if enable_molecule_analysis:
            if molecule_analysis_backend != "ase":
                raise ValueError(
                    f"Unsupported molecule_analysis_backend: {molecule_analysis_backend}. "
                    "Currently only 'ase' is implemented."
                )
            molecule_data = analyze_trajectory("dump.xyz", index=":").fillna(0)
            molecule_num = molecule_data.sum(axis=1).to_numpy() - molecule_data.iloc[:, 0].to_numpy()
            frame_count = min(len(atoms_list), len(time_list), len(property_list_np), len(thermo_new), len(molecule_num))
        else:
            molecule_data = None
            molecule_num = None
            frame_count = min(len(atoms_list), len(time_list), len(property_list_np), len(thermo_new))
        if frame_count == 0:
            raise ValueError("No frame data found when post-processing dump.xyz/thermo.out")
        if enable_molecule_analysis and (len(property_list_np) != len(thermo_new) or len(thermo_new) != len(molecule_num)):
            dlog.warning(
                "frame count mismatch: atoms=%d, properties=%d, thermo=%d, molecules=%d; truncating to %d frames",
                len(atoms_list),
                len(property_list_np),
                len(thermo_new),
                len(molecule_num),
                frame_count,
            )
        atoms_list = atoms_list[:frame_count]
        time_list = time_list[:frame_count]
        property_list_np = property_list_np[:frame_count]
        thermo = thermo[:frame_count]
        thermo_new = thermo_new[:frame_count]
        frame_property = np.concatenate((property_list_np,thermo_new),axis=1)
        if enable_molecule_analysis:
            molecule_data = molecule_data.iloc[:frame_count].copy()
            molecule_num = molecule_num[:frame_count]
        else:
            molecule_num = np.zeros(frame_count, dtype=float)
            molecule_density = np.zeros(frame_count, dtype=float)
        if enable_molecule_analysis:
            molecule_density = molecule_num / frame_property[:,7]
        frame_property = np.hstack((frame_property, molecule_num.reshape(-1, 1)))
        frame_property = np.hstack((frame_property, molecule_density.reshape(-1, 1)))
        if enable_molecule_analysis:
            molecule_data.to_csv("molecule_data.csv")

        temperatures = thermo[:,0]
        shortest_distances = frame_property[:,3]
        result = np.where(np.logical_or(temperatures > allowed_max_temp, shortest_distances < allowed_shortest_distance))
        if result[0].size > 0:
            # 获取第一个大于6000的数的行索引
            first_row_index = result[0][0]
        else:
            first_row_index = len(frame_property)  # 如果没有找到任何大于6000的数

        fmt = "%12.2f"*9+"%12.4f"
        header = f"{'time':^9} {'relative_error':^14} {'energy_error':^14} {'shortest_d':^14} {'temperature':^14} {'potential':^14} {'pressure':^14} {'volume':^14} {'molecule_num':^14} {'molecule_density':^14}"
        np.savetxt("frame_property.txt", frame_property, fmt = fmt, header = header)
        
        if first_row_index != len(frame_property):
            new_total_time = 0.0 if first_row_index == 0 else time_list[first_row_index-1]*1000
            dlog.warning(f"thermo.out has temperature > {allowed_max_temp} K or shortest distance < {allowed_shortest_distance} too early at frame {first_row_index}({new_total_time} ps), the gpumd task should be rerun")
            np.savetxt(f"failed_index.txt", [first_row_index], fmt="%12d")
            return atoms_list, frame_property, first_row_index
            
        return atoms_list, frame_property, first_row_index

    def make_label_task(self):
        label_engine = self.idata.get("label_engine","mattersim")
        if label_engine == "mattersim":
            self.make_mattersim_task()
        elif label_engine == "vasp":
            self.make_vasp_task()
        else:
            raise NotImplementedError(f"The label engine {label_engine} is not implemented")        

    def make_mattersim_task(self):
        """
        change the calculator to mattersim, and write the train.xyz
        """

        # 将工作目录设置为iter_dir下的02.label文件夹
        iter_dir = self.iter_dir
        work_dir = os.path.join(iter_dir, "02.label")
        train_ratio = self.idata.get("training_ratio", 0.8)
        os.chdir(work_dir)
        atoms_list = read("candidate.xyz", index=":", format="extxyz")
        ase_model_cfg = self._ase_model_config()
        sampling_cfg = self._sampling_config(self.idata)
        sampling_descriptor = sampling_cfg.get("dataset_descriptor", "nep")
        Nepactive.run_mattersim(
            atoms_list=atoms_list,
            ase_model_name=ase_model_cfg["model_name"],
            ase_model_file=ase_model_cfg["model_file"],
            ase_nep_backend=ase_model_cfg["nep_backend"],
            train_ratio=train_ratio,
            sampling_method=sampling_cfg.get("dataset_method", "fps"),
            sampling_nep_file=self._resolve_sampling_nep_file(
                sampling_cfg.get("dataset_nep_file"),
                stage="dataset",
            ),
            sampling_descriptor=sampling_descriptor,
            sampling_min_dist=sampling_cfg.get("dataset_min_dist", 0.0),
            sampling_backend=sampling_cfg.get("dataset_backend", "auto"),
            sampling_pca_plot=bool(sampling_cfg.get("fps_pca_plot", True)),
            sampling_pca_max_points=sampling_cfg.get("fps_pca_max_points"),
        )

    @classmethod
    def run_mattersim(
        cls,
        atoms_list: List[Atoms],
        ase_model_name: str = "mattersim",
        ase_model_file: Optional[str] = None,
        ase_nep_backend: str = "gpu",
        train_ratio: float = 0.8,
        tqdm_use: Optional[bool] = True,
        sampling_method: str = "fps",
        sampling_nep_file: Optional[str] = None,
        sampling_descriptor: str = "auto",
        sampling_min_dist: float = 0.0,
        sampling_backend: str = "auto",
        sampling_pca_plot: bool = True,
        sampling_pca_max_points: Optional[int] = None,
    ):
        calculator = create_ase_calculator(
            model_name=ase_model_name,
            model_file=ase_model_file,
            device="cuda",
            nep_backend=ase_nep_backend,
        )
        if os.path.exists("candidate.traj"):
            os.remove("candidate.traj")
        traj = Trajectory('candidate.traj', mode='a')

        def change_calc(atoms:Atoms):
            atoms.calc=calculator
            if hasattr(atoms, "info") and "virial" in atoms.info:
                del atoms.info["virial"]
            atoms.get_potential_energy()
            traj.write(atoms)
            return atoms

        if tqdm_use:
            atoms = [change_calc(atoms_list[i]) for i in tqdm(range(len(atoms_list)))]
        else:
            atoms = [change_calc(atoms_list[i]) for i in range(len(atoms_list))]   
        # 读取Trajectory对象中的原子信息
        atoms = read("candidate.traj",index=":")
        if hasattr(atoms, "calc") and "virial" in atoms.calc.results:
            dlog.info("Removing 'virial' from calculator results to avoid issues with training")
            del atoms.calc.results["virial"]
        train:List[Atoms]=[]
        test:List[Atoms]=[]
        failed:List[Atoms]=[]
        failed_index=[]
        for i in range(len(atoms)):
            if np.max(np.abs(atoms[i].get_forces())) > 60:
                failed.append(atoms[i])
                failed_index.append(i)
        passed_atoms = [atoms[i] for i in range(len(atoms)) if i not in failed_index]
        train, test = split_train_test_structures(
            passed_atoms,
            training_ratio=train_ratio,
            method=sampling_method,
            descriptor_model=sampling_nep_file,
            descriptor_mode=sampling_descriptor,
            min_dist=sampling_min_dist,
            nep_backend=sampling_backend,
            pca_plot_path="train_test_fps_pca.png" if sampling_pca_plot else None,
            pca_plot_title="Train/test FPS coverage" if sampling_pca_plot else None,
            pca_plot_max_points=sampling_pca_max_points,
        )
        if train:
            write_extxyz("iter_train.xyz",train)
        if test:
            write_extxyz("iter_test.xyz",test)
        if failed:
            write_extxyz("iter_failed.xyz",failed)
            np.savetxt("failed_index.txt",failed_index,fmt="%12d")
        # 将原子信息写入train_iter.xyz文件
        dlog.warning(f"failed structures:{len(failed_index)}")
        del calculator
        empty_cache()

    def make_vasp_task(self):
        task = Remotetask(idata=self.idata, work_dirs=[os.getcwd()])
        task.run_submission()

    def run_label_task(self):
        label_engine = self.idata.get("label_engine","mattersim")
        if label_engine == "mattersim":
            return
        elif label_engine == "vasp":
            self.run_vasp_task()

    def run_vasp_task(self):
        assert os.path.isabs(self.iter_dir)
        label_dir = f"{self.iter_dir}/02.label"
        os.chdir(label_dir)
        Remotetask(idata=self.idata, work_dirs=[label_dir]).run_submission()

    def write_steps(self):
        with open(f"{self.work_dir}/steps.txt","a") as f:
            f.write(f"{int(self.run_steps):12d}\n")
        np.savetxt(f"{self.work_dir}/iter.{self.ii:06d}/steps.txt",np.array([self.run_steps]),fmt="%12d")
