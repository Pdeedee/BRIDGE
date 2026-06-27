"""
爆热计算模块 - Heat of Detonation (HOD) Calculation
计算优化后结构的能量与初始能量的差值
"""

from __future__ import annotations

import os
import subprocess
import numpy as np
import csv
import time
import sys
from glob import glob
from ase.io import read, write
from ase import units
from nepactive import dlog


def _normalize_gpu_ids(gpu_ids=None, gpu_id: int = 0) -> list[int]:
    if gpu_ids is None:
        return [int(gpu_id)]
    if isinstance(gpu_ids, (int, str)):
        return [int(gpu_ids)]
    gpu_list = []
    for gpu in gpu_ids:
        gpu = int(gpu)
        if gpu not in gpu_list:
            gpu_list.append(gpu)
    return gpu_list or [int(gpu_id)]


def _numeric_task_index(task_dir: str) -> int | None:
    name = os.path.basename(task_dir)
    parts = name.split(".")
    if len(parts) == 2 and parts[0] == "task" and parts[1].isdigit():
        return int(parts[1])
    return None


def _collect_final_task_structures(work_dir: str) -> list[dict]:
    task_dirs = glob(os.path.join(work_dir, "struc.*", "task.*"))
    records = []
    for task_dir in sorted(task_dirs):
        task_idx = _numeric_task_index(task_dir)
        if task_idx is None:
            continue
        final_xyz = os.path.join(task_dir, "final.xyz")
        if not os.path.exists(final_xyz):
            continue
        struc_name = os.path.basename(os.path.dirname(task_dir))
        records.append({
            "struc": struc_name,
            "task": os.path.basename(task_dir),
            "task_index": task_idx,
            "task_dir": os.path.abspath(task_dir),
            "structure": os.path.abspath(final_xyz),
        })
    return records


def _optimize_atoms_energy(work_dir: str, atoms, label: str, gpu_id: int = 0,
                           job_system: dict = None) -> float:
    """Optimize one structure with MatterSim + LBFGS + UnitCellFilter."""
    work_dir = os.path.abspath(work_dir)
    qrelease_dir = os.path.join(work_dir, "Qrelease")
    os.makedirs(qrelease_dir, exist_ok=True)

    marker_file = f"{label}_task_finished"
    input_file = f"{label}_input.xyz"
    output_file = f"{label}_opt.xyz"
    energy_file = f"{label}_energy.txt"
    log_file = f"{label}_opt.log"

    original_cwd = os.getcwd()
    os.chdir(qrelease_dir)

    try:
        if os.path.exists(marker_file) and os.path.exists(energy_file):
            dlog.info(f"{label} optimization already completed, reading results")
            return float(np.loadtxt(energy_file))

        write(input_file, atoms)
        dlog.info(f"Running MatterSim LBFGS optimization for {label} on GPU {gpu_id}...")

        if job_system and job_system.get("mode") == "local":
            from nepactive.scheduler import create_scheduler, JobManager

            opt_script = f"""#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'

from ase.io import read, write
from mattersim.forcefield import MatterSimCalculator
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter
import numpy as np

atoms = read('{input_file}')
calc = MatterSimCalculator(device='cuda')
atoms.calc = calc
ucf = UnitCellFilter(atoms)
opt = LBFGS(ucf, logfile='{log_file}')
opt.run(fmax=0.02, steps=1000)
energy = atoms.get_potential_energy()
write('{output_file}', atoms)
np.savetxt('{energy_file}', [energy])
print(f'{label} optimized energy: {{energy:.6f}} eV')
"""

            script_file = f"optimize_{label}.py"
            with open(script_file, "w") as f:
                f.write(opt_script)

            scheduler_config = job_system.copy()
            scheduler_config["header"] = job_system.get("gpu_header", "")
            scheduler = create_scheduler(scheduler_config)
            job_manager = JobManager(scheduler)

            commands = [
                f"cd {qrelease_dir}",
                f"export CUDA_VISIBLE_DEVICES={gpu_id}",
                f"python {script_file}",
            ]
            job_script = os.path.join(qrelease_dir, f"job_hod_{label}.sh")
            scheduler.write_script(job_script, commands, qrelease_dir)

            job_id = job_manager.submit(job_script, qrelease_dir, f"hod_{label}_optimization")
            dlog.info(f"Submitted HOD {label} optimization job: {job_id}")
            job_manager.wait_for_jobs([job_id], check_interval=job_system.get("check_interval", 30))

            if not os.path.exists(energy_file):
                raise RuntimeError(f"MatterSim {label} optimization failed. Check log: {qrelease_dir}/log")

        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            from mattersim.forcefield import MatterSimCalculator
            from ase.optimize import LBFGS
            from ase.filters import UnitCellFilter

            calc = MatterSimCalculator(device='cuda')
            atoms.calc = calc
            ucf = UnitCellFilter(atoms)
            opt = LBFGS(ucf, logfile=log_file)
            opt.run(fmax=0.02, steps=1000)
            energy = atoms.get_potential_energy()
            write(output_file, atoms)
            np.savetxt(energy_file, [energy])

        energy = float(np.loadtxt(energy_file))
        os.system(f"touch {marker_file}")
        dlog.info(f"{label} optimized energy: {energy:.6f} eV")
        return energy

    finally:
        os.chdir(original_cwd)


def _optimize_structure_worker(qrelease_dir: str, structure_path: str, label: str,
                               gpu_id: int, job_system: dict = None) -> float:
    atoms = read(structure_path)
    return _optimize_atoms_energy(qrelease_dir, atoms, label, gpu_id, job_system)


def _optimize_final_tasks_local(work_dir: str, task_records: list[dict],
                                gpu_ids: list[int], job_system: dict = None) -> list[dict]:
    qrelease_dir = os.path.join(work_dir, "Qrelease")
    os.makedirs(qrelease_dir, exist_ok=True)

    pending = []
    results = []
    for record in task_records:
        label = f"final_{record['struc']}_{record['task']}"
        record = record.copy()
        record["label"] = label
        record["energy_file"] = os.path.join(qrelease_dir, f"{label}_energy.txt")
        record["opt_structure"] = os.path.join(qrelease_dir, f"{label}_opt.xyz")
        if os.path.exists(record["energy_file"]):
            record["energy"] = float(np.loadtxt(record["energy_file"]))
            record["cached"] = True
            results.append(record)
        else:
            pending.append(record)

    running = []
    pending_index = 0
    original_cwd = os.getcwd()
    try:
        while pending_index < len(pending) or running:
            while pending_index < len(pending) and len(running) < len(gpu_ids):
                record = pending[pending_index]
                busy_gpus = {item["gpu_id"] for item in running}
                gpu_id = next(gpu for gpu in gpu_ids if gpu not in busy_gpus)
                pending_index += 1
                log_path = os.path.join(qrelease_dir, f"{record['label']}_worker.log")
                script = f"""import os
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'
from ase.io import read, write
from mattersim.forcefield import MatterSimCalculator
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter
import numpy as np

atoms = read(r'''{record['structure']}''')
atoms.calc = MatterSimCalculator(device='cuda')
ucf = UnitCellFilter(atoms)
opt = LBFGS(ucf, logfile=r'''{os.path.join(qrelease_dir, record['label'] + '_opt.log')}''')
converged = bool(opt.run(fmax=0.02, steps=1000))
energy = float(atoms.get_potential_energy())
write(r'''{record['opt_structure']}''', atoms)
np.savetxt(r'''{record['energy_file']}''', [energy])
with open(r'''{os.path.join(qrelease_dir, record['label'] + '_converged.txt')}''', 'w') as f:
    f.write(str(converged) + '\\n')
print(f"{record['label']} {{energy:.12f}}")
"""
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                with open(log_path, "w") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-c", script],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=qrelease_dir,
                        env=env,
                    )
                running.append({"process": process, "record": record, "gpu_id": gpu_id, "log_path": log_path})
                dlog.info(
                    f"Started HOD final optimization {record['struc']}/{record['task']} "
                    f"on GPU {gpu_id}"
                )

            if not running:
                continue
            time.sleep(5)
            still_running = []
            for item in running:
                process = item["process"]
                if process.poll() is None:
                    still_running.append(item)
                    continue
                record = item["record"]
                if process.returncode != 0:
                    raise RuntimeError(
                        f"HOD final optimization failed for {record['task_dir']}; "
                        f"check log: {item['log_path']}"
                    )
                record["energy"] = float(np.loadtxt(record["energy_file"]))
                record["cached"] = False
                results.append(record)
                dlog.info(
                    f"Finished HOD final optimization {record['struc']}/{record['task']}: "
                    f"{record['energy']:.6f} eV"
                )
            running = still_running
    finally:
        os.chdir(original_cwd)

    results.sort(key=lambda item: (item["struc"], item["task_index"]))
    return results


def _write_final_task_summary(work_dir: str, results: list[dict]) -> None:
    qrelease_dir = os.path.join(work_dir, "Qrelease")
    summary_file = os.path.join(qrelease_dir, "final_task_energies.csv")
    fieldnames = [
        "struc", "task", "task_index", "energy_eV", "cached",
        "task_dir", "input_structure", "opt_structure",
    ]
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted(results, key=lambda row: row["energy"]):
            writer.writerow({
                "struc": item["struc"],
                "task": item["task"],
                "task_index": item["task_index"],
                "energy_eV": f"{item['energy']:.12f}",
                "cached": item.get("cached", False),
                "task_dir": item["task_dir"],
                "input_structure": item["structure"],
                "opt_structure": item["opt_structure"],
            })


def _read_final_structure(work_dir: str):
    final_xyz_path = os.path.join(work_dir, "struc.000", "task.000", "final.xyz")
    if os.path.exists(final_xyz_path):
        dlog.info(f"Using final structure: {final_xyz_path}")
        return read(final_xyz_path)

    poscar_path = os.path.join(work_dir, "POSCAR")
    dlog.warning(f"final.xyz not found, using POSCAR: {poscar_path}")
    return read(poscar_path)


def calculate_optimized_initial_energy(work_dir: str, gpu_id: int = 0,
                                       job_system: dict = None) -> float:
    """Optimize the initial POSCAR and return its MatterSim potential energy."""
    work_dir = os.path.abspath(work_dir)
    atoms = read(os.path.join(work_dir, "POSCAR"))
    return _optimize_atoms_energy(work_dir, atoms, "initial", gpu_id, job_system)


def calculate_optimized_energy(work_dir: str, nep_path: str = None, gpu_id: int = 0,
                               job_system: dict = None, pot_file: str = None,
                               gpu_ids=None) -> float:
    """
    计算优化后结构的能量（使用 MatterSim + LBFGS 优化）

    Args:
        work_dir: 工作目录
        nep_path: NEP 势函数路径（不使用，保留参数兼容性）
        gpu_id: GPU 编号
        job_system: 作业提交系统配置（可选）
        pot_file: MatterSim 模型文件路径

    Returns:
        energy: 优化后的能量
    """
    work_dir = os.path.abspath(work_dir)
    gpu_ids = _normalize_gpu_ids(gpu_ids, gpu_id)
    task_records = _collect_final_task_structures(work_dir)
    if task_records:
        if job_system and job_system.get("mode") == "local":
            dlog.warning(
                "Parallel final-task HOD currently runs locally; job_system is ignored for final-task fanout."
            )
        results = _optimize_final_tasks_local(work_dir, task_records, gpu_ids, None)
        _write_final_task_summary(work_dir, results)
        best = min(results, key=lambda item: item["energy"])
        best_file = os.path.join(work_dir, "Qrelease", "best_final_task.txt")
        with open(best_file, "w") as f:
            f.write(
                f"{best['struc']}/{best['task']} {best['energy']:.12f} "
                f"{best['opt_structure']}\n"
            )
        dlog.info(
            f"Best optimized final task: {best['struc']}/{best['task']} "
            f"E={best['energy']:.6f} eV"
        )
        return float(best["energy"])

    atoms = _read_final_structure(work_dir)
    return _optimize_atoms_energy(work_dir, atoms, "final", gpu_ids[0], job_system)


def calculate_heat_of_detonation(work_dir: str, nep_path: str, gpu_id: int = 0,
                                 job_system: dict = None, gpu_ids=None) -> float:
    """
    计算爆热：Q = E_initial - E_optimized

    Args:
        work_dir: 工作目录
        nep_path: NEP 势函数路径
        gpu_id: GPU 编号
        job_system: 作业提交系统配置（可选）

    Returns:
        Q_release: 爆热 (kJ/mol)
    """
    original_dir = os.getcwd()

    try:
        work_dir = os.path.abspath(work_dir)
        os.chdir(work_dir)

        # 读取初始能量
        properties_file = os.path.join(work_dir, "properties.txt")
        if not os.path.exists(properties_file):
            raise FileNotFoundError(f"properties.txt not found in {work_dir}")

        rho, e0, p0, v0, nat = np.loadtxt(properties_file)
        dlog.info(f"Initial energy (from properties.txt): {e0:.6f} eV, atoms: {int(nat)}")

        # 初态和末态都使用 MatterSim + LBFGS + UnitCellFilter 优化后的势能。
        gpu_ids = _normalize_gpu_ids(gpu_ids, gpu_id)
        pe0 = calculate_optimized_initial_energy(work_dir, gpu_ids[0], job_system)
        ef = calculate_optimized_energy(work_dir, nep_path, gpu_ids[0], job_system, gpu_ids=gpu_ids)

        # 读取结构获取质量和初始势能
        final_xyz_path = os.path.join(work_dir, "struc.000", "task.000", "final.xyz")
        if os.path.exists(final_xyz_path):
            atoms = read(final_xyz_path)
        else:
            poscar_path = os.path.join(work_dir, "POSCAR")
            atoms = read(poscar_path)

        mass = atoms.get_masses().sum() / units.kg  # kg

        dlog.info(f"Initial energy (optimized): {pe0:.6f} eV")

        # Q_pe: 纯势能差
        Q_pe = pe0 - ef
        Q_pe_per_kg = Q_pe / units.kJ / mass

        # Q_total: 总能量差（e0 来自 NVT 平衡后，含动能贡献）
        Q_total = e0 - ef
        Q_total_per_kg = Q_total / units.kJ / mass

        dlog.info(f"Final energy (optimized): {ef:.6f} eV")
        dlog.info(f"Q_pe  (potential only): {Q_pe:.6f} eV = {Q_pe_per_kg:.2f} kJ/kg")
        dlog.info(f"Q_total (with kinetic): {Q_total:.6f} eV = {Q_total_per_kg:.2f} kJ/kg")
        dlog.info(f"Number of atoms: {int(nat)}, Total mass: {mass*1e27:.6f} g")

        # 保存结果
        q_release_file = os.path.join(work_dir, "Q_release.txt")
        with open(q_release_file, "w") as f:
            f.write(f"# Heat of Detonation\n")
            f.write(f"# E0 (properties.txt): {e0:.6f} eV\n")
            f.write(f"# PE0 (optimized potential energy): {pe0:.6f} eV\n")
            f.write(f"# Ef (lowest optimized final task): {ef:.6f} eV\n")
            f.write(f"# Q_pe  (eV): {Q_pe:.6f}\n")
            f.write(f"# Q_pe  (kJ/kg): {Q_pe_per_kg:.2f}\n")
            f.write(f"# Q_total (eV): {Q_total:.6f}\n")
            f.write(f"# Q_total (kJ/kg): {Q_total_per_kg:.2f}\n")
            f.write(f"# Atoms: {int(nat)}, Mass: {mass*1e27:.6f} g\n")
            f.write(f"{Q_pe_per_kg:.4f} {Q_total_per_kg:.4f}\n")

        return Q_pe_per_kg

    finally:
        os.chdir(original_dir)


def batch_calculate_heat_of_detonation(base_dir: str, pattern: str = "iter.*/03.shock",
                                       gpu_id: int = 0, job_system: dict = None,
                                       gpu_ids=None):
    """
    批量计算爆热（用于命令行工具）

    Args:
        base_dir: 基础目录
        pattern: 搜索模式
        gpu_id: GPU 编号
        job_system: 作业提交系统配置（可选）
    """
    from glob import glob

    shock_dirs = glob(os.path.join(base_dir, pattern))
    shock_dirs.sort()

    if not shock_dirs:
        dlog.warning(f"No shock directories found matching pattern: {pattern}")
        return []

    results = []

    for shock_dir in shock_dirs:
        dlog.info(f"\n{'='*60}")
        dlog.info(f"Processing: {shock_dir}")
        dlog.info(f"{'='*60}")

        # 查找 nep.txt
        iter_dir = os.path.dirname(shock_dir)
        nep_path = os.path.join(iter_dir, "00.nep", "task.000000", "nep.txt")

        if not os.path.exists(nep_path):
            dlog.error(f"NEP file not found: {nep_path}")
            results.append({
                'dir': shock_dir,
                'Q_release': None,
                'status': 'failed',
                'error': 'NEP file not found'
            })
            continue

        try:
            Q_release = calculate_heat_of_detonation(
                shock_dir,
                nep_path,
                gpu_id,
                job_system,
                gpu_ids=gpu_ids,
            )
            results.append({
                'dir': shock_dir,
                'Q_release': Q_release,
                'status': 'success'
            })
        except Exception as e:
            dlog.error(f"Failed to calculate heat of detonation for {shock_dir}: {e}")
            results.append({
                'dir': shock_dir,
                'Q_release': None,
                'status': 'failed',
                'error': str(e)
            })

    # 输出汇总
    dlog.info(f"\n{'='*60}")
    dlog.info("Heat of Detonation Calculation Summary")
    dlog.info(f"{'='*60}")

    for result in results:
        if result['status'] == 'success':
            dlog.info(f"{result['dir']}: {result['Q_release']:.2f} kJ/kg")
        else:
            dlog.error(f"{result['dir']}: FAILED - {result.get('error', 'Unknown error')}")

    return results
