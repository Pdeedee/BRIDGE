"""
爆热计算模块 - Heat of Detonation (HOD) Calculation
计算优化后结构的能量与初始能量的差值
"""

import os
import subprocess
import numpy as np
from ase.io import read, write
from ase import units
from nepactive import dlog


def calculate_optimized_energy(work_dir: str, nep_path: str = None, gpu_id: int = 0,
                               job_system: dict = None, pot_file: str = None) -> float:
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
    qrelease_dir = os.path.join(work_dir, "Qrelease")
    os.makedirs(qrelease_dir, exist_ok=True)

    original_cwd = os.getcwd()
    os.chdir(qrelease_dir)

    try:
        if os.path.exists("task_finished"):
            dlog.info("Energy optimization already completed, reading results")
            energy = np.loadtxt("energy.txt")
            return energy

        # 准备结构文件 - 使用训练后的 final.xyz
        # 优先使用 struc.000/task.000/final.xyz，如果不存在则使用 POSCAR
        final_xyz_path = os.path.join(work_dir, "struc.000", "task.000", "final.xyz")
        if os.path.exists(final_xyz_path):
            dlog.info(f"Using final structure: {final_xyz_path}")
            atoms = read(final_xyz_path)
        else:
            poscar_path = os.path.join(work_dir, "POSCAR")
            dlog.warning(f"final.xyz not found, using POSCAR: {poscar_path}")
            atoms = read(poscar_path)

        # 运行 MatterSim 优化
        dlog.info(f"Running MatterSim LBFGS optimization on GPU {gpu_id}...")

        if job_system and job_system.get("mode") == "local":
            # 使用作业提交系统
            from nepactive.scheduler import create_scheduler, JobManager

            # 创建优化脚本
            opt_script = f"""#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'

from ase.io import read, write
from mattersim.forcefield import MatterSimCalculator
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter
import numpy as np

atoms = read('POSCAR')
calc = MatterSimCalculator(device='cuda')
atoms.calc = calc
ucf = UnitCellFilter(atoms)
opt = LBFGS(ucf, logfile='opt.log')
opt.run(fmax=0.02, steps=1000)
energy = atoms.get_potential_energy()
write('optfinal.xyz', atoms)
np.savetxt('energy.txt', [energy])
print(f'Optimized energy: {{energy:.6f}} eV')
"""

            with open("optimize.py", "w") as f:
                f.write(opt_script)

            # 写入 POSCAR
            write("POSCAR", atoms)

            # 提交作业
            scheduler_config = job_system.copy()
            scheduler_config["header"] = job_system.get("gpu_header", "")
            scheduler = create_scheduler(scheduler_config)
            job_manager = JobManager(scheduler)

            commands = [
                f"cd {qrelease_dir}",
                f"export CUDA_VISIBLE_DEVICES={gpu_id}",
                "python optimize.py"
            ]
            job_script = os.path.join(qrelease_dir, "job_hod.sh")
            scheduler.write_script(job_script, commands, qrelease_dir)

            job_id = job_manager.submit(job_script, qrelease_dir, "hod_optimization")
            dlog.info(f"Submitted HOD optimization job: {job_id}")
            job_manager.wait_for_jobs([job_id], check_interval=job_system.get("check_interval", 30))

            if not os.path.exists("energy.txt"):
                raise RuntimeError(f"MatterSim optimization failed. Check log: {qrelease_dir}/log")

        else:
            # 直接执行
            from mattersim.forcefield import MatterSimCalculator
            from ase.optimize import LBFGS
            from ase.filters import UnitCellFilter

            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

            calc = MatterSimCalculator(device='cuda')
            atoms.calc = calc
            ucf = UnitCellFilter(atoms)
            opt = LBFGS(ucf, logfile='opt.log')
            opt.run(fmax=0.02, steps=1000)
            energy = atoms.get_potential_energy()
            write("optfinal.xyz", atoms)
            np.savetxt("energy.txt", [energy])

        # 读取优化后的能量
        energy = np.loadtxt("energy.txt")

        # 创建完成标记
        os.system("touch task_finished")

        dlog.info(f"Optimized energy: {energy:.6f} eV")

        return energy

    finally:
        os.chdir(original_cwd)


def calculate_heat_of_detonation(work_dir: str, nep_path: str, gpu_id: int = 0,
                                 job_system: dict = None) -> float:
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
        os.chdir(work_dir)

        # 读取初始能量
        properties_file = os.path.join(work_dir, "properties.txt")
        if not os.path.exists(properties_file):
            raise FileNotFoundError(f"properties.txt not found in {work_dir}")

        rho, e0, p0, v0, nat = np.loadtxt(properties_file)
        dlog.info(f"Initial energy: {e0:.6f} eV, atoms: {int(nat)}")

        # 计算优化后的能量
        ef = calculate_optimized_energy(work_dir, nep_path, gpu_id, job_system)

        # 计算爆热
        Q_release = e0 - ef  # eV per structure

        # 读取结构获取质量
        final_xyz_path = os.path.join(work_dir, "struc.000", "task.000", "final.xyz")
        if os.path.exists(final_xyz_path):
            atoms = read(final_xyz_path)
        else:
            poscar_path = os.path.join(work_dir, "POSCAR")
            atoms = read(poscar_path)

        mass = atoms.get_masses().sum() / units.kg  # kg

        # 转换为 kJ/kg (使用 ASE units)
        Q_release_per_kg = Q_release / units.kJ / mass

        dlog.info(f"Final energy: {ef:.6f} eV")
        dlog.info(f"Heat of detonation: {Q_release:.6f} eV = {Q_release_per_kg:.2f} kJ/kg")
        dlog.info(f"Number of atoms: {int(nat)}, Total mass: {mass*1e27:.6f} g")

        # 保存结果
        q_release_file = os.path.join(work_dir, "Q_release.txt")
        with open(q_release_file, "w") as f:
            f.write(f"# Heat of Detonation\n")
            f.write(f"# Q (eV): {Q_release:.6f}\n")
            f.write(f"# Q (kJ/kg): {Q_release_per_kg:.2f}\n")
            f.write(f"# Atoms: {int(nat)}, Mass: {mass*1e27:.6f} g\n")
            f.write(f"{Q_release_per_kg:.4f}\n")

        return Q_release_per_kg

    finally:
        os.chdir(original_dir)


def batch_calculate_heat_of_detonation(base_dir: str, pattern: str = "iter.*/03.shock",
                                       gpu_id: int = 0, job_system: dict = None):
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
            Q_release = calculate_heat_of_detonation(shock_dir, nep_path, gpu_id, job_system)
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
