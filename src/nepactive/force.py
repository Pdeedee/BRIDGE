from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from mattersim.forcefield import MatterSimCalculator
from ase.io import read, write
from ase import Atoms
from typing import List, Optional
from ase.io import Trajectory
import random
from tqdm import tqdm
import multiprocessing
import os
from pynep.calculate import NEP
import atexit

def get_force_list(atoms_list: List[Atoms], nep_file):
    calulator = NEP(model_file = nep_file)
    f_list = []
    for atoms in tqdm(atoms_list):
        atoms._calc = calulator
        forces = atoms.get_forces()
        energy = atoms.get_potential_energy()
        f_list.append((forces, energy))
        # traj.write(atoms)
    return f_list

def force_main(atoms_list: List[Atoms], pot_files: list):
    if not pot_files:
        raise ValueError("pot_files is empty, cannot compute model deviation force")
    multiprocessing.set_start_method('spawn', force=True)
    executor = ProcessPoolExecutor(max_workers=min(len(pot_files), os.cpu_count() or 1))
    atexit.register(lambda: executor.shutdown(wait=False))  # 注册退出清理
    
    try:
        futures = {executor.submit(get_force_list, atoms_list, pf): pf for pf in pot_files}
        f_lists = []
        for future in as_completed(futures):
            try:
                f_lists.append(future.result())
            except Exception as e:
                print(f"任务失败: {e}")
        return f_lists
    finally:
        executor.shutdown(wait=True)  # 确保清理

if __name__ == "__main__":
    try:
        force_main(...)
    except KeyboardInterrupt:
        print("检测到中断，清理子进程...")
        os._exit(1)  # 强制退出主进程（避免僵尸进程）
