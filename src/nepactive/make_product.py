#!/usr/bin/env python3
"""
从 POSCAR 文件生成随机堆积的爆轰产物结构。

用法:
    python make_product.py POSCAR                        # 分析并输出 solutions.yaml + 生成结构
    python make_product.py POSCAR -n 5 -o product        # 每种方案生成5个结构
    python make_product.py POSCAR --only-solve            # 只求解输出 yaml，不生成结构
    python make_product.py POSCAR --pdb-dir /path/to/pdb  # 指定外部 PDB 文件目录

依赖:
    - ase, numpy, scipy, pyyaml
    - packmol (需要在 PATH 中)
    - mattersim
"""

import os
import re
import sys
import time
import random
import shutil
import argparse
import tempfile
import numpy as np
import yaml
from collections import Counter
from scipy.optimize import milp, Bounds
from ase.io import read, write
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter
from ase import units

# ============================================================
# 1. 分子定义
# ============================================================

MOLECULES = {
    'CH4': [1, 4, 0, 0],
    'CO':  [1, 0, 1, 0],
    'CO2': [1, 0, 2, 0],
    'H2':  [0, 2, 0, 0],
    'H2O': [0, 2, 1, 0],
    'N2':  [0, 0, 0, 2],
    'NH3': [0, 3, 0, 1],
    'O2':  [0, 0, 2, 0],
    'CHN': [1, 1, 0, 1],
}

MOLECULE_ENERGIES = {
    'CO':  -14.747749,
    'NH3': -19.33091,
    'O2':  -9.876768,
    'CHN': -19.782274,
    'CO2': -22.8497,
    'H2O': -14.086947,
    'CH4': -24.092106,
    'N2':  -16.628838,
    'H2':  -6.7345247,
}

MOLECULE_NAMES = list(MOLECULES.keys())
ATOM_MATRIX = np.array([MOLECULES[name] for name in MOLECULE_NAMES]).T
ENERGY_ARRAY = np.array([MOLECULE_ENERGIES[name] for name in MOLECULE_NAMES])

PDB_DATA = {
    'H2': (
        "HETATM    1  H1  UNK     1       0.000   0.000   0.000  1.00  0.00           H\n"
        "HETATM    2  H2  UNK     1       0.740   0.000   0.000  1.00  0.00           H\n"
        "END\n"
    ),
    'O2': (
        "HETATM    1  O1  UNK     1       0.000   0.000   0.000  1.00  0.00           O\n"
        "HETATM    2  O2  UNK     1       1.210   0.000   0.000  1.00  0.00           O\n"
        "END\n"
    ),
    'N2': (
        "HETATM    1  N1  UNK     1       0.000   0.000   0.000  1.00  0.00           N\n"
        "HETATM    2  N2  UNK     1       1.098   0.000   0.000  1.00  0.00           N\n"
        "END\n"
    ),
    'H2O': (
        "HETATM    1  O   UNK     1       0.000   0.000   0.000  1.00  0.00           O\n"
        "HETATM    2  H1  UNK     1       0.757   0.586   0.000  1.00  0.00           H\n"
        "HETATM    3  H2  UNK     1      -0.757   0.586   0.000  1.00  0.00           H\n"
        "END\n"
    ),
    'CO': (
        "HETATM    1  C   UNK     1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2  O   UNK     1       1.128   0.000   0.000  1.00  0.00           O\n"
        "END\n"
    ),
    'CO2': (
        "HETATM    1  C   UNK     1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2  O1  UNK     1       1.160   0.000   0.000  1.00  0.00           O\n"
        "HETATM    3  O2  UNK     1      -1.160   0.000   0.000  1.00  0.00           O\n"
        "END\n"
    ),
    'CH4': (
        "HETATM    1  C   UNK     1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2  H1  UNK     1       0.629   0.629   0.629  1.00  0.00           H\n"
        "HETATM    3  H2  UNK     1      -0.629  -0.629   0.629  1.00  0.00           H\n"
        "HETATM    4  H3  UNK     1      -0.629   0.629  -0.629  1.00  0.00           H\n"
        "HETATM    5  H4  UNK     1       0.629  -0.629  -0.629  1.00  0.00           H\n"
        "END\n"
    ),
    'NH3': (
        "HETATM    1  N   UNK     1       0.000   0.000   0.000  1.00  0.00           N\n"
        "HETATM    2  H1  UNK     1       0.940   0.000   0.313  1.00  0.00           H\n"
        "HETATM    3  H2  UNK     1      -0.470   0.814   0.313  1.00  0.00           H\n"
        "HETATM    4  H3  UNK     1      -0.470  -0.814   0.313  1.00  0.00           H\n"
        "END\n"
    ),
    'CHN': (
        "HETATM    1  C   UNK     1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2  H   UNK     1       1.066   0.000   0.000  1.00  0.00           H\n"
        "HETATM    3  N   UNK     1      -1.153   0.000   0.000  1.00  0.00           N\n"
        "END\n"
    ),
}


# ============================================================
# 2. 求解器
# ============================================================

def _verify_solution(solution, target_atoms):
    calculated = np.dot(ATOM_MATRIX, solution)
    return int(np.sum(np.abs(calculated - target_atoms)))


def _sol_to_dict(sol):
    """将解向量转为 {分子名: 数量} 字典"""
    return {name: int(sol[i]) for i, name in enumerate(MOLECULE_NAMES) if sol[i] > 0}


def _sol_key(sol):
    """将解向量转为可哈希的 tuple，用于去重"""
    return tuple(int(x) for x in sol)


def _find_all_valid_solutions(c, h, o, n, num_tries=50000):
    """
    通过随机搜索找到尽可能多的不同有效解 (error=0)。
    返回去重后的解列表 [(sol_array, energy, mol_count), ...]
    """
    target = np.array([c, h, o, n])
    seen = set()
    results = []

    # 先用 MILP 求一个精确解
    try:
        bounds = Bounds(lb=0, ub=np.inf)
        integrality = np.ones(len(MOLECULES))
        milp_result = milp(
            c=ENERGY_ARRAY.copy(),
            constraints=[{"A": ATOM_MATRIX, "b": target, "type": "=="}],
            integrality=integrality, bounds=bounds,
        )
        if milp_result.success:
            sol = milp_result.x.astype(int)
            if _verify_solution(sol, target) == 0:
                key = _sol_key(sol)
                if key not in seen:
                    seen.add(key)
                    energy = float(np.dot(sol, ENERGY_ARRAY))
                    mol_count = int(np.sum(sol))
                    results.append((sol, energy, mol_count))
    except Exception:
        pass

    # 随机搜索更多解
    for trial in range(num_tries):
        seed = int(time.time() * 1e6) % (2**31) + trial
        random.seed(seed)

        sol = np.zeros(len(MOLECULES), dtype=int)
        remaining = target.copy()
        indices = list(range(len(MOLECULES)))
        random.shuffle(indices)
        for idx in indices:
            col = ATOM_MATRIX[:, idx]
            if np.any(col > 0):
                mx = min(remaining[i] // col[i] for i in range(4) if col[i] > 0)
            else:
                mx = 0
            if mx > 0:
                amt = random.randint(0, mx)
                sol[idx] = amt
                remaining -= col * amt

        if int(np.sum(np.abs(remaining))) == 0:
            key = _sol_key(sol)
            if key not in seen:
                seen.add(key)
                energy = float(np.dot(sol, ENERGY_ARRAY))
                mol_count = int(np.sum(sol))
                results.append((sol, energy, mol_count))

    print(f"共找到 {len(results)} 个不同的有效解")
    return results


def rank_solutions(c, h, o, n, top_k=10):
    """
    求解并按两种策略排序，返回:
        lowest_energy: 能量最低的 top_k 个解
        most_molecules: 分子数最多的 top_k 个解
    每个解为 dict: {molecules: {...}, total_energy: float, molecule_count: int}
    """
    all_sols = _find_all_valid_solutions(c, h, o, n)
    if not all_sols:
        raise ValueError(f"无法找到有效解: C={c}, H={h}, O={o}, N={n}")

    # 按能量升序（能量为负数，越小越稳定）
    by_energy = sorted(all_sols, key=lambda x: x[1])[:top_k]
    # 按分子数降序
    by_count = sorted(all_sols, key=lambda x: -x[2])[:top_k]

    def _format(items):
        return [
            {
                "molecules": _sol_to_dict(sol),
                "total_energy": round(energy, 4),
                "molecule_count": mol_count,
            }
            for sol, energy, mol_count in items
        ]

    return _format(by_energy), _format(by_count)


def save_solutions_yaml(poscar_path, yaml_path="solutions.yaml", top_k=10):
    """分析 POSCAR 并将两种方案的 top_k 解保存到 YAML"""
    atoms = read(poscar_path)
    symbols = atoms.get_chemical_symbols()
    counts = Counter(symbols)

    c = counts.get('C', 0)
    h = counts.get('H', 0)
    o = counts.get('O', 0)
    n = counts.get('N', 0)

    other = {e: cnt for e, cnt in counts.items() if e not in ('C', 'H', 'O', 'N')}

    # 打印 POSCAR 中的分子组成
    from nepactive.extract import identify_molecules_in_frame
    mols = identify_molecules_in_frame(atoms)
    mol_formulas = Counter(mol['formula'] for mol in mols)
    print(f"POSCAR 分子组成: {dict(mol_formulas)}")
    if other:
        print(f"非CHON元素(单原子处理): {other}")

    lowest_energy, most_molecules = rank_solutions(c, h, o, n, top_k=top_k)

    # 非CHON元素追加到每个解中
    if other:
        for sol in lowest_energy + most_molecules:
            for elem, cnt in other.items():
                sol["molecules"][elem] = cnt

    data = {
        "composition": {"C": c, "H": h, "O": o, "N": n, **other},
        "lowest_energy": lowest_energy,
        "most_molecules": most_molecules,
    }

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n已保存到 {yaml_path}")
    print(f"  能量最低方案 top {len(lowest_energy)}:")
    for i, s in enumerate(lowest_energy):
        print(f"    {i+1}. E={s['total_energy']:.4f} eV, "
              f"N_mol={s['molecule_count']}, {s['molecules']}")
    print(f"  分子数最多方案 top {len(most_molecules)}:")
    for i, s in enumerate(most_molecules):
        print(f"    {i+1}. E={s['total_energy']:.4f} eV, "
              f"N_mol={s['molecule_count']}, {s['molecules']}")

    return data


# ============================================================
# 3. Packmol 结构生成
# ============================================================

PACKMOL_HEAD = """
tolerance 2.0
add_box_sides 2.0
filetype pdb
output packmol.pdb

"""

PACKMOL_STRUCT = """
structure {pdb_file}
  number {number}
  inside box 0. 0. 0. {l1} {l2} {l3}
end structure

"""


def _write_pdb_files(molecule_dict, work_dir, pdb_dir=None):
    for mol in molecule_dict:
        pdb_path = os.path.join(work_dir, f"{mol}.pdb")
        if os.path.exists(pdb_path):
            continue
        if pdb_dir:
            src = os.path.join(pdb_dir, f"{mol}.pdb")
            if os.path.exists(src):
                shutil.copy(src, pdb_path)
                continue
        if mol in PDB_DATA:
            with open(pdb_path, 'w') as f:
                f.write(PDB_DATA[mol])
        else:
            with open(pdb_path, 'w') as f:
                f.write(f"HETATM    1  {mol:<2}  UNK     1"
                        f"       0.000   0.000   0.000"
                        f"  1.00  0.00           {mol}\n")
                f.write("END\n")


def run_packmol(molecule_dict, work_dir, density=1.8e3):
    old_dir = os.getcwd()
    os.chdir(work_dir)
    try:
        atoms_list = [read(f"{key}.pdb") for key in molecule_dict]
        masses = [a.get_masses().sum() for a in atoms_list]
        total_mass = sum(molecule_dict[k] * masses[i]
                         for i, k in enumerate(molecule_dict))
        volume = total_mass / (density * units.kg / units.m**3)
        length = volume ** (1 / 3)
        print(f"目标密度: {density:.0f} kg/m^3, 盒子边长: {length:.2f} A")

        body = "".join(
            PACKMOL_STRUCT.format(
                pdb_file=f"{k}.pdb", number=v,
                l1=length, l2=length, l3=length)
            for k, v in molecule_dict.items()
        )
        with open("packmol.inp", 'w') as f:
            f.write(PACKMOL_HEAD + body)

        ret = os.system("packmol < packmol.inp")
        if ret != 0:
            raise RuntimeError("packmol 运行失败，请确认 packmol 在 PATH 中")

        atoms = read("packmol.pdb")
        return atoms
    finally:
        os.chdir(old_dir)


# ============================================================
# 4. MatterSim 优化
# ============================================================

def optimize_structure(atoms, fmax=0.05, steps=100):
    from mattersim.forcefield import MatterSimCalculator
    calc = MatterSimCalculator(device="cuda")
    atoms.calc = calc
    ucf = UnitCellFilter(atoms, hydrostatic_strain=True)
    opt = LBFGS(ucf)
    opt.run(fmax=fmax, steps=steps)
    print(f"优化完成, 能量: {atoms.get_potential_energy():.6f} eV")
    return atoms


# ============================================================
# 5. 主函数：生成产物结构
# ============================================================

def make_product(poscar_path, num=1, density=1.8e3, output_prefix="product",
                 output_format="vasp", pdb_dir=None, fmax=0.05, opt_steps=100,
                 only_solve=False, top_k=10, yaml_path="solutions.yaml"):
    """
    从 POSCAR 生成随机堆积的爆轰产物结构。

    1. 求解分子分布，按能量最低和分子数最多各取 top_k，保存到 yaml
    2. 对每种方案的第1个解（最优解）生成 num 个结构
    3. 每个结构保存优化前和优化后两个文件
    """
    poscar_path = os.path.abspath(poscar_path)

    # 求解并保存 yaml
    data = save_solutions_yaml(poscar_path, yaml_path=yaml_path, top_k=top_k)

    if only_solve:
        return

    ext = "vasp" if output_format == "vasp" else "xyz"

    strategies = [
        ("lowest_energy", data["lowest_energy"][0]),
        ("most_molecules", data["most_molecules"][0]),
    ]

    for strategy_name, best_sol in strategies:
        mol_dict = best_sol["molecules"]
        print(f"\n{'='*60}")
        print(f"策略: {strategy_name}")
        print(f"分子分布: {mol_dict}")
        print(f"总能量: {best_sol['total_energy']:.4f} eV, "
              f"分子数: {best_sol['molecule_count']}")
        print(f"{'='*60}")

        out_dir = f"{output_prefix}_{strategy_name}"
        os.makedirs(out_dir, exist_ok=True)

        for idx in range(num):
            print(f"\n--- 生成第 {idx+1}/{num} 个结构 ---")

            work_dir = tempfile.mkdtemp(prefix=f"{strategy_name}_{idx:03d}_")
            _write_pdb_files(mol_dict, work_dir, pdb_dir=pdb_dir)

            atoms = run_packmol(mol_dict, work_dir, density=density)

            # 保存优化前结构
            before_name = os.path.join(out_dir, f"before_opt_{idx:03d}.{ext}")
            write(before_name, atoms)
            print(f"优化前结构: {before_name}")

            # MatterSim 优化
            print("正在用 MatterSim 优化结构...")
            atoms = optimize_structure(atoms, fmax=fmax, steps=opt_steps)

            # 保存优化后结构
            after_name = os.path.join(out_dir, f"after_opt_{idx:03d}.{ext}")
            write(after_name, atoms)
            print(f"优化后结构: {after_name}")

            shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\n完成!")


# ============================================================
# 6. 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="从 POSCAR 生成随机堆积的爆轰产物结构")
    parser.add_argument("poscar", help="输入结构文件 (POSCAR/xyz等)")
    parser.add_argument("-n", "--num", type=int, default=1,
                        help="每种方案生成结构数量 (默认: 1)")
    parser.add_argument("-d", "--density", type=float, default=1.8e3,
                        help="目标密度 kg/m^3 (默认: 1800)")
    parser.add_argument("-o", "--output", default="product",
                        help="输出目录前缀 (默认: product)")
    parser.add_argument("-f", "--format", default="vasp",
                        choices=["vasp", "xyz", "extxyz"],
                        help="输出格式 (默认: vasp)")
    parser.add_argument("--pdb-dir", default=None,
                        help="外部 PDB 分子文件目录")
    parser.add_argument("--fmax", type=float, default=0.05,
                        help="优化收敛力 eV/A (默认: 0.05)")
    parser.add_argument("--opt-steps", type=int, default=100,
                        help="优化最大步数 (默认: 100)")
    parser.add_argument("--only-solve", action="store_true",
                        help="只求解输出 solutions.yaml，不生成结构")
    parser.add_argument("--top-k", type=int, default=10,
                        help="每种方案保留的解数量 (默认: 10)")
    parser.add_argument("--yaml", default="solutions.yaml",
                        help="输出 yaml 文件名 (默认: solutions.yaml)")
    args = parser.parse_args()

    make_product(
        poscar_path=args.poscar,
        num=args.num,
        density=args.density,
        output_prefix=args.output,
        output_format=args.format,
        pdb_dir=args.pdb_dir,
        fmax=args.fmax,
        opt_steps=args.opt_steps,
        only_solve=args.only_solve,
        top_k=args.top_k,
        yaml_path=args.yaml,
    )


if __name__ == "__main__":
    main()
