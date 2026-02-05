#!/usr/bin/env python3
"""
准备好 所有的pdb文件，同时修改好晶胞大小，和pdb文件名。
"""
import os
import argparse
from ase.io import read,write
from glob import glob
import subprocess
import numpy as np
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter
from ase import units
from nepactive import dlog
# os.system("cp /home/liuzf/scripts/packmol/*pdb .")
name = "packmol.pdb"

template ="""
structure {pdb_file}
  number {number}
  inside box 0. 0. 0. {l1} {l2} {l3}
end structure

"""

head_text = """
tolerance 2.0
add_box_sides 2.0
filetype pdb
output packmol.pdb

"""
# structure = "structure.pdb"

# cell = [12, 12, 12]

def make_structure(molecules:dict,name:str,density:float=1.8e3):
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_dir = os.path.dirname(os.path.dirname(current_dir))
    # source_file = os.path.join(parent_dir, 'molecules')
    # os.system(f"cp {source_file}/*pdb .")


    atoms_list = [read(f"{key}.pdb") for key in molecules.keys()]
    atoms_masses = [atoms.get_masses().sum() for atoms in atoms_list]
    total_mass = sum([molecules[key]*atoms_masses[i] for i,key in enumerate(molecules.keys())])
    volume = total_mass / (density * units.kg / units.m**3)  # cm^3
    length = volume ** (1/3)
    dlog.info(f"Target density: {density} g/m^3, Total mass: {total_mass} amu, Volume: {volume:.2f} A^3, Box length: {length:.2f} A")
    cell = [length]*3
    cycle_text = "\n".join([template.format(pdb_file = f"{key}.pdb", number = value, l1 = cell[0], l2 = cell[1], l3 = cell[2]) for key,value in molecules.items()])
    text = head_text + cycle_text

    with open("packmol.inp",'w') as file:
        file.write(text)

    print(f"current_dir:{os.getcwd()}")
    os.system("packmol < packmol.inp")

    # 读取生成的 packmol.pdb 文件
    atoms = read("packmol.pdb")

    # 使用元素符号排序原子
    elements = atoms.get_chemical_symbols()
    sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]

    from mattersim.forcefield import MatterSimCalculator
    calculator=MatterSimCalculator(load_path="mattersim-v1.0.0-1m",device="cuda")
    atoms.calc = calculator
    ucf = UnitCellFilter(atoms,hydrostatic_strain=True)
    opt = LBFGS(ucf)
    opt.run(fmax=0.05,steps=100)

    # 写入 VASP POSCAR 文件
    write(name, atoms)
