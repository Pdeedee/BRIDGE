import os
import argparse
from ase.io import read,write
from glob import glob
import subprocess
import numpy as np
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter
from mattersim.forcefield import MatterSimCalculator
import argparse


parser = argparse.ArgumentParser(description="nepactive")
parser.add_argument("file", type=str, help="input file")
args = parser.parse_args() 

def main():
    calculator=MatterSimCalculator(load_path="mattersim-v1.0.0-1m",device="cuda")
    atoms = read(args.file)
    atoms.calc = calculator
    ucf = UnitCellFilter(atoms,hydrostatic_strain=True)
    opt = LBFGS(ucf)
    opt.run(fmax=0.05,steps=100)
    write("POSCAR", atoms)


if __name__ == "__main__":
    main()
    