from nepactive.nphugo import MTTK, NPHugo
from nepactive.random_stable import solve_molecular_distribution
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from ase import Atoms,units
import numpy as np
import os
from collections import Counter
from nepactive.packmol import make_structure
import numpy as np
from glob import glob
import subprocess
from nepactive import dlog
from ase.io import read,write
from nepactive.nphugo import MTTK
from ase.io.trajectory import Trajectory
from mattersim.forcefield import MatterSimCalculator
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen
from ase.build import make_supercell
from nepactive.template import nvt_pytemplate,nphugo_pytemplate,nphugo_template,shock_test_template
from nepactive.plt import ase_plt,gpumdplt
from nepactive.tools import shock_calculate,run_gpumd_task,compute_volume_from_thermo
from ase.io.extxyz import write_extxyz

class OB:
    def __init__(self, idata):
        pass

    def run(self):
        pass