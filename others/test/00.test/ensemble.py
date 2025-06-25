
from ase.io import  read,write
from ase import Atoms,units
from nepactive.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
import numpy as np
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter
from nepactive.nphugo import NPHugo, MTTK
from ase.io import  read,write
from ase.md.nvtberendsen import NVTBerendsen
from mattersim.forcefield import MatterSimCalculator
calculator=MatterSimCalculator(device="cuda")
atoms = read("packmol.pdb")
# 使用元素符号排序原子
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
opt = LBFGS(atoms)
opt.run(fmax=0.05,steps=100)
steps = 2000
write("opt.pdb",atoms)
temperature_K = 300
MaxwellBoltzmannDistribution(atoms,temperature_K=temperature_K)
traj = Trajectory('out.traj', 'w', atoms)
# pfactor= 100 #120
pressure = 0
timestep = 0.2 * units.fs
dyn = MTTK(atoms,timestep=0.2*units.fs,run_steps=steps,t_stop=temperature_K,p_stop=pressure,pmode=None, tchain=3, pchain=3)
# dyn = NVTBerendsen(atoms, timestep=0.1*units.fs, temperature_K=300*units.kB, taut=0.5*1000*units.fs)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval=10)
dyn.attach(traj.write, interval=10)
dyn.run(steps)

