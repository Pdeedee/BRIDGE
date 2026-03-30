
msst_template = """
replicate       {replicate_cell}
potential		nep.txt

minimize sd 1.0e-6 1000

time_step	    {time_step}
velocity		300

ensemble        nvt_ber 300 300 200 

dump_thermo		{dump_freq}
dump_exyz       {dump_freq} 
run			    20000

dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
ensemble msst {shock_direction} {v_shock} qmass {qmass} mu {viscosity} tscale 0.01
run                         {run_steps}
"""

nvt_template = """
replicate       {replicate_cell}
potential		nep.txt
minimize sd 1.0e-6 1000

time_step	    {time_step}
velocity		{temperature}

ensemble        nvt_ber {temperature} {temperature} 200 
dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
run			    {run_steps}
"""

npt_template = """
replicate       {replicate_cell}
potential		nep.txt
minimize sd 1.0e-6 10000

time_step	    {time_step}
velocity		{temperature}

ensemble        npt_mttk temp {temperature} {temperature} iso {pressure} {pressure} tperiod 200 pperiod 5000

dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
run			    {run_steps}
"""

npt_scr_template = """
replicate       {replicate_cell}
potential		nep.txt
minimize sd 1.0e-6 10000

time_step	    {time_step}
velocity		{temperature}

ensemble        npt_scr {temperature} {temperature} {tau_t} {pressure} {elastic_modulus} {tau_p}

dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
run			    {run_steps}
"""

nphugo_mttk_template = """
replicate       {replicate_cell}
potential		nep.txt
minimize sd 1.0e-6 1000

time_step		{time_step}
velocity		300
ensemble        nvt_ber 300 300 200 
dump_thermo		{dump_freq}
dump_exyz       {dump_freq}

run			  20000

dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
ensemble nphug iso {pressure} {pressure} e0 {e0} p0 {p0} v0 {v0} pperiod {pperiod}

run                       {run_steps}
"""

shock_test_template = """
replicate       {replicate_cell}
potential		nep.txt

time_step		{time_step}
velocity		1000

dump_thermo		{dump_freq}
dump_exyz       {dump_freq}
ensemble nphug iso {pressure} {pressure} e0 {e0} p0 {p0} v0 {v0} pperiod {pperiod}

run                       {run_steps}
"""

model_devi_template = """
# set -e  # 遇到错误立即退出
# set -x  # 打印每条执行的命令

echo "Current directory: $(pwd)"
echo "Changing to work_dir: {work_dir}"
cd {work_dir} || {{ echo "Failed to cd to {work_dir}"; exit 1; }}
echo "Changing to task_dir: {task_dir}"
cd {task_dir} || {{ echo "Failed to cd to {task_dir}"; exit 1; }}

echo "Now in directory: $(pwd)"

if [ ! -f task_finished ]; then 
    echo "Starting gpumd task..."
    rm -f dump.xyz thermo.out log
    gpumd > log 2>&1
    if [ $? -eq 0 ]; then 
        touch task_finished
        echo "Task finished successfully"
    else
        echo "gpumd failed with exit code $?"
        exit 1
    fi
else
    echo "Task already finished, skipping..."
fi

"""

pytask_template = """
cd {work_dir}
cd {task_dir}
if [ ! -f task_finished ] ;then 
rm -f md.log out.traj log
python ensemble.py > log 2>&1
if test $? -eq 0; then touch task_finished;fi
fi

"""

nep_in_template ="""
{nep_in_header}
lambda_1      0
zbl           2
version       4       # default
cutoff        5 4     # default
n_max         4 4     # default
basis_size    8 8     # default
l_max         4 2 0   # default
neuron        30      # default
lambda_e      1.0     # default
lambda_f      1.0     # default
lambda_v      0.1     # default
batch         1000     # default
population    50      # default
generation    {train_steps}  # default
 
"""

nvt_pytemplate = """
from ase.io import  read,write
from ase import Atoms,units
from nepactive.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io.trajectory import Trajectory
import numpy as np
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter
from nepactive.nphugo import NPHugo, MTTK
from ase.io import  read,write
from ase.md.nvtberendsen import NVTBerendsen
from nepactive.nep_backend import create_ase_calculator
calculator = create_ase_calculator(model_name={ase_model_name}, model_file={ase_model_file}, device="cuda", nep_backend={ase_nep_backend})
atoms = read("{structure}")
# 使用元素符号排序原子
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
opt = LBFGS(atoms)
opt.run(fmax=0.05,steps=40)
steps = {steps}
write("opt.pdb",atoms)
temperature_K = {temperature}
MaxwellBoltzmannDistribution(atoms,temperature_K=temperature_K)
Stationary(atoms)
ZeroRotation(atoms)
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

"""

npt_pytemplate = """
from ase.io import  read,write
from ase import Atoms,units
from nepactive.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io.trajectory import Trajectory
import numpy as np
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter
from nepactive.nphugo import NPHugo, MTTK
from ase.io import  read,write
from ase.md.nvtberendsen import NVTBerendsen
from nepactive.nep_backend import create_ase_calculator
calculator = create_ase_calculator(model_name={ase_model_name}, model_file={ase_model_file}, device="cuda", nep_backend={ase_nep_backend})
atoms = read("{structure}")
# 使用元素符号排序原子
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
opt = LBFGS(atoms)
opt.run(fmax=0.05,steps=40)
steps = {steps}
write("opt.pdb",atoms)
temperature_K = {temperature}
MaxwellBoltzmannDistribution(atoms,temperature_K=temperature_K)
Stationary(atoms)
ZeroRotation(atoms)
traj = Trajectory('out.traj', 'w', atoms)
# pfactor= 100 #120
pressure = {pressure} * units.GPa
timestep = 0.2 * units.fs
dyn = MTTK(atoms,timestep=0.2*units.fs,run_steps=steps,t_stop=temperature_K,p_stop=pressure,pmode="iso", tchain=3, pchain=3)
# dyn = NVTBerendsen(atoms, timestep=0.1*units.fs, temperature_K=300*units.kB, taut=0.5*1000*units.fs)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval=10)
dyn.attach(traj.write, interval=10)
dyn.run(steps)

"""

npt_scr_pytemplate = """
from ase.io import read, write
from ase import units
from nepactive.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io.trajectory import Trajectory
from ase.optimize import LBFGS
from nepactive.npt_scr import NPT_SCR
from nepactive.nep_backend import create_ase_calculator

calculator = create_ase_calculator(model_name={ase_model_name}, model_file={ase_model_file}, device="cuda", nep_backend={ase_nep_backend})
atoms = read("{structure}")
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
opt = LBFGS(atoms)
opt.run(fmax=0.05, steps=40)
steps = {steps}
write("opt.pdb", atoms)
temperature_K = {temperature}
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
Stationary(atoms)
ZeroRotation(atoms)
traj = Trajectory('out.traj', 'w', atoms)
timestep = {time_step} * units.fs
pressure = {pressure}
dyn = NPT_SCR(
    atoms,
    timestep=timestep,
    temperature=temperature_K,
    pressure=pressure,
    tau_t={tau_t},
    tau_p={tau_p},
    elastic_modulus={elastic_modulus},
    pmode="{pmode}",
)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval={dump_freq})
dyn.attach(traj.write, interval={dump_freq})
dyn.run(steps)

"""

continue_pytemplate = """
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
from nepactive.nep_backend import create_ase_calculator
calculator = create_ase_calculator(model_name={ase_model_name}, model_file={ase_model_file}, device="cuda", nep_backend={ase_nep_backend})
atoms = read("{structure}")
# 使用元素符号排序原子
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
steps = {steps}
temperature_K = {temperature}
traj = Trajectory('out.traj', 'w', atoms)
# pfactor= 100 #120
pressure = 0
timestep = 0.2 * units.fs
dyn = MTTK(atoms,timestep=0.2*units.fs,run_steps=steps,t_stop=temperature_K,p_stop=pressure,pmode=None, tchain=3, pchain=3)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval=10)
dyn.attach(traj.write, interval=10)
dyn.run(steps)

"""

nphugo_mttk_pytemplate = """
from ase.io import  read,write
from nepactive.logger import MDLogger
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io.trajectory import Trajectory
import numpy as np
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter
from nepactive.nphugo import NPHugo, MTTK
from ase.io import  read,write
from nepactive.nep_backend import create_ase_calculator
calculator = create_ase_calculator(model_name={ase_model_name}, model_file={ase_model_file}, device="cuda", nep_backend={ase_nep_backend})
atoms = read("{structure}")
# 使用元素符号排序原子
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
# ucf = UnitCellFilter(atoms,hydrostatic_strain=True)
opt = LBFGS(atoms)
opt.run(fmax=0.05,steps=40)
steps = {steps}
write("opt.pdb",atoms)
temperature_K = 300
MaxwellBoltzmannDistribution(atoms,temperature_K=temperature_K)
Stationary(atoms)
ZeroRotation(atoms)
traj = Trajectory('out.traj', 'w', atoms)
# pfactor= 100 #120
timestep = 0.2 * units.fs
e0 = {e0}
p0 = {p0}
v0 = {v0}
pressure = {pressure} * units.GPa
dyn = NPHugo(atoms, e0 = e0, p0 = p0, v0=v0, p_stop=pressure, timestep=timestep, tchain=3, pchain=3, tfreq={tfreq}, pfreq={pfreq})
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval={dump_freq})
dyn.attach(traj.write, interval={dump_freq})
dyn.run(steps)
"""

nphugo_mttk_pytemplate_shock = """
from ase.io import  read,write
from nepactive.logger import MDLogger
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io.trajectory import Trajectory
import numpy as np
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter
from nepactive.nphugo import NPHugo, MTTK
from ase.io import  read,write
from nepactive.nep_backend import create_ase_calculator
calculator = create_ase_calculator(model_name={ase_model_name}, model_file={ase_model_file}, device="cuda", nep_backend={ase_nep_backend})
atoms = read("{structure}")
# 使用元素符号排序原子
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
ucf = UnitCellFilter(atoms,hydrostatic_strain=True)
opt = LBFGS(ucf)
opt.run(fmax=0.05,steps=40)
steps = {steps}
write("opt.pdb",atoms)
temperature_K = 300
MaxwellBoltzmannDistribution(atoms,temperature_K=temperature_K)
Stationary(atoms)
ZeroRotation(atoms)
traj = Trajectory('out.traj', 'w', atoms)
# pfactor= 100 #120
timestep = 0.2 * units.fs
e0 = {e0}
p0 = 1*units.GPa
v0 = {v0}
pressure = {pressure} * units.GPa
dyn = NPHugo(atoms, e0 = e0, p0 = p0, v0=v0, p_stop=pressure, timestep=timestep, tchain=3, pchain=3, pfreq=0.025, tfreq=0.1)
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval={dump_freq})
dyn.attach(traj.write, interval={dump_freq})
dyn.run(steps)
"""

nphugo_scr_pytemplate = """
# NPHugo with SCR (Stochastic Cell Rescaling) barostat + BDP thermostat
# Ported from GPUMD ensemble npt_scr
# 参数说明（与 GPUMD 输入一致）:
#   tau_t: BDP 恒温器弛豫时间，单位 timestep（GPUMD 默认 100）
#   tau_p: SCR 气压计弛豫时间，单位 timestep（GPUMD 典型 1000~2000）
#   elastic_modulus: 体积模量 (GPa)，默认 15（含能材料典型值，一般不用改）
#   pmode: 压力模式 iso/x/y/z
#   pressure: 目标压力 (GPa)
#   e0/p0/v0: Hugoniot 参考态（能量 eV，压力 eV/Å³，体积 Å³）
from ase.io import read, write
from nepactive.logger import MDLogger
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io.trajectory import Trajectory
import numpy as np
from ase.optimize import LBFGS
from nepactive.npt_scr import NPT_SCR_Hugo
from nepactive.nep_backend import create_ase_calculator
calculator = create_ase_calculator(model_name={ase_model_name}, model_file={ase_model_file}, device="cuda", nep_backend={ase_nep_backend})
atoms = read("{structure}")
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
opt = LBFGS(atoms)
opt.run(fmax=0.05, steps=40)
steps = {steps}
write("opt.pdb", atoms)
temperature_K = 300
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
Stationary(atoms)
ZeroRotation(atoms)
traj = Trajectory('out.traj', 'w', atoms)
timestep = {time_step} * units.fs
e0 = {e0}
p0 = {p0}
v0 = {v0}
pressure = {pressure}
dyn = NPT_SCR_Hugo(atoms, timestep=timestep, pressure=pressure,
                    e0=e0, p0=p0, v0=v0,
                    tau_t={tau_t}, tau_p={tau_p},
                    pmode="{pmode}")
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval={dump_freq})
dyn.attach(traj.write, interval={dump_freq})
dyn.run(steps)
"""

# NPT with temperature and pressure ramping
npt_ramp_pytemplate = """
from ase.io import read, write
from nepactive.logger import MDLogger
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io.trajectory import Trajectory
import numpy as np
from ase.optimize import LBFGS
from nepactive.npt_scr import NPT_SCR
from nepactive.nep_backend import create_ase_calculator
calculator = create_ase_calculator(model_name={ase_model_name}, model_file={ase_model_file}, device="cuda", nep_backend={ase_nep_backend})
atoms = read("{structure}")
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
opt = LBFGS(atoms)
opt.run(fmax=0.05, steps=40)
steps = {steps}
write("opt.pdb", atoms)
MaxwellBoltzmannDistribution(atoms, temperature_K={t_start})
Stationary(atoms)
ZeroRotation(atoms)
traj = Trajectory('out.traj', 'w', atoms)
timestep = {time_step} * units.fs
dyn = NPT_SCR(atoms, timestep=timestep,
              temperature={t_start}, pressure={p_start},
              run_steps=steps,
              t_start={t_start}, t_stop={t_stop},
              p_start={p_start}, p_stop={p_stop},
              tau_t={tau_t}, tau_p={tau_p},
              pmode="{pmode}")
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval={dump_freq})
dyn.attach(traj.write, interval={dump_freq})
dyn.run(steps)
"""

# NVT with temperature ramping (no barostat)
nvt_ramp_pytemplate = """
from ase.io import read, write
from nepactive.logger import MDLogger
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.io.trajectory import Trajectory
import numpy as np
from ase.optimize import LBFGS
from nepactive.npt_scr import NPT_SCR
from nepactive.nep_backend import create_ase_calculator
calculator = create_ase_calculator(model_name={ase_model_name}, model_file={ase_model_file}, device="cuda", nep_backend={ase_nep_backend})
atoms = read("{structure}")
elements = atoms.get_chemical_symbols()
sorted_atoms = atoms[[i for i in sorted(range(len(elements)), key=lambda x: elements[x])]]
atoms.calc = calculator
opt = LBFGS(atoms)
opt.run(fmax=0.05, steps=40)
steps = {steps}
write("opt.pdb", atoms)
MaxwellBoltzmannDistribution(atoms, temperature_K={t_start})
Stationary(atoms)
ZeroRotation(atoms)
traj = Trajectory('out.traj', 'w', atoms)
timestep = {time_step} * units.fs
dyn = NPT_SCR(atoms, timestep=timestep,
              temperature={t_start}, pressure=0,
              run_steps=steps,
              t_start={t_start}, t_stop={t_stop},
              pmode=None,
              tau_t={tau_t})
dyn.attach(MDLogger(dyn, atoms, 'md.log', header=True, stress=True,
        volume=True, mode="w"), interval={dump_freq})
dyn.attach(traj.write, interval={dump_freq})
dyn.run(steps)
"""
