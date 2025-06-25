from nepactive.nphugo import MTTK, NPHugo
from nepactive.random_stable import solve_molecular_distribution
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from ase import Atoms,units
import numpy as np
import os
from collections import Counter
import numpy as np
from glob import glob
import subprocess
from nepactive import dlog
from ase.io import read,write
from nepactive.nphugo import MTTK
from ase.io.trajectory import Trajectory
from mattersim.forcefield import MatterSimCalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nvtberendsen import NVTBerendsen
from ase.build import make_supercell
from nepactive.template import nvt_pytemplate,nphugo_pytemplate,nphugo_template,shock_test_template
from nepactive.plt import ase_plt,gpumdplt
from nepactive.tools import shock_calculate,run_gpumd_task,compute_volume_from_thermo
from ase.io.extxyz import write_extxyz
from nepactive.train import Nepactive
from collections import Counter
from ase.data import atomic_masses
from ase.data import atomic_numbers
from math import ceil
from ase import units
from nepactive.extract import analyze_trajectory, save_unique_molecules_as_pdb
from nepactive.packmol import make_structure
# from ase.io import read
import os
# from nepactive.extract import 

# 方法1：通过元素符号
mass_H = atomic_masses[atomic_numbers['H']]  # 氢
mass_C = atomic_masses[atomic_numbers['C']]  # 碳
mass_N = atomic_masses[atomic_numbers['N']]  # 氮
mass_O = atomic_masses[atomic_numbers['O']]  # 氧


class OB(Nepactive):
    def __init__(self,idata):
        super().__init__(idata)
    
    def calculate_properties_OB(self, O_give_p=0.1):
        self.calculate_properties()
        prop = np.loadtxt(f"{self.work_dir}/init/properties.txt")
        atoms = read(f"{self.work_dir}/POSCAR")
        atom_list = atoms.get_chemical_symbols()
        counter = Counter(atom_list)
        C_num = counter.get("C", 0)
        O_num = counter.get("O", 0)
        H_num = counter.get("H", 0)
        N_num = counter.get("N", 0)
        mass_o = atoms.get_masses().sum()


        O_need = C_num *2 + H_num/2 - O_num
        OB = -O_need * mass_O/mass_o * 100
        dlog.info(f"OB: {OB:.2f} %") 
        # OB_give = self.idata.get("O_give_p", 0.1)

        O_mgive = mass_o * O_give_p / mass_O /2
        O_mgive_r = ceil(O_mgive)
        mass = mass_o + O_mgive_r * mass_O * 2

        O_give_p_r = O_mgive_r * mass_O * 2 / mass_o
        dlog.info(f"really oxygen give portion : {O_give_p_r*100:.2f} %")

        nat = len(atoms)
        
        Epot_mO2 = -9.876768

        rho = prop[0] 

        e0 = prop[1] + 3/2 * O_mgive_r * 2 * units.kB * 300 + O_mgive_r * Epot_mO2
        p0 = 0
        v0 = mass / (rho * units.kg / units.m**3) /1000

        nat = nat + O_mgive_r * 2
        fmt = "%12.3f "*5
        prop_list = np.array([rho, e0, p0, v0, nat]).reshape(1, -1)
        np.savetxt(f"{self.work_dir}/init/properties_OB_{int(O_give_p*100):d}.txt", prop_list, fmt=fmt)
        return rho, e0, p0, v0, nat, O_mgive_r

    def make_structure_OB(self, OB_give=10):
        rho, e0, p0, v0, nat, O_mgive_r = self.calculate_properties_OB(O_give_p=OB_give/100)
        results = analyze_trajectory(f"{self.work_dir}/POSCAR")
        molecule_dict = results.iloc[0].to_dict()
        save_unique_molecules_as_pdb(f"{self.work_dir}/POSCAR")
        os.system("cp unique_molecules/* .")

        cell_length = np.power(v0, 1/3)*1.2
        cell = [cell_length, cell_length, cell_length]
        molecule_dict['O2'] = O_mgive_r

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        source_file = os.path.join(parent_dir, 'molecules')

        if not os.path.isfile(f"{self.name}.pdb"):
            os.system(f"cp {source_file}/*pdb .")
            dlog.info(f"make structure for {molecule_dict}")
            make_structure(molecule_dict, cell=cell,name=f"{self.name}.pdb")
        atoms = read(f"{self.name}.pdb")
        os.chdir(self.work_dir)
        os.makedirs(f"00.{self.name}", exist_ok=True)
        write(f"00.{self.name}/POSCAR",atoms)

    def prepare(self):
        dlog.info("Running OB initialization")
        self.work_dir = self.idata.get("work_dir", os.getcwd())

        OB_give = self.idata.get("OB_give", 10)
        self.name = f"OB_{OB_give:d}"
        if not os.path.exists(f"{self.work_dir}/00.OB_{self.name}1/POSCAR"):
            self.make_structure_OB(OB_give=OB_give)

        os.chdir(f"{self.work_dir}/00.{self.name}")

        os.makedirs("init", exist_ok=True)
        self.work_dir = os.getcwd()
        dlog.info(f"Working directory changed to {self.work_dir}")
        os.chdir("init")
        os.system(f"ln -snf ../../init/properties_{self.name}.txt properties.txt")
        os.chdir(self.work_dir)
        os.system(f"ln -snf ../in.yaml .")


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
        self.prepare()
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

    