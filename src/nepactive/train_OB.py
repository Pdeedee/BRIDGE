import os
from nepactive import dlog
from nepactive.remote import Remotetask
import shutil
import subprocess
from ase.io import read, write, Trajectory
from glob import glob
from ase import Atoms
import random
from typing import List,Optional
import numpy as np
from tqdm import tqdm
from mattersim.forcefield import MatterSimCalculator
import random
from math import floor
from torch.cuda import empty_cache
import itertools
from concurrent.futures import ThreadPoolExecutor
import re
import copy
from nepactive.stable_OB import StableRun_OB
from nepactive.template import npt_template,nphugo_template,nvt_template,msst_template,model_devi_template,nep_in_template,nphugo_pytemplate,nvt_pytemplate,continue_pytemplate
from nepactive.plt import gpumdplt,nep_plt,ase_plt
from nepactive import parse_yaml
from nepactive.force import force_main
from nepactive.tools import compute_volume_from_thermo,run_gpumd_task
from nepactive.extract import analyze_trajectory
from nepactive.write_extxyz import write_extxyz
import time
from ase import units
from collections import Counter
from nepactive.extract import analyze_trajectory, save_unique_molecules_as_pdb
from nepactive.packmol import make_structure
from ase.data import atomic_masses
from ase.data import atomic_numbers
from math import ceil
from nepactive.train import Nepactive

def calculate_properties_OB(work_dir, idata, OB_give=10):
    if not os.path.exists(f"{work_dir}/init/properties.txt"):
        calculate_properties(work_dir, idata)
    prop = np.loadtxt(f"{work_dir}/init/properties.txt")
    atoms = read(f"{work_dir}/POSCAR")
    atom_list = atoms.get_chemical_symbols()
    counter = Counter(atom_list)
    C_num = counter.get("C", 0)
    O_num = counter.get("O", 0)
    H_num = counter.get("H", 0)
    N_num = counter.get("N", 0)
    mass_o = atoms.get_masses().sum()

    O_give_p = OB_give/100
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
    fmt = "%12.3f " * 5
    prop_list = np.array([rho, e0, p0, v0, nat]).reshape(1, -1)
    os.makedirs(f"{work_dir}/OBfs",exist_ok=True)
    if not os.path.exists(f"{work_dir}/OBfs/properties_OB_{int(O_give_p*100):d}.txt"):
        np.savetxt(f"{work_dir}/OBfs/properties_OB_{int(O_give_p*100):d}.txt", prop_list, fmt=fmt)
        dlog.info(f"save properties at {work_dir}/OBfs/properties_OB_{int(O_give_p*100):d}.txt")
    return rho, e0, p0, v0, nat, O_mgive_r

def make_structure_OB(work_dir, idata, OB_give=10):
    os.chdir(work_dir)
    rho, e0, p0, v0, nat, O_mgive_r = calculate_properties_OB(work_dir=work_dir, idata=idata, OB_give=OB_give)
    os.makedirs("OBfs",exist_ok=True)
    os.chdir("OBfs")
    results = analyze_trajectory(f"{work_dir}/POSCAR")
    molecule_dict = results.iloc[0].to_dict()
    save_unique_molecules_as_pdb(f"{work_dir}/POSCAR")
    os.system("cp unique_molecules/*pdb .")

    cell_length = np.power(v0, 1/3)*1.2
    cell = [cell_length, cell_length, cell_length]
    molecule_dict['O2'] = O_mgive_r

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    source_file = os.path.join(parent_dir, 'molecules')

    name= f"OB_{int(OB_give)}"

    if not os.path.isfile(f"{name}.pdb"):
        os.system(f"cp {source_file}/*pdb .")
        dlog.info(f"make structure for {molecule_dict}")
        make_structure(molecule_dict, cell=cell,name=f"{name}.pdb")

    atoms = read(f"{name}.pdb")
    os.chdir(work_dir)

    return rho, e0, p0, v0, nat, O_mgive_r

def calculate_properties(work_dir:str = None, idata:dict = None):
    os.chdir(work_dir)
    os.makedirs("init",exist_ok=True)
    if os.path.exists("POSCAR"):
        shutil.copy("POSCAR","init")
    work_dir = f"{work_dir}/init"
    os.chdir(work_dir)
    stable_data:dict = idata.get("stable")
    stable_run = StableRun_OB(stable_data)
    stable_run.calculate_properties()

mass_H = atomic_masses[atomic_numbers['H']]  # 氢
mass_C = atomic_masses[atomic_numbers['C']]  # 碳
mass_N = atomic_masses[atomic_numbers['N']]  # 氮
mass_O = atomic_masses[atomic_numbers['O']]  # 氧

class RestartSignal(Exception):
    def __init__(self, restart_total_time = None):
        super().__init__()
        self.restart_total_time = restart_total_time

def process_id(value):
    # 检查是否为单一数字
    if re.match(r'^\d+$', value):
        return [int(value)]  # 如果是数字，返回列表形式的单个数字
    # 检查是否为有效的数字范围，如 "3-11"
    if re.match(r'^\d+-\d+$', value):
        start, end = value.split('-')
        start, end = int(start), int(end)
        if start > end:
            raise ValueError(f"范围的起始值不能大于结束值: {value}")
        return list(range(start, end + 1))  # 返回范围的数字列表
    # 如果不匹配数字或范围格式，抛出异常
    raise ValueError(f"无效的格式: {value}，必须是数字或者数字范围（如 '3-11'）")

def get_shortest_distance(atoms:Atoms,atom_index=None):
    distance_matrix = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distance_matrix, np.inf)
    min_index = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
    if atom_index is not None:
        atom_index.append(min_index)
    return np.min(distance_matrix)

def traj_write(atoms_list:Atoms, calculator):
    traj = Trajectory("out.traj", "w")
    for atoms in atoms_list:
        atoms._calc = calculator
        atoms.get_potential_energy()
        traj.write(atoms)    

def get_force(atoms:Atoms,calculator):
    atoms._calc=calculator
    return atoms.get_forces(),atoms.get_potential_energy()

# task = None
Maxlength = 70
def sepline(ch="-", sp="-"):
    r"""Seperate the output by '-'."""
    # if screen:
    #     print(ch.center(MaxLength, sp))
    # else:
    dlog.info(ch.center(Maxlength, sp))

def record_iter(record, ii, jj):
    with open(record, "a") as frec:
        frec.write("%d %d\n" % (ii, jj))

class Nepactive_OB(Nepactive):
    def __init__(self,idata:dict):
        self.idata:dict = idata
        self.work_dir = os.getcwd()
        self.make_gpumd_task_first = True
        self.gpu_available = self.idata.get("gpu_available")

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

        engine = self.idata.get("ini_engine","ase")
        # label_engine = idata.get("label_engine","mattersim")
        work_dir = self.work_dir
        os.chdir(work_dir)
        os.makedirs("init",exist_ok=True)
        record = "record.nep"
        #init engine choice
        dlog.info(f"Initializing engine and make initial training data using {engine}")
        if not os.path.isfile(record):
            calculate_properties(self.work_dir, self.idata)
            self.make_init_ase_run() 
            dlog.info("Extracting data from initial runs")
            self.make_data_extraction()
        self.make_loop_train()

    def make_init_ase_run(self):
        '''
        For the template file, the file name must be fixed form.
        Assumed that the working directory is already the correct directory.
        '''
        # if_stable_run = self.idata.get("if_stable_run",False)

        work_dir = f"{self.work_dir}/init"
        struc_dirs = []
        os.chdir(work_dir)
        # if if_stable_run:
        rho = self.idata.get("rho", None)
        stable_data:dict = self.idata.get("stable")
        stable_data["steps"] = self.idata.get("ini_traj_steps", 40000)
        if rho:
            dlog.info(f"rho is {rho}, will run stable run for rho={rho}")
            stable_data["rho"] = rho
        stable_run = StableRun_OB(stable_data)
        stable_run.calculate_properties()
        
        # stable_run.calculate_properties()
        for ii in range(stable_run.struc_num):
            os.chdir(work_dir)
            os.makedirs(f"struc.{ii:03d}",exist_ok=True)
            struc_dir = os.path.abspath(f"struc.{ii:03d}")
            struc_dirs.append(struc_dir)
            os.chdir(struc_dir)
            
            stable_run.make_preparations()

        # original_make = stable_data.get("original_make",True)
        # if original_make:
        if True:
            os.chdir(work_dir)
            os.makedirs("original",exist_ok=True)
            struc_dir = os.path.abspath("original")
            struc_dirs.append(struc_dir)
            os.chdir(struc_dir)
            atoms = read(f"{self.work_dir}/POSCAR")
            os.makedirs("structure",exist_ok=True)
            write(f"structure/stable.pdb",atoms)
            stable_run.make_preparations()

        OB_gives = self.idata.get("OB_gives", [])

        if OB_gives:
            os.chdir(self.work_dir)
            rho,e0,p0,v0,nat,_ = make_structure_OB(work_dir=self.work_dir,idata=self.idata,OB_give=OB_gives[-1])
            os.chdir(f"{work_dir}")
            os.makedirs("OB",exist_ok=True)
            os.chdir("OB")
            os.system(f"ln {self.work_dir}/OBfs/properties_OB_{OB_gives[-1]}.txt properties.txt -snf")
            atoms = read(f"{self.work_dir}/OBfs/OB_{OB_gives[-1]}.pdb")
            os.makedirs("structure", exist_ok=True)
            write("structure/stable.pdb", atoms)
            stable_run.make_preparations()
            
            
        ase_ensemble_files = self.idata.get("ini_ase_ensemble_files")
        if ase_ensemble_files:
            ase_ensemble_files:list[str] = [os.path.abspath(path) for path in ase_ensemble_files]

        python_interpreter:str = self.idata.get("python_interpreter")
        processes = []
        self.pot_file:str = self.idata.get("pot_file")
        pot_file =self.pot_file
        self.gpu_available = self.idata.get("gpu_available")
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        # Make initial training task directories
        if ase_ensemble_files:
            for index, file in enumerate(ase_ensemble_files):
                task_name = f"task.{index:06d}"
                # Ensure the task directory is created
                task_dir = os.path.join(work_dir, task_name)
                os.makedirs(task_dir, exist_ok=True)  
                # Create the task directory, if it doesn't exist
                os.system(f"ln -snf {pot_file} {task_dir}/model.pth")
                os.system(f"ln -snf {file} {task_dir}")
        
        os.chdir(work_dir)
        task_dirs = glob("./**/task.*", recursive=True)
        if task_dirs:
            task_dirs = [os.path.abspath(path) for path in task_dirs]

        for index,task_dir in enumerate(task_dirs):
            os.chdir(task_dir)
            if os.path.exists("task_finished"):
                atoms = read("out.traj", index=-1)
                write_extxyz("final.xyz", atoms)
                dlog.warning(f"{task_dir} has already been finished, skip it")
                continue
            # basename = os.path.basename(file)
            basename = "ensemble.py"
            gpu_id = self.gpu_available[index%len(self.gpu_available)]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            # Open subprocess and redirect stdout and stderr to a log file
            log_file = os.path.join(task_dir, 'log')  # Log file path
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    [python_interpreter, basename], 
                    stdout=log, 
                    stderr=subprocess.STDOUT,  # Combine stderr with stdout
                    env = env
                )
                processes.append((process, task_dir))  # Store the process and log file

        # Wait for all subprocesses to complete and check for errors
        for process, task_dir in processes:
            process.wait()  # Wait for the process to complete
            # Check for errors using the return code
            if process.returncode != 0:
                dlog.error(f"Process failed. Check the log at: {task_dir}/log")
                raise RuntimeError(f"Process failed. Check the log at: {task_dir}/log")
            else:
                os.chdir(task_dir)
                ase_plt()
                os.system(f"touch {task_dir}/task_finished")
                atoms = read("out.traj", index=-1)
                write_extxyz("final.xyz", atoms)
                dlog.info(f"Process completed successfully. Log saved at: {task_dir}/log")

        # All scripts executed, proceed to the next step
        dlog.info("Initial training data generated")

    def make_data_extraction(self):
        '''
        extract data from initial runs, turn it into the gpumd format
        '''
        train:List[Atoms]=[]
        test:List[Atoms]=[]
        training_ratio:float = self.idata.get("training_ratio", 0.8)
        init_frames:int = self.idata.get("ini_frames", 100)
        # work_dir = f"{work_dir}/init"
        os.chdir(self.work_dir)

        ####检查
        fs = glob("init/**/task.*/*.traj",recursive=True)
        dlog.info(f"Found {len(fs)} files to extract frames from.")
        if not fs:
            raise ValueError("No files found to extract frames from.")
        # may report error due to the format not matching
        atoms = []
        
        # average sampling
        fnumber= len(fs)
        needed_frames = init_frames/fnumber
        
        for f in fs:
            atom = read(f,index=":")
            if len(atom) < needed_frames:
                dlog.warning(f"Not enough frames in {f} to extract {needed_frames} frames.")
                # raise ValueError(f"Not enough frames in {f} to extract {needed_frames} frames.")
            interval = max(1, floor(len(atom) / needed_frames))  # 确保 interval 至少为 1
            atom = atom[::interval]
            atoms.extend(atom)

        assert atoms is not None

        if len(atoms) < init_frames:
            dlog.warning(f"Not enough frames to extract {init_frames} frames.")
        elif len(atoms) > init_frames:
            # dlog.warning(f"Too many frames to extract {init_frames} frames.")
            # raise ValueError(f"Not enough frames to extract {init_frames} frames.")
            atoms = atoms[:init_frames]
        else:
            dlog.info(f"Extracted {init_frames} frames from {fnumber} files.")

        write_extxyz("init/init.xyz", atoms)
        for i in range(len(atoms)):
            rand=random.random()
            if rand <= training_ratio:
                train.append(atoms[i])
            elif rand > training_ratio:
                test.append(atoms[i])
            else:
                dlog.warning(f"{atoms[i]}failed to be classified")
        write_extxyz("init/train.xyz", train)
        write_extxyz("init/test.xyz", test)
        dlog.info("Initial training data extracted")

    def make_loop_train(self):
        '''
        make loop training task
        '''
        
        os.chdir(self.work_dir)

        record = "record.nep"
        iter_rec = [0, -1]
        if os.path.isfile(record):
            with open(record) as frec:
                for line in frec:
                    iter_rec = [int(x) for x in line.split()]
        cont = True
        self.ii = -1
        numb_task = 9
        max_tasks = 10000
        self.restart_total_time = None

        while True:
            self.ii += 1
            if self.ii < iter_rec[0]:
                continue
            self.iter_dir = os.path.abspath(os.path.join(self.work_dir,f"iter.{self.ii:06d}")) #the work path has been changed
            os.makedirs(self.iter_dir, exist_ok=True)
            iter_name = f"iter.{self.ii:06d}"
            self.restart_gpumd = False

            for jj in range(numb_task):
                if not self.restart_gpumd:
                    self.jj = jj
                yaml_synchro = self.idata.get("yaml_synchro", False)
                if yaml_synchro:
                    dlog.info(f"yaml_synchro is True, reread the in.yaml from {self.work_dir}/in.yaml")
                    self.idata:dict = parse_yaml(f"{self.work_dir}/in.yaml")
                if self.ii * max_tasks + jj <= iter_rec[0] * max_tasks + iter_rec[1] and not self.restart_gpumd:
                    continue
                task_name = "task %02d" % jj
                sepline(f"{iter_name} {task_name}", "-")
                if jj == 0:
                    self.make_nep_train()
                elif jj == 1:
                    self.run_nep_train()
                elif jj == 2:
                    self.post_nep_train()
                elif jj == 3:
                    self.make_model_devi()
                elif jj == 4:
                    self.run_model_devi()
                elif jj == 5:
                    self.post_gpumd_run()
                elif jj == 6:
                    self.make_label_task()
                elif jj == 7:
                    self.run_label_task()
                elif jj == 8:
                    self.post_label_task()
                else:
                    raise RuntimeError("unknown task %d, something wrong" % jj)
                
                os.chdir(self.work_dir)
                record_iter(record, self.ii, jj)

    def shock(self):
        
        stable_data = self.idata.get("stable", None)
        rhos = stable_data.get("rhos", [1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        if stable_data is None:
            raise ValueError("stable data is None, please check your in.yaml")
        
        stable_data["pot"] = "nep"
        if not stable_data.get("nep", None):
            stable_data["nep"] = os.path.join(os.path.abspath(os.path.abspath(os.getcwd())),"nep.txt")
        stable_data["original_make"] = False
        struture_files = os.path.join(os.path.abspath(os.getcwd()), "POSCAR")
        stable_data["structure_files"] = [struture_files]
        if not os.path.isfile("properties.txt"):
            raise ValueError("properties.txt is not found, please check your in.yaml")
        for rho in rhos:
            os.chdir(self.work_dir)
            os.makedirs(f"{rho}",exist_ok=True)
            os.chdir(f"{rho}")
            stable_data["rho"] = rho
            dlog.info(f"Running shock velocity test for rho={rho}")
            os.system(f"ln -snf {self.work_dir}/POSCAR POSCAR")
            os.system(f"ln -snf {self.work_dir}/properties.txt properties.txt")
            stable_task = StableRun_OB(stable_data)
            stable_task.run()
            dlog.info(f"Shock velocity test for rho={rho} completed")


    def shock_vel_test(self):
        work_dir = os.path.abspath(os.path.join(self.iter_dir, "03.shock"))
        os.makedirs(work_dir, exist_ok=True)
        atoms = read(f"{self.work_dir}/POSCAR")
        write(f"{work_dir}/POSCAR", atoms)
        os.system(f"ln -snf {self.work_dir}/init/properties.txt {work_dir}/properties.txt")
        os.chdir(work_dir)
        stable_data = self.idata.get("stable", None)
        assert stable_data is not None, "stable data is None"
        nep_file = os.path.join(self.iter_dir, "00.nep/task.000000/nep.txt")
        stable_data["nep"] = nep_file
        stable_data["pot"] = "nep"

        original_make = stable_data.get("original_make", False)
        final_xyzs = glob(f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd/task.[0-9][0-9][0-9][0-9][0-9][0-9]/final.xyz")
        final_xyzs.sort()

        if original_make:
            structure_files = [final_xyzs[len(self.idata.get("model_devi_general",{}).get("pressure"))-1]]
        else:
            structure_files = []

        final_xyz = final_xyzs[-1]
        
        rho = self.idata.get("rho", None)
        if rho:
            dlog.info(f"rho is {rho}, will run shock velocity test for rho={rho}")
            stable_data["rho"] = rho
        structure_files.append(final_xyz)

        for ii, struc in enumerate(structure_files):
            os.makedirs(f"job.{ii:03d}",exist_ok=True)
            os.chdir(f"job.{ii:03d}")
            os.system(f"ln {os.path.dirname(struc)}/properties*.txt properties.txt -snf")
            dlog.info(f"start running {ii}th task of {len(struc)}")
            atoms = read(struc)
            write("POSCAR", atoms)
            stable_task = StableRun_OB(stable_data)
            stable_data["structure_files"] = [struc]
            stable_task.run()
            with open(f"{self.work_dir}/shock_vel.txt", "a") as f:
                f.write(f"{self.ii}\n")
                np.savetxt(f, stable_task.shock_vels, fmt='%.3f', header='Shock velocities (m/s) for each rho')
            os.chdir(f"{work_dir}")
        dlog.info(f"Shock velocity test completed, results is {stable_task.shock_vels} km/s")



    def run_nep_train(self):
        '''
        run nep training
        '''
        train_steps = self.idata.get("train_steps", 5000)
        work_dir = os.path.abspath(os.path.join(self.iter_dir, "00.nep"))
        pot_num = self.idata.get("pot_num", 4)
        pot_inherit:bool = self.idata.get("pot_inherit", True)
        # nep_template = os.path.abspath(self.idata.get("nep_template"))
        processes = []
        if not pot_inherit:
            dlog.info(f"{os.getcwd()}")
            dlog.info(f"pot_inherit is false, will remove old task files {work_dir}/task*")
            os.system(f"rm -r {work_dir}/task.*")
        for jj in range(pot_num):
            #ensure the work_dir is the absolute path
            task_dir = os.path.join(work_dir, f"task.{jj:06d}")
            os.makedirs(task_dir,exist_ok=True)
            #     absworkdir/iter.000000/00.nep/task.000000/
            os.chdir(task_dir)
            #preparation files
            if not os.path.isfile("train.xyz"):
                os.symlink("../dataset/train.xyz","train.xyz")
            if not os.path.isfile("test.xyz"):
                os.symlink("../dataset/test.xyz","test.xyz")
            if not os.path.isfile("nep.in"):
                nep_in_header = self.idata.get("nep_in_header", "type 4 H C N O")
                if self.ii == 0:
                    ini_train_steps = self.idata.get("ini_train_steps", 10000)
                    nep_in = nep_in_template.format(train_steps=ini_train_steps,nep_in_header=nep_in_header)
                else:
                    nep_in = nep_in_template.format(train_steps=train_steps,nep_in_header=nep_in_header)
                with open("nep.in", "w") as f:
                    f.write(nep_in)
                # os.symlink(nep_template, "nep.in")
            if pot_inherit and self.ii > 0:
                nep_restart = f"{self.work_dir}/iter.{self.ii-1:06d}/00.nep/task.{jj:06d}/nep.restart"
                dlog.info(f"pot_inherit is true, will copy nep.restart from {nep_restart}")
                shutil.copy(nep_restart, "nep.restart")
                # exit()
            log_file = os.path.join(task_dir, 'log')  # Log file path
            env = os.environ.copy()
            gpu_id = self.gpu_available[jj%len(self.gpu_available)]
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            # env['CUDA_VISIBLE_DEVICES'] = str(jj)
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    ["nep"],  # 程序名
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env  # 使用修改后的环境
                )
                processes.append((process, log_file))
            
            self.gpu_numbers = len(self.idata.get("gpu_available"))            
            if (jj+1)%self.gpu_numbers == 0:
                dlog.info(f"jobs submitted, checking status of {self.jj}")
                for process, log_file in processes:
                    process.wait()  # Wait for all processes to complete
                    # Check for errors using the return code
                    if process.returncode != 0:
                        dlog.error(f"Process failed. Check the log at: {log_file}")
                        raise RuntimeError(f"One or more processes failed. Check the log file:({log_file}) for details.")
                    else:
                        dlog.info(f"Process completed successfully. Log saved at: {log_file}")

    def post_label_task(self):
        '''
        '''
        test_interval = self.idata.get("shock_test_interval", 1)
        test_begin_step = self.idata.get("shock_test_begin_step", 400000)

        self.run_steps = int(np.loadtxt(f"{self.work_dir}/steps.txt",ndmin=1,encoding="utf-8")[-1])
        if self.run_steps > test_begin_step and self.ii%test_interval == 0:
            dlog.info(f"run_steps is {self.run_steps}, will run shock velocity test")
            self.shock_vel_test()

    def post_nep_train(self):
        '''
        post nep training
        '''
        os.chdir(f"{self.work_dir}/iter.{self.ii:06d}/00.nep")
        tasks = glob("task.*")
        nep_plot = self.idata.get("nep_plot", True)
        for task in tasks:
            if nep_plot:
                os.chdir(f"{self.work_dir}/iter.{self.ii:06d}/00.nep")
                os.chdir(task)
                if not os.path.exists("loss.png"):
                    nep_plt()

        os.chdir(f"{self.work_dir}/iter.{self.ii:06d}/00.nep")
        # nep_plt(testplt=False)
        os.system("rm dataset/*xyz */*train.out */*test.out")

    def make_nep_train(self):
        '''
        Train nep. 
        '''
        #ensure the work_dir is the absolute pathk
        global_work_dir = os.path.abspath(self.work_dir)
        work_dir = os.path.abspath(os.path.join(self.iter_dir, "00.nep"))
        pot_num = self.idata.get("pot_num", 4)
        init:bool = self.idata.get("init", True)

        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)
        #     absworkdir/iter.000000/00.nep/


        #preparation files
        if pot_num > 4:
            raise ValueError(f"pot_num should be no bigger than 4, and it is now {pot_num}")
        
        #make training data preparation
        os.makedirs("dataset", exist_ok=True)
        def merge_files(input_files, output_file):
            command = ['cat'] + input_files  # 适用于类Unix系统
            with open(output_file, 'wb') as outfile:
                subprocess.run(command, stdout=outfile)
        files = []
        testfiles = []
        # 注意每一代有没有划分训练集和测试集
        if init == True:
            print(f"{os.path.isfile(os.path.join(global_work_dir, 'init/train.xyz'))}")
            # 直接调用 extend 方法，不要尝试将其结果赋值
            files.extend(glob(os.path.join(global_work_dir, "init/train.xyz")))
            testfiles.extend(glob(os.path.join(global_work_dir, "init/test.xyz")))

            # 检查文件列表是否为空
            # if not files:
            #     raise ValueError("No files found to merge.")
        """等于之后反而出错了"""

        if self.ii > 0:
            for iii in range(self.ii):
                newtrainfile = glob(f"../../iter.{iii:06d}/02.label/iter_train.xyz")
                files.extend(newtrainfile)
                newtestfile = glob(f"../../iter.{iii:06d}/02.label/iter_test.xyz")
                testfiles.extend(newtestfile)
                if (not newtrainfile) or (not newtestfile):
                    dlog.warning(f"iter.{iii:06d} has no training or test data")
        if not files:
            raise ValueError("No files found to merge.")
            # dlog.error("No files found to merge.")
        merge_files(files, "dataset/train.xyz")
        merge_files(testfiles, "dataset/test.xyz")
        self.gpu_available = self.idata.get("gpu_available")

    def _setup_parameters_make_gpumd(self, model_devi):
        """设置NPHUGO任务的所有参数"""
        # 基本参数
        work_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
        structure_files = self.idata.get("structure_files")
        structure_prefix = self.idata.get("structure_prefix", self.work_dir)
        structure_files = [os.path.join(structure_prefix, sf) for sf in structure_files]
        
        # 计算运行步数
        self.run_steps_factor = self.idata.get("run_steps_factor", 1.5)
        if self.ii == 0:
            run_steps = self.idata.get("ini_run_steps", 100000)
        else:
            all_run_steps = np.loadtxt(f"{self.work_dir}/steps.txt", ndmin=1, encoding="utf-8")
            old_run_steps = all_run_steps[-1]
            run_steps = int(self.run_steps_factor * int(old_run_steps))
            
            if len(all_run_steps) > 1 and all_run_steps[-2] > all_run_steps[-1]:
                dlog.warning(f"The older run_steps is {old_run_steps}, the new run_steps is {all_run_steps[-1]}, and the older one is bigger")
        
        self.run_steps = run_steps
        dlog.info(f"run_steps is {run_steps}")
        
        return {
            'work_dir': work_dir,
            'structure_files': structure_files,
            'nep_file': self.idata.get("nep_file", "../../00.nep/task.000000/nep.txt"),
            'needed_frames': self.idata.get("needed_frames", 10000),
            'time_step_general': self.idata.get("model_devi_time_step", None),
            'OB_gives': self.idata.get("OB_gives", [0]),
            'pperiod': model_devi.get("pperiod", 2000),
            'run_steps': run_steps
        }

    def make_model_devi(self):
        """运行gpumd，这个函数被make_loop_train调用"""
        model_devi = self.get_model_devi()

        # 备份处理
        if self.make_gpumd_task_first:
            dlog.info(f"make_gpumd_task_first is true, will backup old task directory {self.work_dir}/iter.{self.ii:06d}/01.gpumd")
            
            src_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
            if os.path.exists(src_dir):
                # 找到下一个备份后缀
                bak_files = glob(f"{src_dir}.bak.*")
                new_suffix = 0 if not bak_files else max([int(f.split('.')[-1]) for f in bak_files]) + 1
                
                dst_dir = f"{src_dir}.bak.{new_suffix}"
                shutil.move(src_dir, dst_dir)
                dlog.info(f"Backup completed: {src_dir} -> {dst_dir}")
            else:
                dlog.warning(f"Source directory does not exist: {src_dir}")

        # 设置参数
        params = self._setup_parameters_make_gpumd(model_devi)

        # 生成任务
        self.total_time, self.model_devi_task_numbers = self.make_gpumd_task(model_devi, **params)

    def make_gpumd_task(self, model_devi: dict, structure_files, needed_frames=10000, 
                    time_step_general=0.2, work_dir: str = None, nep_file: str = None, 
                    run_steps: int = 20000, OB_gives=[0], pperiod=1000):
        """生成NPHUGO系综的GPUMD任务"""
        if not work_dir:
            work_dir = os.getcwd()
        
        assert run_steps > 20000, "NPHUGO ensemble requires run_steps > 20000"

        # 获取NPHUGO参数
        structure_id = model_devi.get("structure_id", [[0]])[0]  # 默认使用第一个结构
        structure = [structure_files[ii] for ii in structure_id]
        

        pressure = model_devi.get("pressure")

        assert pressure is not None, "NPHUGO ensemble requires pressure"
        
        # 构建参数字典
        all_dict = {
            "OB_gives": OB_gives,
            "structure": structure,
            "pressure": pressure,
            "pperiod": [pperiod]
        }
        
        # 验证参数
        assert all(v not in [None, '', [], {}, set()] for v in all_dict.values()), "Empty parameters in NPHUGO"
        
        # 生成参数组合
        task_dicts = []
        for combo in itertools.product(*all_dict.values()):
            combo_dict = {key: combo[index] for index, key in enumerate(all_dict.keys())}
            task_dicts.append(combo_dict)
        
        dlog.info(f"NPHUGO generated {len(task_dicts)} tasks")

        # 计算dump频率和其他参数
        frames_per_task = needed_frames / len(task_dicts)
        dump_freq = max(1, floor(run_steps / frames_per_task))
        time_step = model_devi.get("time_step", time_step_general)
        replicate_cell = model_devi.get("replicate_cell", "1 1 1")
        model_devi_task_numbers = len(task_dicts)

        # 创建任务文件
        for index, task in enumerate(task_dicts):
            task_dir = f"{work_dir}/task.{index:06d}"
            os.makedirs(task_dir, exist_ok=True)
            
            # 生成NPHUGO输入文件
            OB_gives_val = task.get("OB_gives", 0)
            dlog.error(f"{self.work_dir}")
            rho, e0, p0, v0, nat, _ = make_structure_OB(work_dir=self.work_dir, idata=self.idata, OB_give=OB_gives_val)
            
            # 计算修正体积
            r_rho = self.idata.get("rho", None)
            r_v = rho * v0 / r_rho if r_rho is not None else v0

            # 保存属性文件
            fmt = "%12.3f " * 5
            prop_list = np.array([rho, e0, p0, v0, nat]).reshape(1, -1)
            np.savetxt(f"{task_dir}/properties_OB_{OB_gives_val}.txt", prop_list, fmt=fmt)
            
            # 处理初始压力
            real_p0 = self.idata.get("real_p0", False)
            if not real_p0:
                dlog.info(f"real_p0 is {real_p0}, will set p0 to 0")
                p0 = 0
                
            text = nphugo_template.format(
                time_step=time_step,
                run_steps=run_steps-20000,
                dump_freq=dump_freq,
                replicate_cell=replicate_cell,
                e0=e0, p0=p0, v0=r_v,
                **task
            )

            # 设置任务环境
            os.chdir(task_dir)
            structure = task["structure"]
            
            # 处理OB结构
            if OB_gives_val != 0:
                structure = f"{self.work_dir}/OBfs/OB_{OB_gives_val:d}.pdb"
                
            atoms = read(structure)
            write_extxyz("model.xyz", atoms)
            
            # 写入输入文件
            with open("run.in", 'w') as f:
                f.write(text)
                
            # 创建NEP文件链接
            if not os.path.isfile("nep.txt"):
                os.symlink(nep_file, "nep.txt")

        total_time = run_steps * time_step
        return total_time, model_devi_task_numbers

    def run_model_devi(self):
        # dlog.info("entering the run_gpumd_task")
        try:
            self.dump_freq
        except AttributeError:
            # 捕获变量不存在时的错误
            self.make_gpumd_task_first = False
            self.make_model_devi()
            self.make_gpumd_task_first = True
            dlog.info("remake the gpumd task")
        model_devi_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
        os.chdir(model_devi_dir)
        gpu_available = self.idata.get("gpu_available")
        self.task_per_gpu = self.idata.get("task_per_gpu")

        Nepactive_OB.run_gpumd_task(work_dir=model_devi_dir, gpu_available=gpu_available, task_per_gpu=self.task_per_gpu)
        # self.write_steps()

    @classmethod
    def run_gpumd_task(cls,work_dir:str=None,gpu_available:List[int]=None,task_per_gpu:int=1):
        run_gpumd_task(work_dir=work_dir, gpu_available=gpu_available, task_per_gpu=task_per_gpu)

        
    def get_model_devi(self):
        '''
        get the model deviation from the gpumd run
        '''
        # dlog.info(f"self.idata:{self.idata}")
        model_devi_general:list[dict] = self.idata.get("model_devi_general", None)
        # dlog.info(f"model_devi_general:{model_devi_general}")
        if os.path.exists(os.path.join(self.work_dir, "model_devi_general.txt")):
            file = os.path.join(self.work_dir, "model_devi_general.txt")
            with open(file, mode='r') as f:
                model_devi_general = float(f.read()[-1])
            self.model_devi_general_id = np.loadtxt(os.path.join(self.work_dir, "model_devi_general_id.txt"), dtype=int)[-1]
        else:
            self.model_devi_general_id = 0
        
        if self.model_devi_general_id >= len(model_devi_general):
            dlog.info(f"finished")
            exit()
        
        model_devi = model_devi_general[self.model_devi_general_id] 

        return model_devi

    def post_gpumd_run(self):
        '''
        优化版本：专注于循环和I/O瓶颈优化
        '''
        try:
            self.total_time
        except AttributeError:
            self.make_gpumd_task_first = False
            self.make_model_devi()
            self.make_gpumd_task_first = True
            dlog.info("remake the gpumd task")

        # 预先获取所有配置参数，避免重复调用
        config = self._get_cached_config()
        nep_dir = os.path.join(self.iter_dir, "00.nep")
        
        # 绘图处理（如果需要，可以考虑跳过或异步处理）
        if config['plot']:
            self._handle_plotting_optimized(config)

        self.gpumd_dir = os.path.join(self.iter_dir, "01.gpumd")
        
        # 预先获取和排序所有任务目录，避免多次调用
        task_dirs = self._get_cached_task_dirs()
        
        dlog.info(f"-----start analysis the trajectory-----"
                f"Processing {len(task_dirs)} tasks with config: {config['summary']}")

        # 核心优化：减少循环内的重复计算和I/O
        results = self._optimized_task_processing(task_dirs, nep_dir, config)
        
        # 快速检查失败任务
        if self._quick_failure_check(results, config):
            return
        
        # 批量输出处理
        self._batch_output_processing(results, config)
        
        # 更新步数和状态
        self._finalize_iteration(config)

    def _get_cached_config(self):
        """一次性获取所有配置，避免重复查询"""
        model_devi = self.get_model_devi()
        threshold = model_devi.get("uncertainty_threshold") or self.idata.get("uncertainty_threshold", [0.3, 1])
        energy_threshold = self.idata.get("energy_threshold", None)
        
        config = {
            'plot': self.idata.get("gpumd_plt", True),
            'threshold': threshold,
            'energy_threshold': energy_threshold,
            'mode': self.idata.get("uncertainty_mode", "mean"),
            'level': self.idata.get("uncertainty_level", 1),
            'sample_method': self.idata.get("sample_method", "relative"),
            'continue_from_old': self.idata.get("continue_from_old", False),
            'max_candidate': self.idata.get("max_candidate", 1000),
            'max_temp': self.idata.get("max_temp", 10000),
            'shortest_d': self.idata.get("shortest_d", 0.5),
            'analyze_range': self.idata.get("analyze_range", [0.5, 1.0]),
            'max_run_steps': self.idata.get("max_run_steps", 1200000),
            'max_iter': self.idata.get("max_iter", 20),
            'time_step': self.idata.get("time_step")
        }
        
        # 预计算一些常用值
        config['max_candidate_per_task'] = config['max_candidate'] // len(self._get_cached_task_dirs())
        config['summary'] = f"threshold:{threshold}, energy_threshold:{energy_threshold}, max_temp:{config['max_temp']}"
        dlog.info(f"Config summary: {config['summary']}")
        return config

    def _get_cached_task_dirs(self):
        """缓存任务目录列表"""
        model_devi_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
        self._cached_task_dirs = sorted([
            os.path.join(model_devi_dir, task) 
            for task in glob(f"{model_devi_dir}/task.[0-9][0-9][0-9][0-9][0-9][0-9]")
        ])
        return self._cached_task_dirs

    def _handle_plotting_optimized(self, config):
        """优化绘图处理，可选择跳过或异步"""
        # 如果绘图不重要，可以考虑跳过以节省时间
            
        model_devi_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd"
        task_dirs = self._get_cached_task_dirs()
        
        current_dir = os.getcwd()
        try:
            dlog.info(f"plotting {len(task_dirs)} tasks")
            for task_dir in task_dirs:
                os.chdir(task_dir)
                try:
                    gpumdplt(self.total_time, config['time_step'])
                except Exception as e:
                    dlog.error(f"plotting {task_dir} failed: {e}")
                    # 可以选择继续而不是抛出异常
                    continue
        finally:
            os.chdir(current_dir)

    def _optimized_task_processing(self, task_dirs, nep_dir, config):
        """优化的任务处理循环"""
        # 预分配结果容器
        results = {
            'failed_indices': [],
            'thermo_averages': [],
            'all_candidates': [],
            'statistics': {'accurate': 0, 'candidate': 0, 'total': 0}
        }
        
        # 预先创建输出目录和文件
        label_dir = os.path.join(self.iter_dir, "02.label")
        os.makedirs(label_dir, exist_ok=True)
        candidate_file = os.path.join(label_dir, "candidate.xyz")
        if os.path.exists(candidate_file):
            os.remove(candidate_file)
        
        # 预计算条件函数，避免在循环中重复创建
        def make_candidate_condition(frame_prop):
            return (
                ((frame_prop[:, 1] >= config['threshold'][0]) & 
                (frame_prop[:, 1] <= config['threshold'][1])) |
                (frame_prop[:, 2] > config['energy_threshold'])
            ) & (frame_prop[:, 3] > config['shortest_d']) & (frame_prop[:, 4] < config['max_temp'])
        
        def make_accurate_condition(frame_prop):
            return (frame_prop[:, 1] < config['threshold'][0]) & (frame_prop[:, 2] < config['energy_threshold'])
        
        # 格式化字符串预定义

        fmt = "%14d"+"%12.2f"*9+"%12.4f"
        header = f"{'indices':>14} {'time':^14} {'relative_error':^14} {'energy_error':^14} {'shortest_d':^14} {'temperature':^14} {'potential':^14} {'pressure':^14} {'volume':^14} {'molecule_num':^14} {'molecule_density':^14}"
        
        # 主循环优化
        current_dir = os.getcwd()
        analyze_start = config['analyze_range'][0]
        analyze_end = config['analyze_range'][1]
        
        # 预先打开候选文件以追加模式写入
        with open(candidate_file, 'w') as candidate_f:
            for ii, task_dir in enumerate(task_dirs):
                # try:
                os.chdir(task_dir)
                dlog.info(f"processing task {ii}")
                
                # 核心计算（这部分您说很快）
                atoms_list, frame_property, failed_row_index = Nepactive_OB.relative_force_error(
                    total_time=self.total_time, 
                    nep_dir=nep_dir, 
                    mode=config['mode'],
                    level=config['level'], 
                    allowed_max_temp=config['max_temp'], 
                    allowed_shortest_distance=config['shortest_d']
                )
                
                # 快速保存最后一帧
                write_extxyz(os.path.join(task_dir, "final.xyz"), atoms_list[-1])
                
                # 优化：预计算数组切片
                prop_len = len(frame_property)
                start_idx = int(analyze_start * prop_len)
                end_idx = int(analyze_end * prop_len)
                
                # 热力学平均值计算
                thermo_avg = np.mean(frame_property[start_idx:end_idx, 1:], axis=0, keepdims=True)

                p_rmse = np.sqrt(np.mean((frame_property[start_idx:end_idx, 6]-thermo_avg[0,5])**2))
                p_mae = np.mean(np.abs(frame_property[start_idx:end_idx, 6]-thermo_avg[0,5]))
                p_rmse = p_rmse.reshape(-1, 1)
                p_mae = p_mae.reshape(-1, 1)
                thermo_avg = np.hstack((thermo_avg, p_rmse, p_mae))
                
                results['thermo_averages'].append(thermo_avg)
                
                # 使用预定义的条件函数
                candidate_condition = make_candidate_condition(frame_property)
                candidate_indices = np.where(candidate_condition)[0]
                
                accurate_condition = make_accurate_condition(frame_property)
                accurate_count = np.sum(accurate_condition)
                
                # 更新统计信息
                results['statistics']['accurate'] += accurate_count
                results['statistics']['candidate'] += len(candidate_indices)
                results['statistics']['total'] += prop_len
                results['failed_indices'].append(failed_row_index)
                
                # 候选帧处理优化
                if len(candidate_indices) > 0:
                    # 使用高级索引一次性获取数据
                    filtered_rows = frame_property[candidate_indices]
                    indices_with_data = np.column_stack((candidate_indices, filtered_rows))
                    
                    # 选择最佳候选
                    if len(candidate_indices) > config['max_candidate_per_task']:
                        # 使用 argpartition 而不是完全排序，更快
                        n_select = config['max_candidate_per_task']
                        partition_idx = np.argpartition(indices_with_data[:, 2], -n_select)[-n_select:]
                        selected_data = indices_with_data[partition_idx]
                    else:
                        selected_data = indices_with_data
                    
                    # 按索引排序
                    final_data = selected_data[selected_data[:, 0].argsort()]
                    
                    # 批量获取候选原子
                    selected_indices = final_data[:, 0].astype(int)
                    candidate_atoms = [atoms_list[idx] for idx in selected_indices]
                    
                    # 直接写入文件，避免内存累积
                    write_extxyz(candidate_f, candidate_atoms)
                    
                    # 保存候选数据
                    np.savetxt(f"candidate_{ii}.txt", final_data, fmt=fmt, header=header, comments=f"_{ii}_")
                    
                # except Exception as e:
                #     dlog.error(f"Error in task {ii}: {e}")
                #     raise RuntimeError(f"Task {ii} failed with error: {e}")
                # finally:
                    os.chdir(current_dir)
        
        return results

    def _quick_failure_check(self, results, config):
        """快速检查失败任务"""
        failed_indices = np.array(results['failed_indices'])
        if len(failed_indices) == 0:
            return False
            
        # 使用第一个任务的长度作为参考
        frame_len = results['statistics']['total'] // len(failed_indices) if len(failed_indices) > 0 else 0
        failed_threshold = int(0.8 * frame_len)
        
        early_failures = failed_indices < failed_threshold
        if np.any(early_failures):
            min_failed = np.min(failed_indices)
            failed_tasks = np.array(self._get_cached_task_dirs())[early_failures]
            
            self.run_steps = max(22000, int(self.run_steps * min_failed / frame_len))
            dlog.info(f"Early failures detected at indices {failed_indices[early_failures]}")
            
            self.handle_bad_job(
                failed_row_indices=failed_indices[early_failures],
                failed_task_dirs=failed_tasks
            )
            self.write_steps()
            return True
        
        return False

    def _batch_output_processing(self, results, config):
        """批量处理输出"""
        # 处理热力学数据
        if results['thermo_averages']:
            thermo_data = np.vstack(results['thermo_averages'])
            thermo_file = f'{self.work_dir}/thermo.txt'
            
            with open(thermo_file, 'a') as f:
                np.savetxt(f, thermo_data, 
                        fmt="%24.2f"+"%14.2f"*6+"%14.4f"*2+"%14.2f"*2,
                        header=f"{'relative_error':^14} {'energy_error':^14} {'shortest_d':^14} {'temperature':^14} {'potential':^14} {'pressure':^14} {'volume':^14} {'molecule_num':^14} {'molecule_density':^14} {'P_RMSE':^14} {'P_MAE':^14}",
                        comments=f"#iter.{self.ii:06d}")
        
        # 计算和记录比例
        stats = results['statistics']
        if stats['total'] > 0:
            accurate_ratio = stats['accurate'] / stats['total']
            candidate_ratio = stats['candidate'] / stats['total']
            failed_ratio = 1 - accurate_ratio - candidate_ratio
            dlog.info(f"Ratios - failed: {failed_ratio:.4f}, candidate: {candidate_ratio:.4f}, accurate: {accurate_ratio:.4f}")

    def _finalize_iteration(self, config):
        """完成迭代处理"""
        if self.run_steps > config['max_run_steps']:
            if self.ii >= config['max_iter']:
                dlog.info(f"Reached max iteration: {config['max_iter']}, finished")
                exit()
            else:
                self.run_steps = config['max_run_steps'] / self.run_steps_factor
        
        self.write_steps()
        dlog.info("All frames processed successfully")
    
    def handle_bad_job(self,failed_row_indices=None,failed_task_dirs=None,allowed_shortest_distance=0.5, run_temp = 1500):
        """
        rerun the failed task
        """
        work_dir = f"{self.work_dir}/iter.{self.ii:06d}/01.gpumd/mattersim"
        if not os.path.exists(work_dir):
            os.makedirs(work_dir,exist_ok=True)
        os.chdir(work_dir)
        dlog.info(f"the failed job will be rerun in {os.getcwd()}")
        task_dirs = []
        os.system(f"rm -rf task.*")
        for ii,failed_row_index in enumerate(failed_row_indices):
            if failed_row_index < 3:
                raise ValueError(f"the first 4 frames are failed, please check the input of {failed_task_dirs[ii]}")
            task_dir = os.path.join(work_dir, f"task.{ii:06d}")
            task_dirs.append(task_dir)
            os.makedirs(task_dir, exist_ok=True)
            os.chdir(task_dir)

            dlog.info(f"the {failed_row_index-2} frames of {failed_task_dirs[ii]} will be rerun")
            atoms = read(os.path.join(failed_task_dirs[ii], "dump.xyz"), index = failed_row_index-2)
            struc_file = os.path.abspath(os.path.join(task_dir, "POSCAR"))
            write(struc_file, atoms)
            py_file = continue_pytemplate.format(structure = struc_file,temperature = run_temp, steps = 20000)
            with open(os.path.join(task_dir, "ensemble.py"), "w",encoding="utf-8") as f:
                f.write(py_file)

        os.chdir(work_dir)
        task_dirs = [os.path.abspath(task_dir) for task_dir in glob("task.*")]
        
        task_dirs.sort()

        sorted_task_dirs = copy.deepcopy(task_dirs)

        for task_dir in task_dirs:
            os.chdir(task_dir)
            if os.path.exists("task_finished"):
                sorted_task_dirs.remove(task_dir)
                dlog.info(f"the task {task_dir} has been finished")

        os.chdir(work_dir)

        self.run_pytasks(sorted_task_dirs)
        
        os.chdir(work_dir)
        trajs = []
        for task_dir in task_dirs:
            traj_file = os.path.join(task_dir, "out.traj")
            traj = read(traj_file, index = ":")
            trajs.extend(traj)
            
        self.max_candidate = self.idata.get("max_candidate", 1000)
        
        if len(trajs) > self.max_candidate:
            # 随机抽取max_candidate个样本
            random_indices = random.sample(range(len(trajs)), self.max_candidate)
            random_indices.sort()  # 可选，保持帧的顺序
            trajs = [trajs[i] for i in random_indices]
        shortest_distances = [] 
        for ii, atoms in enumerate(trajs):
            shortest_distance = get_shortest_distance(atoms)
            shortest_distances.append(shortest_distance)
        np.savetxt("shortest_distance.txt", shortest_distances, fmt = "%12.2f")
        if np.any(np.array(shortest_distances) < allowed_shortest_distance):
            dlog.error(f"some frames have shortest distance less than {allowed_shortest_distance}")
            raise ValueError(f"some frames have shortest distance less than {allowed_shortest_distance}")

        label_dir = os.path.join(self.iter_dir, "02.label")
        os.makedirs(label_dir, exist_ok=True)
        os.chdir(label_dir)
        with open("candidate.xyz", "a") as f:
            write_extxyz(f, trajs)

    def run_pytasks(self, task_dirs):
        """
        运行生成python任务
        """
        self.gpu_available = self.idata.get("gpu_available", [0, 1, 2, 3])
        self.gpu_per_task = self.idata.get("gpu_per_task", 1)
        python_interpreter = self.idata.get("python_interpreter")
        processes = []
        
        # 限制同时运行的任务数量
        max_concurrent_tasks = len(self.gpu_available)  # 或者其他合理的数量
        
        for i in range(0, len(task_dirs), max_concurrent_tasks):
            batch_dirs = task_dirs[i:i + max_concurrent_tasks]
            batch_processes = []
            
            for task_dir in batch_dirs:
                os.chdir(task_dir)
                if os.path.exists("task_finished"):
                    dlog.warning(f"{task_dir} has already been finished, skip it")
                    continue
                    
                basename = "ensemble.py"
                gpu_id = self.gpu_available[len(batch_processes) % len(self.gpu_available)]
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                
                log_file = os.path.join(task_dir, 'log')
                try:
                    with open(log_file, 'w') as log:
                        process = subprocess.Popen(
                            [python_interpreter, basename],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env=env
                        )
                        batch_processes.append((process, task_dir))
                except Exception as e:
                    dlog.error(f"Failed to start process in {task_dir}: {str(e)}")
                    raise Exception(f"Failed to start process in {task_dir}: {str(e)}")

            
            # 添加超时机制和健康检查
            timeout = 3600  # 设置合理的超时时间（秒）
            start_time = time.time()
            
            while batch_processes and time.time() - start_time < timeout:
                for i, (process, task_dir) in enumerate(batch_processes[:]):
                    ret = process.poll()
                    if ret is not None:  # 进程已结束
                        if ret != 0:
                            dlog.error(f"Process failed with return code {ret}. Check log at: {task_dir}/log")
                        else:
                            try:
                                os.chdir(task_dir)
                                ase_plt()
                                os.system(f"touch {task_dir}/task_finished")
                                dlog.info(f"Process completed successfully. Log at: {task_dir}/log")
                            except Exception as e:
                                dlog.error(f"Post-processing failed for {task_dir}: {str(e)}")
                                raise Exception(f"Post-processing failed for {task_dir}: {str(e)}")
                        
                        batch_processes.pop(i)
                
                # 防止 CPU 空转
                if batch_processes:
                    time.sleep(1)
            
            # 处理超时的进程
            for process, task_dir in batch_processes:
                if process.poll() is None:  # 进程仍在运行
                    dlog.warning(f"Process in {task_dir} timed out, terminating...")
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        dlog.error(f"Process in {task_dir} still running after termination, killing...")
                        process.kill()  # 强制终止
                    dlog.error(f"Process in {task_dir} terminated due to timeout")
        
    @classmethod
    def relative_force_error(cls, total_time, nep_dir = None, mode:str = "mean", level = 1 ,allowed_shortest_distance = 0.5, allowed_max_temp = 10000):
        """
        return frame_index dict for accurate, candidate, failed
        the candidate_index may be more than the needed, need to resample
        """

        if os.path.exists("frame_property.txt"):
            frame_property = np.loadtxt("frame_property.txt")

            if os.path.exists("failed_index.txt"):
                with open("failed_index.txt","r") as f:
                    failed_index = int(f.read())
            else :
                failed_index = len(frame_property)

            atoms_list = read(f"dump.xyz", index = ":")
            return  atoms_list, frame_property, failed_index

        if not nep_dir:
            nep_dir = os.getcwd()
        calculator_fs = glob(f"{nep_dir}/**/nep*.txt")
        atoms_list = read(f"dump.xyz", index = ":")
        
        f_lists = force_main(atoms_list, calculator_fs)
        property_list = []
        time_list = np.linspace(0, total_time, len(atoms_list), endpoint=False)/1000
        for ii,atoms in tqdm(enumerate(atoms_list)):
            f_list = [item[ii] for item in f_lists]
            energy_list = [item[1] for item in f_list]
            f_list = [item[0] for item in f_list]
            f_avg = np.average(f_list,axis=0)
            df_sqr = np.sqrt(np.mean(np.square([(f_list[i] - f_avg) for i in range(len(f_list))]).sum(axis=2),axis=0))
            abs_f_avg = np.mean(np.sqrt(np.square(f_list).sum(axis=2)),axis=0)
            relative_error = (df_sqr/(abs_f_avg+level))
            if mode == "mean":
                relative_error = np.mean(relative_error)
            elif mode == "max":
                relative_error = np.max(relative_error)
            energy_error = np.sqrt(np.mean(np.power(np.array(energy_list)-np.mean(energy_list),2)))
            shortest_distance = get_shortest_distance(atoms)
            property_list.append([time_list[ii], relative_error, energy_error, shortest_distance])
        # property_list
        property_list_np = np.array(property_list)
        thermo = np.loadtxt("thermo.out")
        thermo_new = compute_volume_from_thermo(thermo)[:,[0,2,3,-1]]
        frame_property = np.concatenate((property_list_np,thermo_new),axis=1)

        molecule_num = analyze_trajectory("dump.xyz", index=":").sum(axis=1).to_numpy()
        molecule_density = molecule_num / frame_property[:,7]
        frame_property = np.hstack((frame_property, molecule_num.reshape(-1, 1)))
        frame_property = np.hstack((frame_property, molecule_density.reshape(-1, 1)))
        
        temperatures = thermo[:,0]
        shortest_distances = frame_property[:,3]
        result = np.where(np.logical_or(temperatures > allowed_max_temp, shortest_distances < allowed_shortest_distance))
        if result[0].size > 0:
            # 获取第一个大于6000的数的行索引
            first_row_index = result[0][0]
        else:
            first_row_index = len(frame_property)  # 如果没有找到任何大于6000的数

        fmt = "%12.2f"*9+"%12.4f"
        header = f"{'time':^9} {'relative_error':^14} {'energy_error':^14} {'shortest_d':^14} {'temperature':^14} {'potential':^14} {'pressure':^14} {'volume':^14} {'molecule_num':^14} {'molecule_density':^14}"
        np.savetxt("frame_property.txt", frame_property, fmt = fmt, header = header)
        
        if first_row_index != len(frame_property):
            new_total_time = time_list[first_row_index-1]*1000
            dlog.warning(f"thermo.out has temperature > {allowed_max_temp} K or shortest distance < {allowed_shortest_distance} too early at frame {first_row_index}({new_total_time} ps), the gpumd task should be rerun")
            np.savetxt(f"failed_index.txt", [first_row_index], fmt="%12d")
            return atoms_list, frame_property, first_row_index
            
        return atoms_list, frame_property, first_row_index

    def make_label_task(self):
        label_engine = self.idata.get("label_engine","mattersim")
        if label_engine == "mattersim":
            self.make_mattersim_task()
        elif label_engine == "vasp":
            self.make_vasp_task()
        else:
            raise NotImplementedError(f"The label engine {label_engine} is not implemented")        

    def make_mattersim_task(self):
        """
        change the calculator to mattersim, and write the train.xyz
        """

        # 将工作目录设置为iter_dir下的02.label文件夹
        iter_dir = self.iter_dir
        work_dir = os.path.join(iter_dir, "02.label")
        train_ratio = self.idata.get("training_ratio", 0.8)
        os.chdir(work_dir)
        atoms_list = read("candidate.xyz", index=":", format="extxyz")
        self.pot_file = self.idata.get("pot_file")
        # 创建MatterSimCalculator对象，用于计算原子能量
        Nepactive_OB.run_mattersim(atoms_list=atoms_list, pot_file=self.pot_file, train_ratio=train_ratio)

    @classmethod
    def run_mattersim(cls, atoms_list:List[Atoms], pot_file:str, train_ratio:float=0.8,tqdm_use:Optional[bool]=False):
        calculator = MatterSimCalculator(load_path=pot_file,device="cuda")
        if os.path.exists("candidate.traj"):
            os.remove("candidate.traj")
        traj = Trajectory('candidate.traj', mode='a')

        def change_calc(atoms:Atoms):
            atoms._calc=calculator
            atoms.get_potential_energy()
            traj.write(atoms)
            return atoms

        if tqdm_use:
            atoms = [change_calc(atoms_list[i]) for i in tqdm(range(len(atoms_list)))]
        else:
            atoms = [change_calc(atoms_list[i]) for i in range(len(atoms_list))]   
        # 读取Trajectory对象中的原子信息
        atoms = read("candidate.traj",index=":")
        train:List[Atoms]=[]
        test:List[Atoms]=[]
        failed:List[Atoms]=[]
        failed_index=[]
        for i in range(len(atoms)):
            rand=random.random()
            if np.max(np.abs(atoms[i].get_forces())) > 60:
                failed.append(atoms[i])
                failed_index.append(i)
            elif rand <= train_ratio:
                train.append(atoms[i])
            elif rand > train_ratio:
                test.append(atoms[i])
            else:
                dlog.warning(f"{atoms[i]}failed to be classified")
        if train:
            write_extxyz("iter_train.xyz",train)
        if test:
            write_extxyz("iter_test.xyz",test)
        if failed:
            write_extxyz("iter_failed.xyz",failed)
            np.savetxt("failed_index.txt",failed_index,fmt="%12d")
        # 将原子信息写入train_iter.xyz文件
        dlog.warning(f"failed structures:{len(failed_index)}")
        del calculator
        empty_cache()

    def make_vasp_task(self):
        task = Remotetask(iternum = self.ii, idata = self.idata, trajfile = self.trajfile)
        task.run_submission(jj=3)

    def run_label_task(self):
        label_engine = self.idata.get("label_engine","mattersim")
        if label_engine == "mattersim":
            return
        elif label_engine == "vasp":
            self.run_vasp_task()

    def run_vasp_task(self):
        assert os.path.isabs(self.iter_dir)
        os.chdir(f"{self.iter_dir}/02.label")
        if task is None:
            task = Remotetask(idata = self.idata)
        task.run_submission(jj=4)

    def write_steps(self):
        with open(f"{self.work_dir}/steps.txt","a") as f:
            f.write(f"{int(self.run_steps):12d}\n")
        np.savetxt(f"{self.work_dir}/iter.{self.ii:06d}/steps.txt",np.array([self.run_steps]),fmt="%12d")