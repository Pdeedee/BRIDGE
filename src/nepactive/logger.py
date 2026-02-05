"""Logging for molecular dynamics."""
import weakref
import time
from typing import IO, Any, Union

from ase import Atoms, units
from ase.parallel import world
from ase.utils import IOContext


class MDLogger(IOContext):
    def __init__(
        self,
        dyn: Any,
        atoms: Atoms,
        logfile: Union[IO, str],
        header: bool = True,
        stress: bool = True,
        peratom: bool = False,
        volume: bool = True,
        speed: bool = True,  # 新增：是否显示速度
        mode: str = "a",
        comm=world,
    ):
        self.dyn = weakref.proxy(dyn) if hasattr(dyn, "get_time") else None
        self.atoms = atoms
        global_natoms = atoms.get_global_number_of_atoms()
        self.logfile = self.openfile(file=logfile, mode=mode, comm=comm)
        self.stress = stress
        self.peratom = peratom
        self.volume = volume
        self.speed = speed
        
        # 速度监控相关
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_step = 0
        self.call_count = 0
        
        if self.dyn is not None:
            self.hdr = "%-9s " % ("Time[ps]",)
            self.fmt = "%-10.4f "
        else:
            self.hdr = ""
            self.fmt = ""
        if self.peratom:
            self.hdr += "%12s %12s %12s  %6s" % ("Etot/N[eV]", "Epot/N[eV]",
                                                 "Ekin/N[eV]", "T[K]")
            self.fmt += "%12.4f %12.4f %12.4f  %6.1f"
        else:
            self.hdr += "%12s %12s %12s  %6s" % ("Etot[eV]", "Epot[eV]",
                                                 "Ekin[eV]", "T[K]")
            if global_natoms <= 100:
                digits = 4
            elif global_natoms <= 1000:
                digits = 3
            elif global_natoms <= 10000:
                digits = 2
            else:
                digits = 1
            self.fmt += 3 * ("%%12.%df " % (digits,)) + " %6.1f"
        if self.volume:
            self.hdr += "  %12s" % "V[Å^3]"
            self.fmt += " %12.4f"
        if self.stress:
            self.hdr += ('      ---------------------- stress [GPa] '
                         '-----------------------')
            self.fmt += 6 * " %10.3f"
        if self.speed:
            self.hdr += "  %10s %10s" % ("step/s", "ns/day")
            self.fmt += " %10.2f %10.4f"
        self.fmt += "\n"
        if header:
            self.logfile.write(self.hdr + "\n")

    def __del__(self):
        self.close()

    def __call__(self):
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = self.atoms.get_temperature()
        global_natoms = self.atoms.get_global_number_of_atoms()
        if self.peratom:
            epot /= global_natoms
            ekin /= global_natoms
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000 * units.fs)
            dat = (t,)
        else:
            dat = ()
        dat += (epot + ekin, epot, ekin, temp)
        if self.volume:
            dat += (self.atoms.get_volume(),)
        if self.stress:
            dat += tuple(self.atoms.get_stress(
                include_ideal_gas=True) / units.GPa)
        
        if self.speed:
            current_time = time.time()
            self.call_count += 1
            
            # 计算瞬时速度（基于上次调用的间隔）
            dt = current_time - self.last_time
            if dt > 0 and self.dyn is not None:
                # 获取当前步数
                current_step = self.dyn.nsteps if hasattr(self.dyn, 'nsteps') else self.call_count
                dsteps = current_step - self.last_step
                steps_per_sec = dsteps / dt if dsteps > 0 else 0
                
                # 计算 ns/day
                timestep_fs = self.dyn.dt / units.fs if hasattr(self.dyn, 'dt') else 0.2
                ns_per_day = steps_per_sec * timestep_fs * 1e-6 * 86400
                
                dat += (steps_per_sec, ns_per_day)
                
                self.last_time = current_time
                self.last_step = current_step
            else:
                dat += (0.0, 0.0)

        self.logfile.write(self.fmt % dat)
        self.logfile.flush()