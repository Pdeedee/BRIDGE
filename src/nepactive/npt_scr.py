"""
NPT ensemble using Stochastic Cell Rescaling (SCR) barostat
combined with BDP (Bussi-Donadio-Parrinello) thermostat.

Ported from GPUMD: src/integrate/ensemble_npt_scr.cu + svr_utilities.cuh

References:
[1] Mattia Bernetti and Giovanni Bussi,
    Pressure control using stochastic cell rescaling,
    J. Chem. Phys. 153, 114107 (2020).

[2] G. Bussi, D. Donadio, and M. Parrinello,
    Canonical sampling through velocity rescaling,
    J. Chem. Phys. 126, 014101 (2007).
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union, IO, List
from ase import Atoms, units
from ase.md.md import MolecularDynamics

K_B = units.kB  # eV/K


# ============================================================
# BDP thermostat utilities (from svr_utilities.cuh)
# ============================================================

def gasdev(rng: np.random.Generator) -> float:
    """Standard normal deviate. Matches GPUMD gasdev()."""
    return rng.standard_normal()


def gamdev(ia: int, rng: np.random.Generator) -> float:
    """
    Gamma deviate with integer shape parameter.
    Matches GPUMD gamdev() from svr_utilities.cuh.
    """
    if ia < 1:
        return 0.0
    if ia < 6:
        x = 1.0
        for _ in range(ia):
            x *= rng.random()
        return -np.log(x)
    else:
        while True:
            while True:
                v1 = rng.random()
                v2 = 2.0 * rng.random() - 1.0
                if v1 * v1 + v2 * v2 <= 1.0:
                    break
            y = v2 / v1
            am = ia - 1
            s = np.sqrt(2.0 * am + 1.0)
            x = s * y + am
            if x > 0:
                break
        e = (1.0 + y * y) * np.exp(am * np.log(x / am) - s * y)
        while rng.random() > e:
            while True:
                while True:
                    v1 = rng.random()
                    v2 = 2.0 * rng.random() - 1.0
                    if v1 * v1 + v2 * v2 <= 1.0:
                        break
                y = v2 / v1
                am = ia - 1
                s = np.sqrt(2.0 * am + 1.0)
                x = s * y + am
                if x > 0:
                    break
            e = (1.0 + y * y) * np.exp(am * np.log(x / am) - s * y)
        return x


def resamplekin_sumnoises(nn: int, rng: np.random.Generator) -> float:
    """
    Sum of nn independent squared Gaussian noises.
    Matches GPUMD resamplekin_sumnoises().
    """
    if nn == 0:
        return 0.0
    elif nn == 1:
        rr = gasdev(rng)
        return rr * rr
    elif nn % 2 == 0:
        return 2.0 * gamdev(nn // 2, rng)
    else:
        rr = gasdev(rng)
        return 2.0 * gamdev((nn - 1) // 2, rng) + rr * rr


def resamplekin(kk: float, sigma: float, ndeg: int, taut: float,
                rng: np.random.Generator) -> float:
    """
    Resample kinetic energy (BDP algorithm).
    Matches GPUMD resamplekin() exactly.

    Parameters
    ----------
    kk : current kinetic energy (eV)
    sigma : target kinetic energy = ndeg * kB * T / 2 (eV)
    ndeg : degrees of freedom
    taut : thermostat coupling (dimensionless, in units of call frequency)
    rng : random number generator
    """
    if taut > 0.1:
        factor = np.exp(-1.0 / taut)
    else:
        factor = 0.0
    rr = gasdev(rng)
    return (kk
            + (1.0 - factor) * (sigma * (resamplekin_sumnoises(ndeg - 1, rng) + rr * rr) / ndeg - kk)
            + 2.0 * rr * np.sqrt(kk * sigma / ndeg * (1.0 - factor) * factor))


# ============================================================
# Barostat functions (from ensemble_npt_scr.cu)
# All pressures in ASE internal units (eV/Å³)
# ============================================================

def cpu_pressure_isotropic(rng, target_pressure, p_coupling, p_diag,
                           volume, target_temperature):
    """
    Isotropic SCR barostat. Returns scalar scale_factor.
    Matches GPUMD cpu_pressure_isotropic().
    """
    p_instant = (p_diag[0] + p_diag[1] + p_diag[2]) / 3.0
    scale_berendsen = 1.0 - p_coupling[0] * (target_pressure[0] - p_instant)
    # Factor 2/3: 3 directions coupled
    scale_stochastic = (
        np.sqrt(2.0 / 3.0 * p_coupling[0] * K_B * target_temperature / volume)
        * gasdev(rng)
    )
    return scale_berendsen + scale_stochastic


def cpu_pressure_orthogonal(rng, deform, deform_rate, cell_diag,
                            target_temperature, target_pressure,
                            p_coupling, p_diag, volume):
    """
    Orthogonal SCR barostat. Returns scale_factors[3].
    Matches GPUMD cpu_pressure_orthogonal().
    """
    scale_factors = np.ones(3)
    for i in range(3):
        if deform[i]:
            old_len = cell_diag[i]
            scale_factors[i] = (old_len + deform_rate[i]) / old_len
        else:
            scale_berendsen = 1.0 - p_coupling[i] * (target_pressure[i] - p_diag[i])
            scale_stochastic = (
                np.sqrt(2.0 * p_coupling[i] * K_B * target_temperature / volume)
                * gasdev(rng)
            )
            scale_factors[i] = scale_berendsen + scale_stochastic
    return scale_factors


def cpu_pressure_triclinic(rng, target_temperature, target_pressure,
                           p_coupling, stress_voigt, volume):
    """
    Triclinic SCR barostat. Returns mu[3x3].
    Matches GPUMD cpu_pressure_triclinic().

    Note on index mapping:
      GPUMD thermo order: [xx, yy, zz, xy, xz, yz]
      ASE Voigt order:    [xx, yy, zz, yz, xz, xy]
      target_pressure/p_coupling: Voigt order [xx, yy, zz, yz, xz, xy]

    GPUMD source (line 174-179):
      mu[0,0] = 1 - tau[0]*(p0[0] - p[0])   # xx
      mu[1,1] = 1 - tau[1]*(p0[1] - p[1])   # yy
      mu[2,2] = 1 - tau[2]*(p0[2] - p[2])   # zz
      mu_xy = -tau[5]*(p0[5] - p[3])         # xy: voigt[5], thermo[3]
      mu_xz = -tau[4]*(p0[4] - p[4])         # xz: voigt[4], thermo[4]
      mu_yz = -tau[3]*(p0[3] - p[5])         # yz: voigt[3], thermo[5]

    Since ASE gives us Voigt order directly, stress_voigt[5]=xy, etc.
    """
    p0 = target_pressure  # Voigt: [xx, yy, zz, yz, xz, xy]
    p = stress_voigt       # Voigt: [xx, yy, zz, yz, xz, xy]
    tau = p_coupling       # Voigt: [xx, yy, zz, yz, xz, xy]

    mu = np.eye(3)

    # Diagonal: xx, yy, zz
    for i in range(3):
        mu[i, i] = 1.0 - tau[i] * (p0[i] - p[i])

    # Off-diagonal (symmetric)
    # xy: Voigt index 5 -> mu[0,1], mu[1,0]
    mu[0, 1] = mu[1, 0] = -tau[5] * (p0[5] - p[5])
    # xz: Voigt index 4 -> mu[0,2], mu[2,0]
    mu[0, 2] = mu[2, 0] = -tau[4] * (p0[4] - p[4])
    # yz: Voigt index 3 -> mu[1,2], mu[2,1]
    mu[1, 2] = mu[2, 1] = -tau[3] * (p0[3] - p[3])

    # Stochastic part: p_coupling as 3x3 matrix
    tau_3x3 = np.array([
        [tau[0], tau[5], tau[4]],
        [tau[5], tau[1], tau[3]],
        [tau[4], tau[3], tau[2]],
    ])
    for r in range(3):
        for c in range(3):
            mu[r, c] += (
                np.sqrt(2.0 * tau_3x3[r, c] * K_B * target_temperature / volume)
                * gasdev(rng)
            )

    return mu


# 默认体积模量 (GPa)，含能材料典型值 10~30 GPa
# RDX ~12, HMX ~15, CL-20 ~18, TATB ~16, TNT ~11
# 取 15 GPa 作为通用默认值
DEFAULT_BULK_MODULUS = 15.0  # GPa


def _tau_p_to_coupling(tau_p, elastic_modulus=DEFAULT_BULK_MODULUS):
    """
    将 GPUMD 风格的 tau_p (单位 timestep) 转换为 SCR 内部 p_coupling。
    与 GPUMD integrate.cu 一致:
        p_coupling = PRESSURE_UNIT_CONVERSION / (tau_p * 3.0 * elastic_modulus)
    其中 PRESSURE_UNIT_CONVERSION = 160.2177 (eV/Å³ <-> GPa)
    """
    PRESSURE_UNIT_CONVERSION = 160.2177
    return PRESSURE_UNIT_CONVERSION / (tau_p * 3.0 * elastic_modulus)


# ============================================================
# NPT_SCR class
# ============================================================

class NPT_SCR(MolecularDynamics):
    """
    NPT MD using SCR barostat + BDP thermostat.
    Ported from GPUMD ensemble_npt_scr.

    Parameters
    ----------
    atoms : Atoms
    timestep : float, ASE 时间单位 (fs * units.fs)
    temperature : float, 目标温度 (K)
    pressure : float or list, 目标压力 (GPa)，内部转换为 eV/Å³
    tau_t : float, BDP 恒温器弛豫时间，单位 timestep 数。
        与 GPUMD 的 tau_T 一致，默认 100。越大恒温越弱。
    tau_p : float, SCR 气压计弛豫时间，单位 timestep 数。
        与 GPUMD 的 tau_p 一致，默认 2000。
        内部通过 1/(tau_p * 3 * bulk_modulus) 转换为耦合系数。
    elastic_modulus : float, 体积模量 (GPa)，默认 15 GPa（含能材料典型值）。
        一般不需要用户输入。
    pmode : str, 压力模式: 'iso'(各向同性) / 'ortho'(正交) / 'tri'(三斜)
    deform : list or None, 形变速率 [rate_x, rate_y, rate_z] (Å/step)
    seed : int or None, 随机数种子
    """

    def __init__(self, atoms, timestep, temperature, pressure,
                 tau_t=100.0,                              # 单位 timestep，与 GPUMD tau_T 一致
                 tau_p=2000.0,                             # 单位 timestep，与 GPUMD tau_p 一致
                 elastic_modulus=DEFAULT_BULK_MODULUS,      # GPa，默认 15
                 pmode='iso', deform=None, seed=None,
                 # --- 渐变参数 ---
                 run_steps=None,           # 总步数，用于计算渐变进度
                 t_start=None,             # 起始温度 (K)，默认为 temperature
                 t_stop=None,              # 终止温度 (K)，默认为 temperature
                 t_schedule=None,          # 自定义温度函数 callable(progress) -> T(K)
                 p_start=None,             # 起始压力 (GPa)，默认为 pressure
                 p_stop=None,              # 终止压力 (GPa)，默认为 pressure
                 p_schedule=None,          # 自定义压力函数 callable(progress) -> P(GPa)
                 v_start=1.0,              # 起始相对体积，1.0 = 初始体积
                 v_stop=None,              # 终止相对体积，None = 不做体积渐变
                 v_schedule=None,          # 自定义体积函数 callable(progress) -> relative_volume
                 **kwargs):
        super().__init__(atoms=atoms, timestep=timestep, **kwargs)

        self.temp_coupling = tau_t
        self.pmode = pmode.lower() if pmode is not None else None
        self.natoms = len(atoms)
        self.rng = np.random.default_rng(seed)

        # 温度渐变设置
        self.t_start = t_start if t_start is not None else temperature
        self.t_stop = t_stop if t_stop is not None else temperature
        self.t_schedule = t_schedule
        self.temp_target = self.t_start

        # 压力渐变设置
        if np.isscalar(pressure):
            p_init = float(pressure)
        else:
            p_init = float(pressure[0]) if len(pressure) > 0 else 0.0
        self.p_start_gpa = p_start if p_start is not None else p_init
        self.p_stop_gpa = p_stop if p_stop is not None else self.p_start_gpa
        self.p_schedule = p_schedule

        # 体积渐变设置
        self.v_start = v_start
        self.v_stop = v_stop
        self.v_schedule = v_schedule
        self.initial_cell = None
        self.initial_volume = None
        self.volume_ramp_active = (v_stop is not None)

        # 渐变控制
        self.run_steps_total = run_steps
        self._has_ramp = (self.t_start != self.t_stop or
                          self.p_start_gpa != self.p_stop_gpa or
                          self.volume_ramp_active or
                          t_schedule is not None or p_schedule is not None or v_schedule is not None)
        if self._has_ramp and run_steps is None:
            raise ValueError("run_steps is required when using ramping (t_start!=t_stop, p_start!=p_stop, or v_stop is set)")

        # 气压计控制
        self.use_barostat = (pmode is not None)
        if self.volume_ramp_active and self.use_barostat:
            # 体积渐变和气压计默认互斥，但用户可以手动覆盖
            self.use_barostat = False

        self._setup_pressure(pressure, tau_p, elastic_modulus)
        self._setup_deform(deform)

    def _setup_pressure(self, pressure, tau_p, elastic_modulus):
        """
        Convert GPa input to eV/Å³ internal units.
        Convert tau_p (timestep) + elastic_modulus (GPa) to p_coupling.
        与 GPUMD 一致: p_coupling = PRESSURE_UNIT_CONVERSION / (tau_p * 3 * C)
        """
        gpa = units.GPa  # eV/Å³ per GPa
        p_coup = _tau_p_to_coupling(tau_p, elastic_modulus)

        self.target_pressure = np.zeros(6)
        self.p_coupling = np.zeros(6)

        if self.pmode is None:
            # NVT 模式：无气压计
            self.num_p_components = 0
            return
        elif self.pmode == 'iso':
            self.num_p_components = 1
            p = float(pressure) if np.isscalar(pressure) else float(pressure[0])
            self.target_pressure[:3] = p * gpa
            self.p_coupling[:3] = p_coup
        elif self.pmode == 'ortho':
            self.num_p_components = 3
            if np.isscalar(pressure):
                self.target_pressure[:3] = float(pressure) * gpa
            else:
                self.target_pressure[:3] = np.array(pressure[:3]) * gpa
            self.p_coupling[:3] = p_coup
        elif self.pmode == 'tri':
            self.num_p_components = 6
            if np.isscalar(pressure):
                self.target_pressure[:] = float(pressure) * gpa
            else:
                self.target_pressure[:] = np.array(pressure) * gpa
            self.p_coupling[:] = p_coup
        else:
            raise ValueError(f"Unknown pmode: {self.pmode}")

    def _setup_deform(self, deform):
        if deform is None:
            self.deform = [False, False, False]
            self.deform_rate = [0.0, 0.0, 0.0]
        else:
            self.deform = [d != 0 for d in deform]
            self.deform_rate = list(deform)

    def _get_stress_voigt(self):
        """
        Get pressure tensor in Voigt notation [xx, yy, zz, yz, xz, xy].
        Returns positive pressure (compression positive), in eV/Å³.
        ASE get_stress returns -sigma/V * V = -sigma, so negate it.
        """
        return -self.atoms.get_stress(voigt=True, include_ideal_gas=True)

    def _update_targets(self):
        """根据当前进度更新温度、压力目标值"""
        if not self._has_ramp:
            return

        progress = min(self.nsteps / self.run_steps_total, 1.0)

        # 更新温度目标
        if self.t_schedule is not None:
            self.temp_target = self.t_schedule(progress)
        else:
            self.temp_target = self.t_start + (self.t_stop - self.t_start) * progress

        # 更新压力目标
        if self.use_barostat:
            if self.p_schedule is not None:
                p_gpa = self.p_schedule(progress)
            else:
                p_gpa = self.p_start_gpa + (self.p_stop_gpa - self.p_start_gpa) * progress

            # 更新内部 target_pressure 数组 (eV/Å³)
            gpa = units.GPa
            if self.pmode == 'iso':
                self.target_pressure[:3] = p_gpa * gpa
            elif self.pmode == 'ortho':
                self.target_pressure[:3] = p_gpa * gpa
            elif self.pmode == 'tri':
                self.target_pressure[:] = p_gpa * gpa

    def _apply_thermostat(self):
        """
        BDP thermostat: stochastic velocity rescaling.
        Matches GPUMD compute2() thermostat section.
        """
        velocities = self.atoms.get_velocities()
        masses = self.atoms.get_masses()

        # Kinetic energy: ek = 0.5 * sum(m * v^2)
        ke = 0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2)
        ndeg = 3 * self.natoms
        sigma = ndeg * K_B * self.temp_target * 0.5

        ke_new = resamplekin(ke, sigma, ndeg, self.temp_coupling, self.rng)
        if ke > 0:
            factor = np.sqrt(ke_new / ke)
            self.atoms.set_velocities(velocities * factor)

    def _apply_barostat(self):
        """Apply SCR barostat matching GPUMD compute2()."""
        stress = self._get_stress_voigt()  # [xx, yy, zz, yz, xz, xy] in eV/Å³
        volume = self.atoms.get_volume()
        cell = self.atoms.get_cell().array.copy()

        if self.num_p_components == 1:
            # Isotropic
            p_diag = stress[:3]
            sf = cpu_pressure_isotropic(
                self.rng, self.target_pressure, self.p_coupling,
                p_diag, volume, self.temp_target)
            # Scale all cell vectors uniformly
            new_cell = cell * sf
            self.atoms.set_cell(new_cell, scale_atoms=True)

        elif self.num_p_components == 3:
            # Orthogonal
            p_diag = stress[:3]
            cell_diag = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
            sf = cpu_pressure_orthogonal(
                self.rng, self.deform, self.deform_rate, cell_diag,
                self.temp_target, self.target_pressure, self.p_coupling,
                p_diag, volume)
            # Scale cell diagonal and positions independently
            for i in range(3):
                cell[i, i] *= sf[i]
            positions = self.atoms.get_positions()
            for i in range(3):
                positions[:, i] *= sf[i]
            self.atoms.set_cell(cell, scale_atoms=False)
            self.atoms.set_positions(positions)

        else:
            # Triclinic
            mu = cpu_pressure_triclinic(
                self.rng, self.temp_target, self.target_pressure,
                self.p_coupling, stress, volume)
            # h_new = mu @ h_old, r_new = mu @ r_old
            new_cell = mu @ cell
            positions = self.atoms.get_positions()
            new_positions = (mu @ positions.T).T
            self.atoms.set_cell(new_cell, scale_atoms=False)
            self.atoms.set_positions(new_positions)

    def _apply_volume_ramp(self):
        """直接缩放晶胞以匹配目标相对体积"""
        if not self.volume_ramp_active:
            return

        if self.initial_cell is None:
            self.initial_cell = self.atoms.get_cell().array.copy()
            self.initial_volume = self.atoms.get_volume()

        progress = min(self.nsteps / self.run_steps_total, 1.0)

        if self.v_schedule is not None:
            target_rel_vol = self.v_schedule(progress)
        else:
            target_rel_vol = self.v_start + (self.v_stop - self.v_start) * progress

        # 各向同性缩放：scale_factor^3 = target_rel_vol / current_rel_vol
        current_volume = self.atoms.get_volume()
        current_rel_vol = current_volume / self.initial_volume

        if current_rel_vol > 0:
            scale = (target_rel_vol / current_rel_vol) ** (1.0 / 3.0)
            cell = self.atoms.get_cell().array
            self.atoms.set_cell(cell * scale, scale_atoms=True)

    def step(self):
        """
        One MD step: velocity Verlet + BDP thermostat + SCR barostat + volume ramp.
        Matches GPUMD compute1() + compute2() sequence.
        """
        # 更新渐变目标值
        self._update_targets()

        atoms = self.atoms
        masses = atoms.get_masses()[:, np.newaxis]

        # compute1: first half velocity Verlet
        velocities = atoms.get_velocities()
        forces = atoms.get_forces()
        velocities += 0.5 * self.dt * forces / masses
        atoms.set_velocities(velocities)
        positions = atoms.get_positions()
        positions += self.dt * velocities
        atoms.set_positions(positions)

        # compute2: second half velocity Verlet
        forces = atoms.get_forces()
        velocities = atoms.get_velocities()
        velocities += 0.5 * self.dt * forces / masses

        # 移除质心速度
        total_mass = masses.sum()
        vcm = (masses * velocities).sum(axis=0) / total_mass
        velocities -= vcm

        atoms.set_velocities(velocities)

        # compute2: thermostat then barostat
        self._apply_thermostat()
        if self.use_barostat:
            self._apply_barostat()

        # 体积渐变（在气压计之后）
        self._apply_volume_ramp()

    def run(self, steps):
        for _ in range(steps):
            self.step()
            self.nsteps += 1
            self.call_observers()

    def get_temperature(self):
        return self.atoms.get_temperature()

    def get_pressure(self):
        """Instantaneous pressure in GPa."""
        s = self._get_stress_voigt()
        return (s[0] + s[1] + s[2]) / 3.0 / units.GPa

    def get_volume(self):
        return self.atoms.get_volume()


# ============================================================
# Hugoniot helper
# ============================================================

def _compute_hugoniot(atoms, e0, p0, v0, pmode_hugo):
    """
    Compute Hugoniot deviation in temperature units (K).
    dhugo = [0.5*(P0+P)*(V0-V) + E0 - E] / (3N * kB)
    Positive => system needs to heat up.
    """
    stress = -atoms.get_stress(voigt=False)  # real stress tensor
    if pmode_hugo == 'iso':
        p_current = stress.trace() / 3.0
    elif pmode_hugo == 'x':
        p_current = stress[0, 0]
    elif pmode_hugo == 'y':
        p_current = stress[1, 1]
    elif pmode_hugo == 'z':
        p_current = stress[2, 2]
    else:
        p_current = stress.trace() / 3.0

    v_current = atoms.get_volume()
    e_current = atoms.get_total_energy()
    tdof = 3 * len(atoms)

    dhugo = (0.5 * (p0 + p_current) * (v0 - v_current)
             + e0 - e_current)
    return dhugo / (tdof * K_B)


# ============================================================
# NPT_SCR_Hugo: SCR barostat + BDP thermostat + Hugoniot
# ============================================================

class NPT_SCR_Hugo(NPT_SCR):
    """
    NPT_SCR with Hugoniot constraint.
    Target temperature is dynamically updated from the Hugoniot relation.

    Parameters
    ----------
    atoms : Atoms
    timestep : float, ASE 时间单位 (fs * units.fs)
    pressure : float, 目标压力 (GPa)
    e0 : float, Hugoniot 参考能量 (eV)，默认取当前总能量
    p0 : float, Hugoniot 参考压力 (GPa)，默认取当前压力
    v0 : float, Hugoniot 参考体积 (Å³)，默认取当前体积
    tau_t : float, BDP 恒温器弛豫时间，单位 timestep（默认 100，与 GPUMD 一致）
    tau_p : float, SCR 气压计弛豫时间，单位 timestep（默认 2000，与 GPUMD 一致）
    elastic_modulus : float, 体积模量 (GPa)，默认 15（含能材料典型值，一般不需要改）
    pmode : str, 压力模式: 'iso'(各向同性) / 'x'/'y'/'z'(单轴)
    """

    def __init__(self, atoms, timestep, pressure,
                 e0=None, p0=None, v0=None,
                 tau_t=100.0,                              # 单位 timestep
                 tau_p=2000.0,                             # 单位 timestep
                 elastic_modulus=DEFAULT_BULK_MODULUS,      # GPa
                 pmode='iso', **kwargs):
        self._pmode_hugo = pmode.lower()

        self.v0 = v0 if v0 is not None else atoms.get_volume()
        self.e0 = e0 if e0 is not None else atoms.get_total_energy()
        if p0 is None:
            self.p0 = -atoms.get_stress(voigt=False).trace() / 3.0
        else:
            self.p0 = p0 * units.GPa
        self.tdof = 3 * len(atoms)
        self.dhugo = 0.0

        t_init = max(300.0, atoms.get_temperature()
                     + _compute_hugoniot(atoms, self.e0, self.p0, self.v0,
                                         self._pmode_hugo))

        # For pmode x/y/z, barostat still uses iso mode
        baro_pmode = 'iso' if pmode in ('iso', 'x', 'y', 'z') else pmode
        super().__init__(
            atoms=atoms, timestep=timestep, temperature=t_init,
            pressure=pressure, tau_t=tau_t, tau_p=tau_p,
            elastic_modulus=elastic_modulus, pmode=baro_pmode,
            **kwargs)

        print(f"NPT_SCR_Hugo: e0={self.e0:.4f} eV, v0={self.v0:.2f} A^3, "
              f"p0={self.p0 / units.GPa:.4f} GPa, pmode={self._pmode_hugo}, "
              f"tau_t={tau_t}, tau_p={tau_p}, C={elastic_modulus} GPa")

    def _apply_thermostat(self):
        """Update target temp from Hugoniot, then apply BDP."""
        self.dhugo = _compute_hugoniot(
            self.atoms, self.e0, self.p0, self.v0, self._pmode_hugo)
        self.temp_target = max(300.0, self.atoms.get_temperature() + self.dhugo)
        super()._apply_thermostat()

    def get_hugoniot_deviation(self):
        return self.dhugo


# ============================================================
# NPH_SCR: SCR barostat only (no thermostat)
# ============================================================

class NPH_SCR(NPT_SCR):
    """NPH using SCR barostat. Temperature evolves freely."""

    def __init__(self, atoms, timestep, pressure,
                 tau_p=2000.0, elastic_modulus=DEFAULT_BULK_MODULUS,
                 pmode='iso', deform=None, seed=None, **kwargs):
        super().__init__(
            atoms=atoms, timestep=timestep, temperature=300.0,
            pressure=pressure, tau_t=100.0, tau_p=tau_p,
            elastic_modulus=elastic_modulus, pmode=pmode,
            deform=deform, seed=seed, **kwargs)

    def _apply_thermostat(self):
        """No thermostat in NPH."""
        pass


# ============================================================
# NPH_SCR_Hugo: SCR barostat + Hugoniot velocity correction
# ============================================================

class NPH_SCR_Hugo(NPH_SCR):
    """
    NPH with Hugoniot constraint using SCR barostat.
    Monitors Hugoniot deviation and applies gentle velocity
    scaling to stay on the Hugoniot curve.
    """

    def __init__(self, atoms, timestep, pressure,
                 e0=None, p0=None, v0=None,
                 tau_p=2000.0,                             # 单位 timestep
                 elastic_modulus=DEFAULT_BULK_MODULUS,      # GPa
                 pmode='iso',
                 hugoniot_correction=True, **kwargs):
        self._pmode_hugo = pmode.lower()
        self.hugoniot_correction = hugoniot_correction

        self.v0 = v0 if v0 is not None else atoms.get_volume()
        self.e0 = e0 if e0 is not None else atoms.get_total_energy()
        if p0 is None:
            self.p0 = -atoms.get_stress(voigt=False).trace() / 3.0
        else:
            self.p0 = p0 * units.GPa
        self.tdof = 3 * len(atoms)
        self.dhugo = 0.0

        baro_pmode = 'iso' if pmode in ('iso', 'x', 'y', 'z') else pmode
        super().__init__(
            atoms=atoms, timestep=timestep, pressure=pressure,
            tau_p=tau_p, elastic_modulus=elastic_modulus,
            pmode=baro_pmode, **kwargs)

        print(f"NPH_SCR_Hugo: e0={self.e0:.4f} eV, v0={self.v0:.2f} A^3, "
              f"p0={self.p0 / units.GPa:.4f} GPa, pmode={self._pmode_hugo}")

    def step(self):
        """NPH step + Hugoniot monitoring and correction."""
        super().step()

        self.dhugo = _compute_hugoniot(
            self.atoms, self.e0, self.p0, self.v0, self._pmode_hugo)

        if self.hugoniot_correction and abs(self.dhugo) > 1.0:
            ke = self.atoms.get_kinetic_energy()
            ke_target = ke + self.dhugo * self.tdof * K_B * 0.5
            if ke > 0 and ke_target > 0:
                scale = np.sqrt(ke_target / ke)
                scale = np.clip(scale, 0.99, 1.01)
                self.atoms.set_velocities(self.atoms.get_velocities() * scale)

    def get_hugoniot_deviation(self):
        return self.dhugo


# ============================================================
# Convenience factory
# ============================================================

def create_npt_scr(atoms, timestep=1.0, temperature=300.0, pressure=0.0,
                   tau_t=100.0, tau_p=2000.0,
                   elastic_modulus=DEFAULT_BULK_MODULUS,
                   pmode='iso',
                   run_steps=None,
                   t_start=None, t_stop=None, t_schedule=None,
                   p_start=None, p_stop=None, p_schedule=None,
                   v_start=1.0, v_stop=None, v_schedule=None,
                   **kwargs):
    """
    Create NPT_SCR dynamics with optional ramping support.

    Parameters
    ----------
    timestep : float, 时间步长 (fs)
    temperature : float, 目标温度 (K)
    pressure : float, 目标压力 (GPa)
    tau_t : float, BDP 恒温器弛豫时间，单位 timestep（默认 100，与 GPUMD 一致）
    tau_p : float, SCR 气压计弛豫时间，单位 timestep（默认 2000，与 GPUMD 一致）
    elastic_modulus : float, 体积模量 (GPa)，默认 15（含能材料典型值）
    pmode : str, 压力模式 iso/ortho/tri，None 表示 NVT 模式
    run_steps : int, 总步数（渐变时必需）
    t_start, t_stop : float, 温度渐变起止值 (K)
    t_schedule : callable, 自定义温度函数 f(progress) -> T(K)
    p_start, p_stop : float, 压力渐变起止值 (GPa)
    p_schedule : callable, 自定义压力函数 f(progress) -> P(GPa)
    v_start, v_stop : float, 相对体积渐变起止值（1.0 = 初始体积）
    v_schedule : callable, 自定义体积函数 f(progress) -> relative_volume
    """
    return NPT_SCR(
        atoms=atoms,
        timestep=timestep * units.fs,
        temperature=temperature,
        pressure=pressure,
        tau_t=tau_t,
        tau_p=tau_p,
        elastic_modulus=elastic_modulus,
        pmode=pmode,
        run_steps=run_steps,
        t_start=t_start, t_stop=t_stop, t_schedule=t_schedule,
        p_start=p_start, p_stop=p_stop, p_schedule=p_schedule,
        v_start=v_start, v_stop=v_stop, v_schedule=v_schedule,
        **kwargs)

