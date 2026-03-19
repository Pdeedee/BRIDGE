"""
NPT_SCR 渐变扩展：支持温度、压力和体积的渐变控制

包含两个子类：
- NPT_SCR_Ramp: 温度和压力渐变（支持各向异性）
- NPT_SCR_VolumeRamp: 体积渐变
"""

import numpy as np
from ase import units
from nepactive.npt_scr import NPT_SCR


class NPT_SCR_Ramp(NPT_SCR):
    """
    NPT_SCR 扩展：支持温度和压力的线性或自定义渐变

    支持各向异性压力渐变：
    - iso 模式：标量渐变
    - ortho 模式：[Pxx, Pyy, Pzz] 独立渐变
    - tri 模式：[Pxx, Pyy, Pzz, Pyz, Pxz, Pxy] 独立渐变
    """

    def __init__(self, atoms, timestep, temperature, pressure,
                 tau_t=100.0, tau_p=2000.0,
                 elastic_modulus=15.0,
                 pmode='iso', deform=None, seed=None,
                 # --- 渐变参数 ---
                 run_steps=None,           # 总步数（必需）
                 t_start=None,             # 起始温度 (K)
                 t_stop=None,              # 终止温度 (K)
                 t_schedule=None,          # 自定义温度函数 f(progress) -> T(K)
                 p_start=None,             # 起始压力 (GPa)，标量或数组
                 p_stop=None,              # 终止压力 (GPa)，标量或数组
                 p_schedule=None,          # 自定义压力函数 f(progress) -> P(GPa) 或 [Pxx,Pyy,Pzz]
                 **kwargs):
        """
        参数
        ----
        run_steps : int
            总步数，用于计算渐变进度（必需）
        t_start, t_stop : float
            温度渐变起止值 (K)，默认为 temperature
        t_schedule : callable
            自定义温度函数 f(progress) -> T(K)，覆盖线性渐变
        p_start, p_stop : float or array-like
            压力渐变起止值 (GPa)
            - iso 模式：标量
            - ortho 模式：[Pxx, Pyy, Pzz]
            - tri 模式：[Pxx, Pyy, Pzz, Pyz, Pxz, Pxy]
        p_schedule : callable
            自定义压力函数 f(progress) -> P 或 [Pxx, Pyy, Pzz, ...]

        示例
        ----
        # 各向同性压力渐变
        dyn = NPT_SCR_Ramp(atoms, 0.2*units.fs, 300, 0,
                           run_steps=10000,
                           t_start=300, t_stop=1000,
                           p_start=0, p_stop=10,
                           pmode='iso')

        # 各向异性压力渐变（z 方向和 xy 平面不同）
        dyn = NPT_SCR_Ramp(atoms, 0.2*units.fs, 300, [0, 0, 0],
                           run_steps=10000,
                           p_start=[0, 0, 0],
                           p_stop=[0.5, 0.5, 1.0],  # xy=0.5, z=1.0
                           pmode='ortho')

        # 自定义温度曲线
        def cosine_heat(progress):
            return 300 + 700 * (1 - np.cos(np.pi * progress)) / 2

        dyn = NPT_SCR_Ramp(atoms, 0.2*units.fs, 300, 0,
                           run_steps=10000,
                           t_schedule=cosine_heat,
                           pmode='iso')
        """
        if run_steps is None:
            raise ValueError("run_steps is required for NPT_SCR_Ramp")

        # 初始化父类
        super().__init__(atoms, timestep, temperature, pressure,
                         tau_t, tau_p, elastic_modulus,
                         pmode, deform, seed, **kwargs)

        self.run_steps_total = run_steps

        # 温度渐变设置
        self.t_start = t_start if t_start is not None else temperature
        self.t_stop = t_stop if t_stop is not None else temperature
        self.t_schedule = t_schedule
        self.temp_target = self.t_start

        # 压力渐变设置
        self._setup_pressure_ramp(pressure, p_start, p_stop, p_schedule)

    def _setup_pressure_ramp(self, pressure, p_start, p_stop, p_schedule):
        """设置压力渐变参数"""
        gpa = units.GPa

        # 解析初始压力
        if self.pmode == 'iso':
            # 标量模式
            p_init = float(pressure) if np.isscalar(pressure) else float(pressure[0])
            self.p_start_gpa = p_start if p_start is not None else p_init
            self.p_stop_gpa = p_stop if p_stop is not None else self.p_start_gpa

            # 转换为数组形式（内部统一处理）
            if np.isscalar(self.p_start_gpa):
                self.p_start_array = np.array([self.p_start_gpa] * 3)
            else:
                self.p_start_array = np.array(self.p_start_gpa[:3])

            if np.isscalar(self.p_stop_gpa):
                self.p_stop_array = np.array([self.p_stop_gpa] * 3)
            else:
                self.p_stop_array = np.array(self.p_stop_gpa[:3])

        elif self.pmode == 'ortho':
            # 正交模式：3 个独立分量
            if np.isscalar(pressure):
                p_init = np.array([float(pressure)] * 3)
            else:
                p_init = np.array(pressure[:3])

            if p_start is None:
                self.p_start_array = p_init
            elif np.isscalar(p_start):
                self.p_start_array = np.array([float(p_start)] * 3)
            else:
                self.p_start_array = np.array(p_start[:3])

            if p_stop is None:
                self.p_stop_array = self.p_start_array.copy()
            elif np.isscalar(p_stop):
                self.p_stop_array = np.array([float(p_stop)] * 3)
            else:
                self.p_stop_array = np.array(p_stop[:3])

        elif self.pmode == 'tri':
            # 三斜模式：6 个独立分量
            if np.isscalar(pressure):
                p_init = np.array([float(pressure)] * 6)
            else:
                p_init = np.array(pressure[:6])

            if p_start is None:
                self.p_start_array = p_init
            elif np.isscalar(p_start):
                self.p_start_array = np.array([float(p_start)] * 6)
            else:
                self.p_start_array = np.array(p_start[:6])

            if p_stop is None:
                self.p_stop_array = self.p_start_array.copy()
            elif np.isscalar(p_stop):
                self.p_stop_array = np.array([float(p_stop)] * 6)
            else:
                self.p_stop_array = np.array(p_stop[:6])
        else:
            # NVT 模式
            self.p_start_array = np.zeros(6)
            self.p_stop_array = np.zeros(6)

        self.p_schedule = p_schedule

    def _update_targets(self):
        """根据当前进度更新温度和压力目标值"""
        progress = min(self.nsteps / self.run_steps_total, 1.0)

        # 更新温度目标
        if self.t_schedule is not None:
            self.temp_target = self.t_schedule(progress)
        else:
            self.temp_target = self.t_start + (self.t_stop - self.t_start) * progress

        # 更新压力目标
        if self.pmode is not None:
            gpa = units.GPa

            if self.p_schedule is not None:
                # 自定义压力函数
                p_result = self.p_schedule(progress)
                if np.isscalar(p_result):
                    p_gpa_array = np.array([float(p_result)] * len(self.p_start_array))
                else:
                    p_gpa_array = np.array(p_result)
            else:
                # 线性插值
                p_gpa_array = self.p_start_array + (self.p_stop_array - self.p_start_array) * progress

            # 更新内部 target_pressure 数组 (eV/Å³)
            if self.pmode == 'iso':
                self.target_pressure[:3] = p_gpa_array[0] * gpa
            elif self.pmode == 'ortho':
                self.target_pressure[:3] = p_gpa_array[:3] * gpa
            elif self.pmode == 'tri':
                self.target_pressure[:] = p_gpa_array[:6] * gpa

    def step(self):
        """重写 step，在开始时更新渐变目标"""
        self._update_targets()
        super().step()


class NPT_SCR_VolumeRamp(NPT_SCR):
    """
    NPT_SCR 扩展：支持体积渐变

    直接缩放晶胞以匹配目标相对体积，与气压计互斥
    """

    def __init__(self, atoms, timestep, temperature, pressure=0,
                 tau_t=100.0, tau_p=2000.0,
                 elastic_modulus=15.0,
                 pmode=None,  # 默认 None（NVT），体积渐变时通常不用气压计
                 deform=None, seed=None,
                 # --- 体积渐变参数 ---
                 run_steps=None,           # 总步数（必需）
                 v_start=1.0,              # 起始相对体积（1.0 = 初始体积）
                 v_stop=None,              # 终止相对体积（必需）
                 v_schedule=None,          # 自定义体积函数 f(progress) -> relative_volume
                 **kwargs):
        """
        参数
        ----
        run_steps : int
            总步数（必需）
        v_start : float
            起始相对体积，1.0 表示初始体积
        v_stop : float
            终止相对体积，例如 0.9 表示压缩到 90% 体积（必需）
        v_schedule : callable
            自定义体积函数 f(progress) -> relative_volume
        pmode : str or None
            压力模式，默认 None（NVT）。如果设置了 pmode，气压计和体积渐变会同时作用

        示例
        ----
        # 体积压缩到 95%
        dyn = NPT_SCR_VolumeRamp(atoms, 0.2*units.fs, 300,
                                 run_steps=10000,
                                 v_start=1.0, v_stop=0.95)

        # 自定义体积曲线
        def smooth_compress(progress):
            # 先快后慢的压缩
            return 1.0 - 0.1 * progress**2

        dyn = NPT_SCR_VolumeRamp(atoms, 0.2*units.fs, 300,
                                 run_steps=10000,
                                 v_schedule=smooth_compress)
        """
        if run_steps is None:
            raise ValueError("run_steps is required for NPT_SCR_VolumeRamp")
        if v_stop is None and v_schedule is None:
            raise ValueError("Either v_stop or v_schedule must be provided")

        # 初始化父类
        super().__init__(atoms, timestep, temperature, pressure,
                         tau_t, tau_p, elastic_modulus,
                         pmode, deform, seed, **kwargs)

        self.run_steps_total = run_steps
        self.v_start = v_start
        self.v_stop = v_stop if v_stop is not None else v_start
        self.v_schedule = v_schedule

        self.initial_cell = None
        self.initial_volume = None

    def _apply_volume_ramp(self):
        """直接缩放晶胞以匹配目标相对体积"""
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
        """重写 step，在气压计之后应用体积渐变"""
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
        if self.pmode is not None:
            self._apply_barostat()

        # 体积渐变（在气压计之后）
        self._apply_volume_ramp()
