#!/usr/bin/env python3
"""测试各向异性压力渐变和体积渐变"""

import sys
sys.path.insert(0, '/workplace/liuzf/code/nepactive/src')

import numpy as np
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from nepactive.npt_scr_ramp import NPT_SCR_Ramp, NPT_SCR_VolumeRamp
from mattersim.forcefield import MatterSimCalculator

# 读取结构
poscar_path = "/workplace/liuzf/code/example/nepactivemain/05.HMX/POSCAR"
atoms = read(poscar_path)

# 设置 calculator
calc = MatterSimCalculator(device='cuda')
atoms.calc = calc

print("=" * 60)
print("测试 1: 各向异性压力渐变 (ortho 模式)")
print("  xy 平面: 0 -> 0.5 GPa")
print("  z 方向: 0 -> 1.0 GPa")
print("=" * 60)

MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)
ZeroRotation(atoms)

dyn1 = NPT_SCR_Ramp(
    atoms,
    timestep=0.2*units.fs,
    temperature=300,
    pressure=[0, 0, 0],
    run_steps=1000,
    p_start=[0, 0, 0],
    p_stop=[0.5, 0.5, 1.0],  # [Pxx, Pyy, Pzz]
    pmode='ortho',
    tau_t=100,
    tau_p=2000
)

pressures_x = []
pressures_y = []
pressures_z = []

def record_pressure1():
    stress = -atoms.get_stress(voigt=False)
    pressures_x.append(stress[0, 0] / units.GPa)
    pressures_y.append(stress[1, 1] / units.GPa)
    pressures_z.append(stress[2, 2] / units.GPa)

dyn1.attach(record_pressure1, interval=50)

try:
    dyn1.run(1000)
    print(f"✓ 测试 1 完成")
    print(f"  Pxx: {pressures_x[0]:.2f} -> {pressures_x[-1]:.2f} GPa (目标: 0 -> 0.5)")
    print(f"  Pyy: {pressures_y[0]:.2f} -> {pressures_y[-1]:.2f} GPa (目标: 0 -> 0.5)")
    print(f"  Pzz: {pressures_z[0]:.2f} -> {pressures_z[-1]:.2f} GPa (目标: 0 -> 1.0)")
except Exception as e:
    print(f"✗ 测试 1 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试 2: 自定义各向异性压力函数")
print("  xy 平面: 余弦升压到 0.5 GPa")
print("  z 方向: 线性升压到 1.0 GPa")
print("=" * 60)

atoms = read(poscar_path)
atoms.calc = calc
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)
ZeroRotation(atoms)

def aniso_pressure_schedule(progress):
    """自定义各向异性压力"""
    pxy = 0.5 * (1 - np.cos(np.pi * progress)) / 2  # 余弦
    pz = 1.0 * progress  # 线性
    return [pxy, pxy, pz]

dyn2 = NPT_SCR_Ramp(
    atoms,
    timestep=0.2*units.fs,
    temperature=300,
    pressure=[0, 0, 0],
    run_steps=1000,
    p_schedule=aniso_pressure_schedule,
    pmode='ortho',
    tau_t=100,
    tau_p=2000
)

pressures_x2 = []
pressures_z2 = []
target_x2 = []
target_z2 = []

def record_pressure2():
    stress = -atoms.get_stress(voigt=False)
    pressures_x2.append(stress[0, 0] / units.GPa)
    pressures_z2.append(stress[2, 2] / units.GPa)
    progress = len(pressures_x2) * 50 / 1000
    targets = aniso_pressure_schedule(progress)
    target_x2.append(targets[0])
    target_z2.append(targets[2])

dyn2.attach(record_pressure2, interval=50)

try:
    dyn2.run(1000)
    print(f"✓ 测试 2 完成")
    print(f"  Pxx: {pressures_x2[-1]:.2f} GPa (目标: {target_x2[-1]:.2f})")
    print(f"  Pzz: {pressures_z2[-1]:.2f} GPa (目标: {target_z2[-1]:.2f})")
except Exception as e:
    print(f"✗ 测试 2 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试 3: NPT_SCR_VolumeRamp 体积渐变")
print("  体积: 1.0 -> 0.92")
print("=" * 60)

atoms = read(poscar_path)
atoms.calc = calc
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)
ZeroRotation(atoms)

initial_volume = atoms.get_volume()

dyn3 = NPT_SCR_VolumeRamp(
    atoms,
    timestep=0.2*units.fs,
    temperature=300,
    run_steps=1000,
    v_start=1.0,
    v_stop=0.92,
    tau_t=100
)

volumes3 = []

def record_volume3():
    volumes3.append(atoms.get_volume())

dyn3.attach(record_volume3, interval=50)

try:
    dyn3.run(1000)
    print(f"✓ 测试 3 完成")
    print(f"  初始体积: {initial_volume:.1f} Å³")
    print(f"  终止体积: {volumes3[-1]:.1f} Å³")
    print(f"  目标体积: {initial_volume * 0.92:.1f} Å³")
    print(f"  相对误差: {abs(volumes3[-1] - initial_volume * 0.92) / (initial_volume * 0.92) * 100:.2f}%")
except Exception as e:
    print(f"✗ 测试 3 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("所有测试完成！")
print("=" * 60)
