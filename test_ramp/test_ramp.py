#!/usr/bin/env python3
"""测试 NPT_SCR 渐变功能"""

import sys
sys.path.insert(0, '/workplace/liuzf/code/nepactive/src')

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from nepactive.npt_scr import NPT_SCR
from mattersim.forcefield import MatterSimCalculator

# 读取结构
poscar_path = "/workplace/liuzf/code/example/nepactivemain/05.HMX/POSCAR"
atoms = read(poscar_path)

# 设置 calculator
calc = MatterSimCalculator(device='cuda')
atoms.calc = calc

# 初始化速度
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)
ZeroRotation(atoms)

print("=" * 60)
print("测试 1: NPT 线性温度渐变 (300K -> 1000K, 1000步)")
print("=" * 60)

dyn1 = NPT_SCR(
    atoms,
    timestep=0.2*units.fs,
    temperature=300,
    pressure=0,
    run_steps=1000,
    t_start=300,
    t_stop=1000,
    pmode='iso',
    tau_t=100,
    tau_p=2000
)

# 记录温度和压力
temps = []
pressures = []
volumes = []

def record_state():
    temps.append(atoms.get_temperature())
    stress = -atoms.get_stress(voigt=False)
    p = stress.trace() / 3.0 / units.GPa
    pressures.append(p)
    volumes.append(atoms.get_volume())

dyn1.attach(record_state, interval=50)

try:
    dyn1.run(1000)
    print(f"✓ 测试 1 完成")
    print(f"  起始温度: {temps[0]:.1f} K")
    print(f"  终止温度: {temps[-1]:.1f} K")
    print(f"  平均压力: {np.mean(pressures):.2f} GPa")
    print(f"  体积变化: {volumes[0]:.1f} -> {volumes[-1]:.1f} Å³")
except Exception as e:
    print(f"✗ 测试 1 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试 2: NVT 模式 (pmode=None, 300K -> 800K, 1000步)")
print("=" * 60)

# 重新初始化
atoms = read(poscar_path)
atoms.calc = calc
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)
ZeroRotation(atoms)

dyn2 = NPT_SCR(
    atoms,
    timestep=0.2*units.fs,
    temperature=300,
    pressure=0,
    run_steps=1000,
    t_start=300,
    t_stop=800,
    pmode=None,  # NVT 模式
    tau_t=100
)

temps2 = []
volumes2 = []

def record_state2():
    temps2.append(atoms.get_temperature())
    volumes2.append(atoms.get_volume())

dyn2.attach(record_state2, interval=50)

try:
    dyn2.run(1000)
    print(f"✓ 测试 2 完成")
    print(f"  起始温度: {temps2[0]:.1f} K")
    print(f"  终止温度: {temps2[-1]:.1f} K")
    print(f"  体积变化: {volumes2[0]:.1f} -> {volumes2[-1]:.1f} Å³ (NVT应该变化不大)")
except Exception as e:
    print(f"✗ 测试 2 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试 3: 体积渐变 (1.0 -> 0.95, 1000步)")
print("=" * 60)

# 重新初始化
atoms = read(poscar_path)
atoms.calc = calc
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)
ZeroRotation(atoms)

initial_volume = atoms.get_volume()

dyn3 = NPT_SCR(
    atoms,
    timestep=0.2*units.fs,
    temperature=300,
    pressure=0,
    run_steps=1000,
    v_start=1.0,
    v_stop=0.95,
    pmode=None,  # 体积渐变时禁用气压计
    tau_t=100
)

temps3 = []
volumes3 = []

def record_state3():
    temps3.append(atoms.get_temperature())
    volumes3.append(atoms.get_volume())

dyn3.attach(record_state3, interval=50)

try:
    dyn3.run(1000)
    print(f"✓ 测试 3 完成")
    print(f"  初始体积: {initial_volume:.1f} Å³")
    print(f"  起始体积: {volumes3[0]:.1f} Å³")
    print(f"  终止体积: {volumes3[-1]:.1f} Å³")
    print(f"  目标体积: {initial_volume * 0.95:.1f} Å³")
    print(f"  相对误差: {abs(volumes3[-1] - initial_volume * 0.95) / (initial_volume * 0.95) * 100:.2f}%")
except Exception as e:
    print(f"✗ 测试 3 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试 4: 自定义温度函数 (余弦升温, 1000步)")
print("=" * 60)

# 重新初始化
atoms = read(poscar_path)
atoms.calc = calc
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)
ZeroRotation(atoms)

def cosine_heat(progress):
    """余弦升温：300K -> 1000K"""
    return 300 + 700 * (1 - np.cos(np.pi * progress)) / 2

dyn4 = NPT_SCR(
    atoms,
    timestep=0.2*units.fs,
    temperature=300,
    pressure=0,
    run_steps=1000,
    t_schedule=cosine_heat,
    pmode=None,
    tau_t=100
)

temps4 = []
target_temps4 = []

def record_state4():
    temps4.append(atoms.get_temperature())
    progress = len(temps4) * 50 / 1000
    target_temps4.append(cosine_heat(progress))

dyn4.attach(record_state4, interval=50)

try:
    dyn4.run(1000)
    print(f"✓ 测试 4 完成")
    print(f"  起始温度: {temps4[0]:.1f} K (目标: {target_temps4[0]:.1f} K)")
    print(f"  终止温度: {temps4[-1]:.1f} K (目标: {target_temps4[-1]:.1f} K)")
    print(f"  中间温度: {temps4[len(temps4)//2]:.1f} K (目标: {target_temps4[len(temps4)//2]:.1f} K)")
except Exception as e:
    print(f"✗ 测试 4 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("所有测试完成！")
print("=" * 60)
