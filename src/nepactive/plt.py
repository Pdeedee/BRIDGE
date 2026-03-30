#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Times New Roman'
# plt.rcParams['font.size'] = 18
# plt.rcParams.update({'axes.linewidth': 2, 'axes.edgecolor': 'black'})
# # 设置全局字体加粗
# plt.rcParams.update({
#     'font.weight': 'bold',  # 全局字体加粗
#     'axes.labelweight': 'bold',  # 坐标轴标签加粗
#     'axes.titleweight': 'bold',  # 标题加粗
#     'axes.linewidth': 2,  # 设置坐标轴边框线宽
#     'axes.edgecolor': 'black',  # 设置坐标轴边框颜色
#     'mathtext.default': 'regular',  # 关键设置
#     'mathtext.rm': 'Times New Roman',  # 设置数学文本的正常字体
#     'mathtext.it': 'Times New Roman:italic',  # 设置数学文本的斜体字体
#     'mathtext.bf': 'Times New Roman:bold'  #
# })
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']   # 图表字体 Arial
plt.rcParams['font.size'] = 18

plt.rcParams.update({
    # 'font.weight': 'bold',
    # 'axes.labelweight': 'bold',
    # 'axes.titleweight': 'bold',
    'font.weight': 'normal',
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
    'axes.linewidth': 1,
    'axes.edgecolor': 'black',
    'mathtext.default': 'regular',
    'mathtext.rm': 'Arial',                 # 数学公式用 Arial
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold'
})
# from cycler import cycler
# default_cycler = (cycler(color=['b','r', 'g', 'y']) +
#                   cycler(linestyle=['-', '--', ':', '-.']))
# plt.rc('axes', prop_cycle=default_cycler)

def _loadtxt_2d(path, **kwargs):
    data = np.loadtxt(path, **kwargs)
    return np.atleast_2d(data)


def _resolve_axis_values(length, total_time=None):
    if total_time is None:
        return np.arange(length, dtype=float), "Step"
    return np.linspace(0, float(total_time), length, endpoint=False) / 1000.0, "Time (ps)"


def gpumdplt(total_time=None, time_step=0.2, thermo_file='thermo.out', output_file='thermo.png'):

    data = _loadtxt_2d(thermo_file)

    num_points = len(data)  # 数据点数
    time, xlabel = _resolve_axis_values(num_points, total_time=total_time)

    temperature = data[:, 0]
    kinetic_energy = data[:, 1]
    potential_energy = data[:, 2]
    pressure_x = data[:, 3]
    pressure_y = data[:, 4]
    pressure_z = data[:, 5]

    num_columns = data.shape[1]

    if num_columns == 12:
        box_length_x = data[:, 9]
        box_length_y = data[:, 10]
        box_length_z = data[:, 11]
        volume = np.abs(box_length_x * box_length_y * box_length_z)
        
    elif num_columns == 18:
        ax, ay, az = data[:, 9], data[:, 10], data[:, 11]
        bx, by, bz = data[:, 12], data[:, 13], data[:, 14]
        cx, cy, cz = data[:, 15], data[:, 16], data[:, 17]
        
        # 计算晶胞的体积（使用行列式公式）
        # 叉积 (b x c)
        bx_cy_bz = by * cz - bz * cy
        bx_cz_by = bz * cx - bx * cz
        bx_cx_by = bx * cy - by * cx
        
        # 点积 a · (b x c)
        volume = ax * bx_cy_bz + ay * bx_cz_by + az * bx_cx_by
        volume = np.abs(volume)  # 体积取绝对值
        
    else:
        raise ValueError("不支持的 thermo.out 文件列数。期望 12 或 18 列。")

    # 子图
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), dpi=100)

    # 温度
    # print(f"time = {time.shape}, temperature = {temperature.shape}")
    axs[0, 0].plot(time, temperature, color="b")
    axs[0, 0].set_title('Temperature')
    axs[0, 0].set_xlabel(xlabel)
    axs[0, 0].set_ylabel('Temperature (K)')

    # 势能与动能
    color_potential = 'tab:orange'
    color_kinetic = 'tab:green'
    axs[0, 1].set_title(r'$E_P$ vs $E_K$')
    axs[0, 1].set_xlabel(xlabel)
    axs[0, 1].set_ylabel('Potential Energy (eV)', color=color_potential)
    axs[0, 1].plot(time, potential_energy, color=color_potential)
    axs[0, 1].tick_params(axis='y', labelcolor=color_potential)

    axs_kinetic = axs[0, 1].twinx()
    axs_kinetic.set_ylabel('Kinetic Energy (eV)', color=color_kinetic)
    axs_kinetic.plot(time, kinetic_energy, color=color_kinetic)
    axs_kinetic.tick_params(axis='y', labelcolor=color_kinetic)

    # 压力
    axs[1, 0].plot(time, pressure_x, label=r'$P_{xx}$')
    axs[1, 0].plot(time, pressure_y, label=r'$P_{yx}$')
    axs[1, 0].plot(time, pressure_z, label=r'$P_{zz}$')
    axs[1, 0].set_title('Pressure')
    axs[1, 0].set_xlabel(xlabel)
    axs[1, 0].set_ylabel('Pressure (GPa)')
    axs[1, 0].set_ylim(0,max(int(1.2*np.max(pressure_z)),60))
    axs[1, 0].legend(framealpha=0)

    # 相对体积
    axs[1, 1].plot(time, volume, color="b")
    axs[1, 1].set_title('Volume')
    axs[1, 1].set_xlabel(xlabel)
    axs[1, 1].set_ylabel(r'Volume ($\AA^{3}$)')
    # axs[1, 1].legend(framealpha=0)

    plt.tight_layout()

    # 保存或显示图像
    #if len(sys.argv) > 2 and sys.argv[2] == 'save':
    plt.savefig(output_file)
    plt.close()


from pylab import *

def nep_plt(testplt=True, input_dir='.', output_dir='.', plot_loss=True):
    if testplt:
        prefix = 'test_'
        energy_file = 'energy_test.out'
        force_file = 'force_test.out'
        virial_file = 'virial_test.out'
    else:
        prefix = 'train_'
        energy_file = 'energy_train.out'
        force_file = 'force_train.out'
        virial_file = 'virial_train.out'
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    created = []
    loss_path = input_dir / 'loss.out'
    if plot_loss and loss_path.exists():
        loss = loadtxt(loss_path)
        loglog(loss[:, 1:6])
        loglog(loss[:, 7:9])
        xlabel('Generation/100')
        ylabel('Loss')
        legend(['Total', 'L1-regularization', 'L2-regularization', 'Energy-train', 'Force-train', 'Energy-test', 'Force-test'],framealpha=0)
        tight_layout()
        loss_png = output_dir / 'loss.png'
        plt.savefig(loss_png, dpi=300)
        plt.close()
        created.append(str(loss_png))

    energy_path = input_dir / energy_file
    force_path = input_dir / force_file
    virial_path = input_dir / virial_file
    if not (energy_path.exists() and force_path.exists() and virial_path.exists()):
        return created

    energy = loadtxt(energy_path)
    x_min = np.min(energy[:, :])
    x_max = np.max(energy[:, :])
    plot(energy[:, 1], energy[:, 0], '.')
    plot(linspace(x_min,x_max), linspace(x_min,x_max), '-')
    xlabel('MatterSim energy (eV/atom)')
    ylabel('NEP energy (eV/atom)')
    tight_layout()
    energy_png = output_dir / f'{prefix}energy.png'
    plt.savefig(energy_png, dpi=300)
    plt.close()
    created.append(str(energy_png))

    force = loadtxt(force_path)
    x_min = np.min(force[:, :])
    x_max = np.max(force[:, :])
    plot(force[:, 3:6], force[:, 0:3], '.')
    plot(linspace(x_min,x_max), linspace(x_min,x_max), '-',color='r')
    xlabel('MatterSim force (eV/A)')
    ylabel('NEP force (eV/A)')
    legend(['x direction', 'y direction', 'z direction'],framealpha=0)
    tight_layout()
    force_png = output_dir / f'{prefix}force.png'
    plt.savefig(force_png, dpi=300)
    plt.close()
    created.append(str(force_png))

    virial = loadtxt(virial_path)
    x_min = np.min(virial[:, :])
    x_max = np.max(virial[:, :])
    plot(virial[:, 6:11], virial[:, 0:5], '.')
    plot(linspace(x_min,x_max), linspace(x_min,x_max), '-',color='r')
    xlabel('MatterSim Virial (eV/A)')
    ylabel('NEP Virial (eV/A)')
    legend(['xx', 'yy', 'zz', 'xy', 'yz', 'zx'],framealpha=0)
    tight_layout()
    virial_png = output_dir / f'{prefix}virial.png'
    plt.savefig(virial_png, dpi=300)
    plt.close()
    created.append(str(virial_png))
    return created

def ase_plt(md_log_file='md.log', output_file='thermo.png'):
    data = _loadtxt_2d(md_log_file, skiprows=1, encoding='utf-8')
    num_points = len(data)  # 数据点数
    time = data[:, 0]

    temperature = data[:, 4]
    kinetic_energy = data[:, 3]
    potential_energy = data[:, 2]
    pressure_x = data[:, 6]
    pressure_y = data[:, 7]
    pressure_z = data[:, 8]
    volume = data[:, 5]

    # 子图
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), dpi=300)

    # 温度
    # print(f"time = {time.shape}, temperature = {temperature.shape}")
    axs[0, 0].plot(time, temperature, color="b")
    axs[0, 0].set_title('Temperature')
    axs[0, 0].set_xlabel('Time (ps)')
    axs[0, 0].set_ylabel('Temperature (K)')

    # 势能与动能
    color_potential = 'tab:orange'
    color_kinetic = 'tab:green'
    axs[0, 1].set_title(r'$P_E$ vs $K_E$')
    axs[0, 1].set_xlabel('Time (ps)')
    axs[0, 1].set_ylabel('Potential Energy (eV)', color=color_potential)
    axs[0, 1].plot(time, potential_energy, color=color_potential)
    axs[0, 1].tick_params(axis='y', labelcolor=color_potential)

    axs_kinetic = axs[0, 1].twinx()
    axs_kinetic.set_ylabel('Kinetic Energy (eV)', color=color_kinetic)
    axs_kinetic.plot(time, kinetic_energy, color=color_kinetic)
    axs_kinetic.tick_params(axis='y', labelcolor=color_kinetic)

    # 压力
    axs[1, 0].plot(time, pressure_x, label='Px')
    axs[1, 0].plot(time, pressure_y, label='Py')
    axs[1, 0].plot(time, pressure_z, label='Pz')
    axs[1, 0].set_title('Pressure')
    axs[1, 0].set_xlabel('Time (ps)')
    axs[1, 0].set_ylabel('Pressure (GPa)')
    axs[1, 0].legend(framealpha=0)

    # 相对体积
    axs[1, 1].plot(time, volume)#, label='Volume')
    axs[1, 1].set_title('Volume')
    axs[1, 1].set_xlabel('Time (ps)')
    axs[1, 1].set_ylabel('Volume')
    # axs[1, 1].legend()

    plt.tight_layout()

    # 保存或显示图像
    #if len(sys.argv) > 2 and sys.argv[2] == 'save':
    plt.savefig(output_file)
    plt.close()


def _parse_run_in_total_time(run_in_path):
    if not Path(run_in_path).is_file():
        return None
    time_step = None
    run_steps = 0
    with open(run_in_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = re.split(r'\s+', line)
            key = parts[0].lower()
            if key == 'time_step' and len(parts) >= 2:
                try:
                    time_step = float(parts[1])
                except ValueError:
                    pass
            elif key == 'run' and len(parts) >= 2:
                try:
                    run_steps += int(float(parts[1]))
                except ValueError:
                    pass
    if time_step is None or run_steps <= 0:
        return None
    return run_steps * time_step


def plot_auto(workdir='.', output_dir=None, verbose=True):
    workdir = Path(workdir).resolve()
    output_dir = workdir if output_dir is None else Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    missing = []

    md_log = workdir / 'md.log'
    thermo_out = workdir / 'thermo.out'
    loss_out = workdir / 'loss.out'

    if md_log.exists():
        md_output = output_dir / ('ase_thermo.png' if thermo_out.exists() else 'thermo.png')
        ase_plt(md_log_file=str(md_log), output_file=str(md_output))
        created.append(str(md_output))
    else:
        missing.append('md.log')

    if thermo_out.exists():
        total_time = _parse_run_in_total_time(workdir / 'run.in')
        gpumd_output = output_dir / ('gpumd_thermo.png' if md_log.exists() else 'thermo.png')
        gpumdplt(total_time=total_time, thermo_file=str(thermo_out), output_file=str(gpumd_output))
        created.append(str(gpumd_output))
    else:
        missing.append('thermo.out')

    if loss_out.exists():
        current_dir = os.getcwd()
        try:
            os.chdir(workdir)
            created.extend(nep_plt(testplt=True, input_dir=workdir, output_dir=output_dir, plot_loss=True))
            created.extend(nep_plt(testplt=False, input_dir=workdir, output_dir=output_dir, plot_loss=False))
        finally:
            os.chdir(current_dir)
    else:
        missing.append('loss.out')

    created = list(dict.fromkeys(created))
    if verbose:
        print(f'Workdir: {workdir}')
        for path in created:
            print(f'Created/Updated: {path}')
        if not created:
            print('No supported plot inputs found.')
    return {
        'workdir': str(workdir),
        'output_dir': str(output_dir),
        'created': created,
        'missing': missing,
    }


def cli():
    parser = argparse.ArgumentParser(description='Auto-plot nepactive outputs in a directory.')
    parser.add_argument('workdir', nargs='?', default='.', help='Directory containing md.log, thermo.out, loss.out, or run.in')
    parser.add_argument('-o', '--output-dir', default=None, help='Directory used to write PNG files; defaults to workdir')
    args = parser.parse_args()
    plot_auto(args.workdir, output_dir=args.output_dir, verbose=True)


if __name__ == '__main__':
    cli()
