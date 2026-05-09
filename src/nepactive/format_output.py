"""
格式化输出工具 - 统一管理所有 txt 文件的格式
"""

import os
import numpy as np


def save_formatted_txt(filename: str, data: np.ndarray, headers: list,
                       title: str = None, fmt: str = None):
    """
    保存格式化的 txt 文件，列对齐，方便 vi 查看

    Args:
        filename: 文件名
        data: 数据数组
        headers: 列标题列表
        title: 文件标题（可选）
        fmt: 数据格式（可选，默认 %.3f）
    """
    if fmt is None:
        fmt = "%.3f"

    # 计算每列的宽度
    col_widths = []
    for i, header in enumerate(headers):
        # 标题宽度
        header_width = len(header)

        # 数据宽度（格式化后）
        if data.ndim == 1:
            data_sample = data if i == 0 else None
        else:
            data_sample = data[:, i] if i < data.shape[1] else None

        if data_sample is not None:
            if isinstance(fmt, str):
                sample_str = fmt % data_sample[0] if len(data_sample) > 0 else fmt % 0
            else:
                sample_str = fmt[i] % data_sample[0] if len(data_sample) > 0 else fmt[i] % 0
            data_width = len(sample_str)
        else:
            data_width = 12

        col_widths.append(max(header_width + 2, data_width + 2))

    total_width = sum(col_widths) + len(col_widths) - 1

    with open(filename, 'w', encoding='utf-8') as f:
        # 写入标题
        if title:
            f.write("=" * total_width + "\n")
            f.write(title.center(total_width) + "\n")
            f.write("=" * total_width + "\n")

        # 写入列标题
        header_line = ""
        for i, (header, width) in enumerate(zip(headers, col_widths)):
            header_line += header.ljust(width)
        f.write(header_line + "\n")
        f.write("-" * total_width + "\n")

        # 写入数据
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        for row in data:
            row_line = ""
            for i, (value, width) in enumerate(zip(row, col_widths)):
                if isinstance(fmt, str):
                    value_str = fmt % value
                else:
                    value_str = fmt[i] % value
                row_line += value_str.ljust(width)
            f.write(row_line + "\n")


def save_thermo_txt(filename: str, data: np.ndarray, thermo_type: str = "gpumd"):
    """
    保存热力学数据

    Args:
        filename: 文件名
        data: 热力学数据
        thermo_type: 类型 ("gpumd" 或 "ase")
    """
    if thermo_type == "gpumd":
        headers = [
            "Temperature(K)",
            "Pot(eV)",
            "Pressure(GPa)",
            "Volume(Å³)",
            "Rel_Volume"
        ]
        title = "Thermodynamic Properties (GPUMD)"
        fmt = ["%12.2f", "%12.4f", "%12.3f", "%12.2f", "%12.4f"]
    else:  # ase
        headers = [
            "Time(ps)",
            "Etot(eV)",
            "Epot(eV)",
            "Ekin(eV)",
            "T(K)",
            "V(Å³)",
            "Sxx(GPa)",
            "Syy(GPa)",
            "Szz(GPa)",
            "Sxy(GPa)",
            "Syz(GPa)",
            "Szx(GPa)"
        ]
        title = "Thermodynamic Properties (ASE)"
        fmt = "%12.3f"

    save_formatted_txt(filename, data, headers, title, fmt)


def save_shock_vel_txt(filename: str, shock_vels: np.ndarray):
    """
    保存爆速数据

    Args:
        filename: 文件名
        shock_vels: 爆速数据 (可以是 1D 或 2D 数组)
    """
    shock_vels = np.asarray(shock_vels)

    if shock_vels.ndim == 1:
        if shock_vels.size in (3, 4):
            data = shock_vels.reshape(1, -1)
        else:
            data = shock_vels.reshape(-1, 1)
    else:
        data = shock_vels

    if data.shape[1] == 1:
        headers = ["Dv(km/s)"]
    elif data.shape[1] == 3:
        headers = ["Dv(km/s)", "V_CJ", "P_CJ(GPa)"]
    elif data.shape[1] == 4:
        headers = ["Dv(km/s)", "V_CJ", "P_CJ(GPa)", "rho(g/cm3)"]
    else:
        headers = [f"Shock_Vel_{i+1}(km/s)" for i in range(data.shape[1])]

    title = "Shock Velocity Results"
    fmt = "%12.3f"

    save_formatted_txt(filename, data, headers, title, fmt)


def append_shock_summary_txt(
    filename: str,
    iteration_tag: str,
    dv: float,
    vcj: float,
    pcj: float,
    rho: float,
    hod: float = None,
):
    """
    追加 shock 主流程摘要到 txt，保持列对齐。
    """
    headers = ["Iteration", "Dv(km/s)", "V_CJ", "P_CJ(GPa)", "rho(g/cm3)", "HOD(kJ/kg)"]
    widths = [15, 14, 14, 14, 14, 14]
    total_width = sum(widths)

    if hod is not None and not np.isfinite(hod):
        hod = None

    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=" * total_width + "\n")
            f.write("Shock Velocity and Detonation Results".center(total_width) + "\n")
            f.write("=" * total_width + "\n")
            f.write("".join(header.ljust(width) for header, width in zip(headers, widths)) + "\n")
            f.write("-" * total_width + "\n")

    hod_str = f"{hod:.2f}" if hod is not None else "N/A"
    row = [
        iteration_tag,
        f"{dv:.3f}",
        f"{vcj:.4f}",
        f"{pcj:.2f}",
        f"{rho:.3f}",
        hod_str,
    ]
    with open(filename, "a", encoding="utf-8") as f:
        f.write("".join(value.ljust(width) for value, width in zip(row, widths)) + "\n")


def append_shock_detail_txt(filename: str, shock_vels: np.ndarray, rho: float = None):
    """
    追加 shock 明细结果到 txt，适用于 `nepactive shock` 目录下的汇总文件。
    """
    headers = ["rho(g/cm3)", "Structure", "Dv(km/s)", "V_CJ", "P_CJ(GPa)"]
    widths = [14, 14, 14, 14, 14]
    total_width = sum(widths)

    data = np.asarray(shock_vels)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=" * total_width + "\n")
            f.write("Shock Velocity Results".center(total_width) + "\n")
            f.write("=" * total_width + "\n")
            f.write("".join(header.ljust(width) for header, width in zip(headers, widths)) + "\n")
            f.write("-" * total_width + "\n")

    with open(filename, "a", encoding="utf-8") as f:
        for index, row in enumerate(data):
            row_rho = float(row[3]) if row.shape[0] > 3 else rho
            row_vcj = float(row[1]) if row.shape[0] > 1 else np.nan
            row_pcj = float(row[2]) if row.shape[0] > 2 else np.nan
            values = [
                f"{row_rho:.3f}" if row_rho is not None else "N/A",
                f"struc.{index:03d}",
                f"{float(row[0]):.3f}",
                f"{row_vcj:.4f}" if np.isfinite(row_vcj) else "N/A",
                f"{row_pcj:.2f}" if np.isfinite(row_pcj) else "N/A",
            ]
            f.write("".join(value.ljust(width) for value, width in zip(values, widths)) + "\n")


def append_shock_hod_txt(filename: str, iteration: int, shock_vel: float, hod: float = None):
    """
    追加爆速和爆热数据到文件

    Args:
        filename: 文件名
        iteration: 迭代次数
        shock_vel: 爆速 (km/s)
        hod: 爆热 (kJ/mol)，可选
    """
    import os

    # 检查文件是否存在
    if not os.path.exists(filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Shock Velocity and Heat of Detonation Results".center(80) + "\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Iteration':<15} {'Shock_Vel(km/s)':<20} {'HOD(kJ/mol)':<20}\n")
            f.write("-" * 80 + "\n")

    # 追加数据
    with open(filename, 'a', encoding='utf-8') as f:
        hod_str = f"{hod:.2f}" if hod is not None else "N/A"
        f.write(f"iter.{iteration:06d:<8} {shock_vel:<20.3f} {hod_str:<20}\n")
