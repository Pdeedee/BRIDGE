# nepactive

基于主动学习的含能材料机器学习势函数（NEP）训练与爆轰性能预测工具。通过 NEP + MatterSim 的主动学习闭环，自动完成势函数训练、模型偏差采样、候选结构标注，并持续评估爆速、爆热等爆轰性能。

## 安装

```bash
cd nepactive
pip install -e .
```

依赖：ASE 3.23.0, MatterSim, NumPy 1.26.4, GPUMD, Packmol

## 命令行工具

安装后提供三个命令：

```bash
# 主程序
nepactive              # 运行主动学习闭环
nepactive --shock      # 爆速测试（多密度）
nepactive --hod        # 爆热计算（--gpu N 指定GPU）
nepactive --OB         # 氧平衡工作流
nepactive --remote     # 远程提交标注任务

# 独立工具
nep-identify POSCAR                # 识别结构中的分子组成
nep-identify dump.xyz --index -1   # 识别轨迹最后一帧
nep-product POSCAR                 # 生成爆轰产物结构
nep-product POSCAR --only-solve    # 只求解分子分布，不生成结构
```

## 主动学习工作流

准备工作目录：

```
project/
├── in.yaml     # 配置文件
├── POSCAR      # 初始结构
└── prep/       # VASP 输入文件（远程标注时需要）
    ├── INCAR
    ├── KPOINTS
    └── POTCAR
```

运行 `nepactive` 后自动执行以下循环：

```
初始化 (init/)
├── NVT 弛豫获取初始能量/压力/体积
├── 随机产物结构生成（Packmol）
└── 初始数据集提取

迭代 (iter.XXXXXX/)
├── 00.nep/     NEP 势函数训练（多个并行）
├── 01.gpumd/   GPUMD 模型偏差采样（多系综/压力/温度）
├── 02.label/   不确定性分析 → 候选结构选取
├── 03.shock/   爆速测试（达到条件时）
└── 标注 → 合并数据 → 下一轮迭代
```

收敛条件：爆速误差低于 `accuracy` 阈值，或达到最大迭代次数。支持断点续传（基于 `record.nep`）。

## 支持的系综

| 系综 | 实现 | 说明 |
|------|------|------|
| `nphugo` | Nose-Hoover chain (MTTK) | 默认 NPHugo，Hugoniot 约束恒温恒压 |
| `nphugo_scr` | SCR 气压计 + BDP 恒温器 | 随机胞缩放，参数与 GPUMD 一致 |
| `nvt` | Berendsen NVT | 恒温采样 |
| `npt` | MTTK NPT | 恒温恒压 |

在 `in.yaml` 中通过 `model_devi_general.ensembles` 或 `stable.ensemble` 选择。

## 爆速测试 (--shock)

支持两种势函数引擎：

```yaml
stable:
  shock_pot: "nep"         # GPUMD + NEP（默认）
  # shock_pot: "mattersim" # MatterSim + ASE
```

流程：对 `rhos` 列表中的每个密度，在 `shock_pressure_list` 各压力点运行 NPHugo/NPHugo_SCR，从 Hugoniot 曲线拟合爆速。

## 爆热计算 (--hod)

对最新迭代的 shock 结果，用 MatterSim 优化最终产物结构，计算 Q = (E_initial - E_final) / mass (kJ/kg)。

## 独立工具

### nep-product：爆轰产物结构生成

从 POSCAR 的元素组成求解可能的爆轰产物分子分布（MILP + 随机搜索），用 Packmol 堆积并用 MatterSim 优化。

```bash
nep-product POSCAR                        # 默认：求解 + 生成 + 优化
nep-product POSCAR -n 3 -d 1800           # 每种方案生成 3 个结构，密度 1800 kg/m³
nep-product POSCAR --only-solve --top-k 20  # 只求解前 20 个方案到 solutions.yaml
nep-product POSCAR --pdb-dir ./my_pdbs    # 使用自定义分子 PDB 文件
```

输出两种策略：能量最低方案 + 分子数最多方案，各保留 top-k 个解到 `solutions.yaml`。

支持的产物分子：H2, O2, N2, H2O, CO, CO2, CH4, NH3, CHN。非 CHON 元素（金属等）作为单原子处理。

### nep-identify：分子组成识别

基于共价半径邻居列表 + DFS 连通分量识别结构中的分子。

```bash
nep-identify POSCAR                       # 识别单帧结构
nep-identify dump.xyz --index ":"         # 分析整条轨迹
nep-identify dump.xyz --index "-1"        # 只看最后一帧
nep-identify POSCAR --mult 0.85           # 调整成键截断系数
```

输出：分子种类统计、唯一分子 PDB 文件、`molecule_counts.csv`。

## 配置文件 (in.yaml)

### 基本参数

```yaml
project_name: "HMX"
python_interpreter: "python"
gpu_available: [0,1,2,3]
task_per_gpu: 1
time_step: 0.2                    # fs
accuracy: 1                       # 爆速收敛精度 (%)
```

### NEP 训练

```yaml
pot_number: 4                     # 并行训练的势函数个数
pot_inherit: true                 # 继承上一代权重
ini_train_steps: 10000            # 第一代训练步数
train_steps: 5000                 # 后续训练步数
training_ratio: 0.8               # 训练集比例
pot_file: "/path/to/mattersim.pth"  # MatterSim 模型（标注用）
```

### 主动学习采样

```yaml
uncertainty_threshold: [0.3, 1]   # 不确定性选取阈值
uncertainty_mode: "max"           # max 或 mean
needed_frames: 10000              # 每代保存结构数
max_candidate: 1000               # 每代最大候选数
max_run_steps: 1000000            # 最大 MD 步数

model_devi_general:
  - ensembles: ["nphugo", "nvt"]
    structure_id: [[0,1], [0,1]]
    pressure: [60, 35, 10]        # GPa
    temperature: [3000]           # K
    pperiod: 2000                 # 压浴周期（步数）
```

### 爆速测试 (stable 段)

```yaml
stable:
  structure: "POSCAR"
  # ensemble: "nphugo"            # 可选 nphugo / nphugo_scr / nvt / npt
  # shock_pot: "nep"              # nep（默认）或 mattersim

  # 初始数据集参数
  struc_num: 1
  pressure: [20, 40, 60, 80]     # GPa
  temperature: [3000]             # K
  steps: 40000
  dump_freq: 10

  # 爆速测试参数
  shock_pressure_list: [20,25,30,35,40,45,50,55]  # GPa
  shock_steps: 600000
  analyze_range: [0.5, 1]

  # Nose-Hoover (MTTK) 参数（ensemble="nphugo"）
  # tfreq: 0.025                  # 恒温器频率（无量纲）
  # pfreq:                        # 气压计频率（默认 1/400/dt）

  # SCR 参数（ensemble="nphugo_scr"，与 GPUMD 输入一致）
  # tau_t: 100                    # BDP 恒温器弛豫时间（timestep 单位）
  # tau_p: 2000                   # SCR 气压计弛豫时间（timestep 单位）
  # pmode: "iso"                  # iso / x / y / z

rhos: [1.62, 1.65, 1.68, 1.70, 1.72]  # 爆速测试密度列表 (g/cm³)
```

### 远程计算

```yaml
fp_command: "mpirun -np 96 vasp_std"
ssh_username: "user@cluster"
ssh_hostname: "login.cluster.com"
ssh_port: 22
remote_root: "/path/to/remote/workspace"
slurm_header_script:
  - "#!/bin/bash"
  - "#SBATCH --job-name=vasp"
  - "#SBATCH -N 1"
  - "#SBATCH -n 96"
```

## 输出文件

```
project/
├── nepactive.log          # 运行日志
├── record.nep             # 迭代记录（断点续传依据）
├── shock_vel.txt          # 爆速结果
├── init/
│   ├── properties.txt     # 初始 ρ, E, P, V, N
│   └── struc.000/         # 生成的产物结构
├── iter.000000/
│   ├── 00.nep/task.*/nep.txt    # 训练好的势函数
│   ├── 01.gpumd/task.*/         # MD 轨迹 + 热力学数据
│   └── 03.shock/                # 爆速测试结果
└── ...
```

## 常见问题

**训练中断了？** 直接重新运行 `nepactive`，会从 `record.nep` 记录的位置继续。删除 `record.nep` 可重新开始。

**不用远程 VASP？** 不配置 SSH 参数即可，标注会使用 MatterSim。

**纯金属体系？** 自动识别非 CHON 元素并作为单原子处理。

**内存不足？** 减小 `task_per_gpu`、`pot_number` 或 `max_candidate`。

## 参考文献

- GPUMD: Fan et al., J. Chem. Phys. 157, 114801 (2022)
- NEP: Fan et al., Phys. Rev. B 104, 104309 (2021)
- MatterSim: Yang et al., arXiv:2405.04967 (2024)
- SCR barostat: Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020)
- BDP thermostat: Bussi, Donadio & Parrinello, J. Chem. Phys. 126, 014101 (2007)
