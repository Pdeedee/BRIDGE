# NEPactive - Neural Evolution Potential Active Learning Framework

NEPactive 是一个基于主动学习的神经进化势（NEP）训练框架，专门用于分子动力学模拟和爆轰性能预测。

## 主要特性

- 🚀 **主动学习训练**: 自动化的 NEP 势函数训练流程
- 🔬 **初始数据集生成**: 支持 CHON 分子和金属单原子的结构生成
- 💥 **爆速测试**: 基于 Hugoniot 曲线的冲击波速度计算
- 🎯 **不确定性采样**: 基于模型偏差的智能结构选择
- 🔄 **远程任务提交**: 支持 SSH 远程提交 VASP 第一性原理计算
- 📊 **自动化工作流**: 从初始结构到收敛势函数的全自动流程

## 安装

### 依赖项

```bash
# Python 依赖
pip install numpy scipy ase mattersim pyyaml pandas tqdm

# 外部工具
# - GPUMD: GPU 加速分子动力学模拟
# - Packmol: 分子结构打包工具
# - VASP (可选): 第一性原理计算
```

### 安装 NEPactive

```bash
cd /path/to/nepactive
pip install -e .
```

## 快速开始

### 1. 准备输入文件

创建工作目录并准备以下文件：

```
your_project/
├── in.yaml          # 主配置文件
├── POSCAR           # 初始结构文件
└── prep/            # 第一性原理计算文件（可选）
    ├── INCAR
    ├── KPOINTS
    └── POTCAR
```

### 2. 配置 in.yaml

```yaml
# 项目基本信息
project_name: "MyProject"
yaml_synchro: true

# NEP 训练参数
nep_in_header: "type 2 V N"  # 元素类型
accuracy: 1                   # 收敛精度（爆速误差百分比）
ini_frames: 1500             # 初始数据集结构数
ini_train_steps: 10000       # 第一代训练步数
train_steps: 5000            # 后续训练步数
pot_number: 4                # 势函数个数

# 初始数据集生成
ini_engine: "ase"
pot_file: "/path/to/mattersim-v1.0.0-1M.pth"

# 结构文件
structure_files:
  - "POSCAR"
  - "init/struc.000/structure/POSCAR"

# 主动学习参数
uncertainty_threshold: [0.3, 1]
uncertainty_level: 1
uncertainty_mode: "max"
needed_frames: 10000
max_candidate: 1000

# GPU 设置
gpu_available: [0, 1, 2, 3]
task_per_gpu: 1

# 动力学系综设置
model_devi_general:
  - ensembles: ["nphugo", "nvt"]
    structure_id: [[0, 1], [0, 1]]
    pressure: [60, 35, 10]
    pperiod: 2000
    temperature: [3000]

# 初始结构生成和爆速测试
stable:
  structure: "POSCAR"
  struc_num: 1                    # 生成的初始结构数
  pressure: [20, 40, 60, 80]      # 初始数据集压力
  temperature: [3000]
  steps: 40000

  # 爆速测试参数
  gpumd_pressure_list: [20, 25, 30, 35, 40, 45, 50, 55]
  gpumd_steps: 600000
  time_step: 0.2
  analyze_range: [0.5, 1]

# 远程计算设置（可选）
fp_command: "mpirun -np 96 vasp_std"
ssh_username: "user@cluster"
ssh_hostname: "login.cluster.com"
ssh_port: 22
remote_root: "/path/to/remote/workspace"

slurm_header_script:
  - "#!/bin/bash"
  - "#SBATCH --job-name=vasp"
  - "#SBATCH -p partition_name"
  - "#SBATCH -N 1"
  - "#SBATCH -n 96"
  - "module load vasp"

# 爆速测试控制
shock_test_interval: 1
shock_test_begin_step: 400000
if_stable_run: true
```

### 3. 运行训练

```bash
cd your_project
nepactive
```

或使用 nohup 后台运行：

```bash
nohup nepactive > log 2>&1 &
tail -f log
```

## 工作流程

### 阶段 1: 初始化 (init/)

1. **结构生成**:
   - 对于 CHON 体系：自动求解分子分布（CO2, H2O, N2 等）
   - 对于金属体系：支持单原子（V, Fe, Cu 等）
   - 使用 Packmol 打包生成初始结构

2. **初始数据集**:
   - 使用 MatterSim 运行 NVT/NPT 系综
   - 生成约 1500 帧训练数据
   - 计算初始能量和压力

### 阶段 2: 主动学习循环 (iter.XXXXXX/)

每个迭代包含以下步骤：

```
iter.000000/
├── 00.nep/          # NEP 训练
│   ├── task.000000/ # 势函数 1
│   ├── task.000001/ # 势函数 2
│   ├── task.000002/ # 势函数 3
│   └── task.000003/ # 势函数 4
├── 01.gpumd/        # 模型偏差采样
│   ├── task.000000/ # NPHugo 系综
│   ├── task.000001/ # NVT 系综
│   └── ...
├── 02.fp/           # 第一性原理计算（远程）
│   ├── task.000000/
│   └── ...
└── 03.train/        # 数据集更新
```

#### 步骤说明：

1. **00.nep - NEP 训练**
   - 训练多个势函数（默认 4 个）
   - 使用 GPUMD 的 nep 可执行文件
   - 输出 nep.txt 势函数文件

2. **01.gpumd - 模型偏差采样**
   - 使用训练好的势函数运行 MD
   - 计算不确定性（多个势函数的预测偏差）
   - 选择高不确定性结构作为候选

3. **02.fp - 第一性原理计算**
   - 将候选结构提交到远程集群
   - 使用 VASP 计算精确的能量和力
   - 自动下载计算结果

4. **03.train - 数据集更新**
   - 将新数据加入训练集
   - 准备下一轮迭代

### 阶段 3: 爆速测试

当训练收敛后（或达到指定步数），自动进行爆速测试：

1. 使用最终势函数运行冲击波模拟
2. 计算不同压力下的 Hugoniot 曲线
3. 拟合得到爆速 (D) 和粒子速度 (u) 关系
4. 输出爆速预测结果

## 配置参数详解

### 基本参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `project_name` | 项目名称 | - |
| `yaml_synchro` | 每步重新读取配置 | true |
| `nep_in_header` | NEP 元素类型定义 | - |
| `accuracy` | 收敛精度（%） | 1 |

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `ini_frames` | 初始数据集大小 | 1500 |
| `ini_train_steps` | 第一代训练步数 | 10000 |
| `train_steps` | 后续训练步数 | 5000 |
| `pot_number` | 势函数个数 | 4 |
| `pot_inherit` | 继承上一代势函数 | true |
| `training_ratio` | 训练集比例 | 0.8 |

### 主动学习参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `uncertainty_threshold` | 不确定性阈值 [下限, 上限] | [0.3, 1] |
| `uncertainty_level` | 不确定性计算级别 | 1 |
| `uncertainty_mode` | 不确定性模式 (max/mean) | "max" |
| `energy_threshold` | 能量阈值 (eV) | 1 |
| `needed_frames` | 每代需要的结构数 | 10000 |
| `max_candidate` | 最大候选结构数 | 1000 |

### 动力学参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `time_step` | 时间步长 (fs) | 0.2 |
| `ini_run_steps` | 第 0 代运行步数 | 100000 |
| `max_run_steps` | 最大运行步数 | 1000000 |
| `gpu_available` | 可用 GPU 列表 | [0,1,2,3] |
| `task_per_gpu` | 每个 GPU 的任务数 | 1 |

### 初始结构生成参数 (stable)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `structure` | 初始结构文件 | "POSCAR" |
| `struc_num` | 生成的结构数 | 1 |
| `pressure` | 压力列表 (GPa) | [20,40,60,80] |
| `temperature` | 温度列表 (K) | [3000] |
| `steps` | MD 步数 | 40000 |
| `dump_freq` | 输出频率 | 10 |

### 爆速测试参数 (stable)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `gpumd_pressure_list` | 测试压力列表 (GPa) | [20,25,30,...] |
| `gpumd_steps` | 测试步数 | 600000 |
| `analyze_range` | 分析范围 [开始, 结束] | [0.5, 1] |
| `shock_test_interval` | 测试间隔（代数） | 1 |
| `shock_test_begin_step` | 开始测试的步数 | 400000 |

### 远程计算参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `fp_command` | VASP 运行命令 | - |
| `ssh_username` | SSH 用户名 | - |
| `ssh_hostname` | SSH 主机名 | - |
| `ssh_port` | SSH 端口 | 22 |
| `remote_root` | 远程工作目录 | - |
| `group_numbers` | 任务分组数 | 10 |
| `slurm_header_script` | SLURM 脚本头 | - |

## 支持的体系

### CHON 分子体系

自动求解以下分子的最优分布：
- CH4 (甲烷)
- CO (一氧化碳)
- CO2 (二氧化碳)
- H2 (氢气)
- H2O (水)
- N2 (氮气)
- NH3 (氨)
- O2 (氧气)
- CHN (氰化氢)

### 金属和其他元素

支持任意单原子元素，如：
- V (钒)
- Fe (铁)
- Cu (铜)
- Al (铝)
- 等等

### 混合体系

支持 CHON 分子 + 金属单原子的混合体系，例如：
- VN5: 48个V + 240个N → 48个V + 120个N2
- FeO: Fe + O2 混合

## 输出文件

### 训练过程

- `nepactive.log`: 详细日志
- `record.nep`: 训练记录（可删除以重新开始）
- `steps.txt`: 当前迭代的步数信息

### 每个迭代

- `iter.XXXXXX/00.nep/task.XXXXXX/nep.txt`: 训练好的势函数
- `iter.XXXXXX/00.nep/task.XXXXXX/loss.out`: 训练损失
- `iter.XXXXXX/01.gpumd/task.XXXXXX/dump.xyz`: MD 轨迹
- `iter.XXXXXX/01.gpumd/task.XXXXXX/thermo.out`: 热力学输出

### 爆速测试

- `shock_vel.txt`: 冲击波速度结果
- `total.txt`: 汇总数据
- 各个压力点的 Hugoniot 数据

## 常见问题

### 1. 如何处理纯金属体系？

对于纯金属体系（如纯 V），代码会自动识别并将其作为单原子处理：

```yaml
stable:
  structure: "POSCAR"  # V 的 POSCAR
  struc_num: 1
```

代码会自动：
- 检测到 V 是非 CHON 元素
- 创建 V.pdb 文件
- 使用 Packmol 生成结构

### 2. 如何调整收敛标准？

修改 `accuracy` 参数（爆速误差百分比）：

```yaml
accuracy: 1  # 1% 误差
```

### 3. 如何跳过初始结构生成？

如果已经有 `init/struc.000/structure/POSCAR`，代码会自动跳过生成步骤。

或者设置：

```yaml
stable:
  struc_num: 0  # 不生成新结构
```

### 4. 如何只在本地运行（不使用远程 VASP）？

注释掉或删除远程计算相关参数：

```yaml
# fp_command: "mpirun -np 96 vasp_std"
# ssh_username: "user@cluster"
# ...
```

代码会使用 MatterSim 进行标注（而不是 VASP）。

### 5. 训练过程中断了怎么办？

NEPactive 支持断点续传。只需重新运行 `nepactive`，它会：
- 读取 `record.nep` 文件
- 从上次中断的地方继续

如果要重新开始，删除 `record.nep` 文件。

### 6. 如何查看训练进度？

```bash
# 查看日志
tail -f nepactive.log

# 查看当前迭代
cat record.nep

# 查看爆速测试结果
cat shock_vel.txt
```

### 7. 内存不足怎么办？

减少并行任务数：

```yaml
task_per_gpu: 1  # 减少到 1
pot_number: 2    # 减少势函数个数
```

### 8. 如何加速训练？

- 增加 GPU 数量：`gpu_available: [0,1,2,3,4,5,6,7]`
- 减少训练步数：`train_steps: 3000`
- 减少候选结构数：`max_candidate: 500`
- 使用更少的势函数：`pot_number: 2`

## 引用

如果使用 NEPactive，请引用：

```bibtex
@article{nepactive2024,
  title={NEPactive: Active Learning Framework for Neural Evolution Potentials},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

相关工具引用：

- **GPUMD**: Fan et al., J. Chem. Phys. 157, 114801 (2022)
- **NEP**: Fan et al., Phys. Rev. B 104, 104309 (2021)
- **MatterSim**: Zuo et al., arXiv:2405.04967 (2024)

## 许可证

[添加许可证信息]

## 联系方式

- 问题反馈: [GitHub Issues]
- 邮箱: [your.email@example.com]

## 更新日志

### v1.0.0 (2024-01-15)

- ✅ 支持 CHON 分子和金属单原子混合体系
- ✅ 自动创建单原子 PDB 文件
- ✅ 修复原子数计算逻辑
- ✅ 修复 scipy bounds 参数兼容性
- ✅ 修复 struc 目录重复创建问题
- ✅ 改进错误处理和日志输出

---

**祝您使用愉快！如有问题，欢迎提 Issue。**
