# nepactive

`nepactive` 是一个面向含能材料的主动学习工作流，用于 NEP 势函数训练、模型偏差采样、MatterSim/远程标注、爆速测试和爆热计算。

当前仓库的主文档维护在仓库根目录 `README.md`。`src/nepactive/README.md` 仅保留跳转说明。

## 安装

### 基础安装

在仓库根目录执行：

```bash
pip install -e .
```

安装完成后，`pyproject.toml` 中注册的命令行入口会直接可用：

```bash
nepactive
nep-identify
nep-product
nep-fps
```

如果你刚拉取到最新代码，尤其是新增了命令行入口例如 `nep-fps`，需要重新执行一次：

```bash
pip install -e .
```

否则新的 console script 不会出现在当前环境里。

当前 `pyproject.toml` 中声明的基础依赖包括：

- `ase==3.23.0`
- `mattersim`
- `numpy==1.26.4`

### 可选依赖

运行完整工作流时还需要根据你的实际路径启用或安装：

- `GPUMD`
- `Packmol`
- 远程 VASP 环境（如果使用 `remote`/VASP 标注）

### 本地 NEP backend 编译

如果你要使用本地 `nep89` ASE 势或者更快的本地 NEP 描述符/偏差分析后端，可以按需编译 native backend：

```bash
cd src/nepactive
python build_native_nep.py build_ext --inplace
```

编译成功后会在 `src/nepactive/` 下生成本地扩展，例如：

- `nep_cpu*.so`
- `nep_gpu*.so`

这一步是可选的；主库安装不应该依赖它强制成功。

## 命令行入口

安装后提供以下脚本：

```bash
nepactive                 # 默认主动学习主流程
nepactive shock           # 爆速测试
nepactive hod --gpu 0     # 爆热计算
nepactive OB              # 氧平衡工作流
nepactive remote          # 扫描 remote.dirs 并远程提交
nepactive mktask          # 从轨迹批量生成 task.*
nepactive reset           # 清理 task_finished/task_failed

nep-identify POSCAR
nep-product POSCAR
nep-fps dump.xyz --number 2000
nep-fps dump.xyz --r2 0.95
```

也可以直接从源码入口运行：

```bash
python -m nepactive.main
python -m nepactive.main shock
```

## 工作流概览

典型项目目录：

```text
project/
├── in.yaml
├── POSCAR
└── prep/
    ├── INCAR
    ├── KPOINTS
    └── POTCAR
```

默认主动学习流程会依次执行：

```text
init/
├── 初始结构生成 / 弛豫
├── 初始数据提取
└── 初始 train/test 划分

iter.XXXXXX/
├── 00.nep/     多个 NEP 势并行训练
├── 01.gpumd/   采样任务
├── 02.label/   候选结构标注与数据集更新
└── 03.shock/   爆速测试
```

## 势函数与 `nep_in_header`

### 默认主流程

- 主流程训练和 deviation 分析走 NEP 路径。
- `shock_vel_test()` 默认使用当前迭代训练得到的 `iter.xxx/00.nep/task.000000/nep.txt`。

### 主动学习 ASE 模型切换

主动学习中的两个 ASE 路径现在统一支持两种基础模型：

- `mattersim`
- `nep89`

覆盖范围：

- 初始 `init` 数据集生成
- `02.label` 候选结构标注
- 坏任务恢复时的 ASE 续跑脚本

推荐配置：

```yaml
ase_model: nep89                     # mattersim / nep89
ase_model_file: null                 # 不写时，nep89 默认读取仓库顶层 resources/nep89_20250409.txt
ase_nep_backend: gpu                 # 当 ase_model=nep89 时使用 gpu / cpu / auto
label_engine: mattersim              # 这里表示走本地 ASE 标注路径；实际模型由 ase_model 决定
```

说明：

- `ase_model: mattersim` 时，默认使用 MatterSim 预训练模型；如果你有自定义模型，也可以通过 `ase_model_file` 显式覆盖。
- `ase_model: nep89` 时，默认从仓库顶层 `resources/nep89_20250409.txt` 加载；也可以用 `ase_model_file` 换成你自己的 `nep.txt`。
- 旧参数 `pot_file` 仍然兼容，但现在等价于 `ase_model_file`，不再推荐作为主入口。

### `nepactive shock`

- `shock` 子命令使用 `in.yaml` 中 `shock.pot` 指定的势函数类型。
- `shock.pot: mattersim` 走 MatterSim。
- `shock.pot: nep` 走 NEP；若未显式提供 `shock.nep`，默认使用当前目录 `nep.txt`。

示例：

```yaml
shock:
  pot: mattersim   # 或 nep
  # nep: /path/to/nep.txt
```

### `nep_in_header` 注意事项

`nep.in` 的 `type N ...` header 现在支持从多个 `structure_files` 自动推断，但如果你开启：

```yaml
pot_inherit: true
```

强烈建议显式写死 `nep_in_header`，避免元素顺序变化导致继承旧权重时势函数失真：

```yaml
nep_in_header: "type 4 H C N O"
```

当前代码会在 `pot_inherit=True` 且自动推断 header 与上一代 `nep.in` 不一致时直接报错。

## 采样与 FPS

当前代码里，以下路径都已支持用 FPS 替代随机采样：

- 初始 `init` 数据切分
- 每轮 `max_candidate` 候选池截断
- `02.label` 后的数据集划分
- 坏任务重跑后的候选池合并

### `nep-fps`

`nep-fps` 是一个独立命令行工具，用来从单个结构文件、轨迹文件或多结构 `xyz/extxyz` 中做 farthest point sampling，输出一个代表性子集。

常见用法：

```bash
nep-fps dump.xyz --number 2000
nep-fps dump.xyz --r2 0.95
nep-fps dump.xyz --number 500 --reference train.xyz
nep-fps dump.xyz --index '0:1000:5' --output selected.xyz
```

主要参数：

- `--number`：最多保留多少个结构。
- `--r2`：达到指定覆盖度后提前停止，不必选满 `--number`。
- `--output`：输出文件名，默认 `fps_selected.xyz`。
- `--index`：ASE 风格切片，例如 `:`、`0:1000:5`。
- `--descriptor`：描述符类型，支持 `structural`、`nep`、`soap`、`auto`。
- `--model`：当 `--descriptor nep` 时指定 `nep.txt/nep89` 模型文件。
- `--backend`：NEP 描述符后端，支持 `auto`、`gpu`、`cpu`、`native`。
- `--reference`：给定一个已覆盖的数据集，新的 FPS 会相对这份 reference 继续挑点，而不是只在当前输入里做局部去重。
- `--reference-index`：reference 文件的 ASE 风格切片。
- `--min-dist`：最小距离阈值，小于该值时提前停止。

输出内容：

- 一个筛选后的结构文件，默认 `fps_selected.xyz`
- 终端摘要，包括选中数、总结构数、描述符类型、reference 点数、R2 和被选中的 frame 索引

这个工具适合单独分析轨迹，也适合在主动学习之外手动压缩数据集。

推荐把主动学习相关参数集中写在 `sampling:` 下：

```yaml
sampling:
  method: relative
  time_step: 0.2
  needed_frames: 10000
  max_candidate: 1000
  max_reference_points: 10000
  final_r2_threshold: null

  uncertainty_threshold: [0.3, 1]
  uncertainty_level: 1
  uncertainty_mode: max
  energy_threshold: 1

  deviation_backend: auto              # auto / native / gpu / cpu
  enable_molecule_analysis: false
  molecule_analysis_backend: ase

  init_method: fps
  init_descriptor: soap                # soap / structural / nep
  init_min_dist: 0.0

  dataset_method: fps
  dataset_descriptor: structural       # structural / nep
  dataset_min_dist: 0.0
  dataset_backend: auto                # auto / native / gpu / cpu
  # dataset_nep_file: /path/to/nep.txt

  general:
    ensembles: [nphugo_scr, nvt]
    structure_id: [[0, 1], [0, 1]]
    pressure: [60, 35, 10]
    pperiod: 2000
    temperature: [3000]
```

说明：

- `sampling.dataset_descriptor: structural` 使用内置结构指纹做 FPS，不依赖 `nep.txt`。
- `sampling.dataset_descriptor: nep` 使用本地 NEP native backend 计算描述符，需要 `sampling.dataset_nep_file`。
- `sampling.init_descriptor: soap` 用于初始 `init` 数据的 FPS，依赖 `dscribe`。
- `sampling.max_reference_points` 现在表示全局 reference 代表集上限；程序会维护一个落盘的增量 representative cache，而不是每轮对全历史 reference 全量重算。
- `sampling.final_r2_threshold` 可用于最后一步全局 FPS 的提前停止；达到该描述符覆盖度后，不必再选满 `max_candidate`。
- 一旦轨迹在第 `i` 帧出现异常，候选池只保留 `i` 之前的结构；异常帧及其后的结构不会参与后续 FPS。
- 合并 `candidate.xyz` 前会再按 `shortest_d` 过滤最短原子距离异常的结构。
- 如果你想强制暴露后端错误，就显式写 `gpu/cpu/native`，不要用 `auto`。

## 分子识别与后处理

- 默认 `sampling.enable_molecule_analysis: false`，因为整条轨迹做分子识别后处理较慢。
- 当前 `molecule_analysis_backend` 支持 `ase`。
- 分子识别本身还可以单独通过 `nep-identify` 使用。

## 本地 NEP CPU/GPU backend

仓库已内置一份本地 native NEP backend 源码与编译脚本，位置：

- `src/nepactive/native_nep/`
- `src/nepactive/build_native_nep.py`

如果你已经完成上面的基础安装，这里只需要额外执行编译步骤，不需要重新手动拼装 Python 包：

```bash
cd src/nepactive
python build_native_nep.py build_ext --inplace
```

成功后会在 `src/nepactive/` 下生成：

- `nep_cpu*.so`
- `nep_gpu*.so`

然后可在 `in.yaml` 中设置：

```yaml
sampling:
  deviation_backend: gpu
```

或：

```yaml
sampling:
  dataset_backend: gpu
```

## `in.yaml` 结构

当前推荐配置骨架：

```yaml
project_name: HMX
yaml_synchro: true
python_interpreter: python

pot_number: 4
pot_inherit: true
ini_train_steps: 10000
train_steps: 5000
training_ratio: 0.8
nep_in_header: "type 4 H C N O"   # 推荐在 pot_inherit=true 时显式给出
ase_model: mattersim                # mattersim / nep89
ase_model_file: null                # nep89 不写时默认读 resources/nep89_20250409.txt
ase_nep_backend: gpu                # gpu / cpu / auto
pot_file: null                      # 旧参数兼容；等价于 ase_model_file

gpu_available: [0, 1, 2, 3]
task_per_gpu: 1
max_temp: 10000
shortest_d: 0.5

structure_files: ["POSCAR"]
kpoints_file: prep/KPOINTS
incar_file: prep/INCAR
potcar_file: prep/POTCAR

sampling:
  ...

init:
  struc_num: 1
  pressure: [20, 40, 60, 80]
  temperature: [3000]
  steps: 40000
  time_step: 0.2
  dump_freq: 10

shock:
  structure: POSCAR
  pressure_list: [20, 25, 30, 35, 40, 45, 50, 55]
  steps: 100000
  time_step: 0.4
  dump_freq: 10
  analyze_range: [0.8, 1]
  pot: mattersim

remote:
  dirs: ["."]
  fp_command: mpirun -np 96 vasp_std
  ssh_username: user@cluster
  ssh_hostname: login.cluster.com
  ssh_port: 22
  remote_root: /path/to/remote/workspace
```

## 其他工具

### `nep-product`

用于根据元素组成求解可能的爆轰产物分布，并生成 Packmol 初始结构。

### `nep-identify`

用于基于邻居关系识别结构或轨迹中的分子组成。

## NepTrainKit 引用与许可证说明

本仓库中的部分采样思路、NEP backend 集成方向及相关实现参考了 `NepTrainKit`：

- 仓库：`https://github.com/aboys-cb/NepTrainKit`
- 许可证：`GNU General Public License v3.0 or later (GPL-3.0-or-later)`

如果你在本仓库中继续复用、分发或修改这些参考/移植的 GPL 代码，需要遵守 GPL 的分发义务，并保留相应版权与许可证声明。

建议在学术使用中同时引用 NepTrainKit：

```bibtex
@article{CHEN2025109859,
  title = {NepTrain and NepTrainKit: Automated active learning and visualization toolkit for neuroevolution potentials},
  journal = {Computer Physics Communications},
  volume = {317},
  pages = {109859},
  year = {2025},
  issn = {0010-4655},
  doi = {10.1016/j.cpc.2025.109859},
  author = {Chengbing Chen and Yutong Li and Rui Zhao and Zhoulin Liu and Zheyong Fan and Gang Tang and Zhiyong Wang}
}
```

## 其他文档

- `src/nepactive/README_product.md`：产物求解与结构生成说明
