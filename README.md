# nepactive

`nepactive` 是一个面向含能材料体系的主动学习工作流，覆盖 NEP 势训练、偏差驱动采样、ASE/MatterSim 标注、爆速测试和相关后处理。

## 功能概览

- 主动学习主流程：初始化、训练、采样、标注、迭代更新
- NEP 与 MatterSim 双路径支持
- 偏差分析与数据集压缩
- 分子识别、产物求解与结构生成
- 本地 CPU/GPU NEP backend

## 安装

推荐在仓库根目录执行：

```bash
uv pip install -e .
# 或
pip install -e .
```

安装完成后可直接使用：

```bash
nepactive
nep-identify
nep-product
nep-fps
```

### 可选本地 NEP backend

默认安装会尝试编译本地 NEP 扩展：

- CPU backend：默认尝试编译
- GPU backend：检测到 `nvcc` 或 `CUDA_HOME`/`CUDA_PATH` 后自动尝试编译
- GPU 工具链缺失时：跳过 GPU backend，不阻塞主包安装

如需显式控制：

```bash
NEP_NATIVE_BUILD=none uv pip install -e .
NEP_NATIVE_BUILD=cpu uv pip install -e .
NEP_NATIVE_BUILD=all uv pip install -e .
```

如果 GPU 编译较慢，建议显式指定目标架构，例如：

```bash
CUDAARCHS=89 uv pip install -e .
```

更完整的 native backend 说明见 [src/nepactive/native_nep/README.md](/workplace/liuzf/code/BRIDGE/src/nepactive/native_nep/README.md)。

## 环境要求

完整工作流通常还需要以下外部程序或环境：

- `GPUMD`
- `Packmol`
- 远程 VASP 环境

是否需要它们取决于你启用的具体模块。

## 快速开始

典型项目目录如下：

```text
project/
├── in.yaml
├── POSCAR
└── prep/
    ├── INCAR
    ├── KPOINTS
    └── POTCAR
```

在项目目录准备好 `in.yaml` 后，直接运行：

```bash
nepactive
```

默认主动学习流程会依次执行：

```text
init/
├── 初始结构生成 / 弛豫
├── 初始数据提取
└── 初始 train/test 划分

iter.XXXXXX/
├── 00.nep/     势函数训练
├── 01.gpumd/   采样任务
├── 02.label/   候选结构标注与数据集更新
└── 03.shock/   爆速测试
```

## 常用命令

```bash
nepactive                 # 主动学习主流程
nepactive shock           # 爆速测试
nepactive hod --gpu 0     # 爆热计算
nepactive OB              # 氧平衡工作流
nepactive remote          # 扫描 remote.dirs 并远程提交
nepactive mktask          # 从轨迹批量生成 task.*
nepactive reset           # 清理 task_finished/task_failed

nep-identify POSCAR
nep-product POSCAR
nep-fps dump.xyz --number 2000
```

也可以直接从源码入口运行：

```bash
python -m nepactive.main
python -m nepactive.main shock
```

## 配置要点

主配置文件为 `in.yaml`。常用项包括：

- `structure_files`：输入结构
- `ase_model`：`mattersim` 或 `nep89`
- `ase_model_file`：自定义模型文件
- `sampling`：采样、FPS 和候选池控制
- `shock`：爆速测试参数
- `remote`：远程提交设置

一个精简示例如下：

```yaml
project_name: HMX
python_interpreter: python

structure_files: ["POSCAR"]
kpoints_file: prep/KPOINTS
incar_file: prep/INCAR
potcar_file: prep/POTCAR

ase_model: mattersim
ase_model_file: null
ase_nep_backend: gpu

sampling:
  method: relative
  needed_frames: 10000
  max_candidate: 1000
  deviation_backend: auto
  fps_pot: nep89
  init_method: fps
  init_descriptor: nep
  dataset_method: fps
  dataset_descriptor: nep
  fps_pca_plot: true

shock:
  structure: POSCAR
  pressure_list: [20, 25, 30, 35, 40]
  steps: 100000
  time_step: 0.4
  pot: mattersim
```

如果项目根目录下放了自定义 `nep.in`，训练时会优先把它当作模板使用；程序仍会按当前轮次自动覆盖其中的 `generation`：

- 第 0 轮使用 `ini_train_steps`
- 后续轮次使用 `train_steps`

如果不想把模板命名为根目录 `nep.in`，也可以在 `in.yaml` 里显式设置：

```yaml
nep_template: prep/nep.in
```

如果你启用了：

```yaml
pot_inherit: true
```

建议显式给出：

```yaml
nep_in_header: "type 4 H C N O"
```

以避免元素顺序变化导致继承权重失配。

## 势函数与采样

主动学习中的 ASE 路径支持两种基础模型：

- `mattersim`
- `nep89`

其中：

- `ase_model: mattersim` 使用 MatterSim 路径
- `ase_model: nep89` 使用本地 `nep89` 模型，可配合本地 CPU/GPU backend
- 当 `sampling.init_descriptor` 设为 `nep` 且未显式给出 `init_nep_file` 时，init FPS 默认使用仓库根目录的 [resources/nep89_20250409.txt](/workplace/liuzf/code/BRIDGE/resources/nep89_20250409.txt)
- 当 `sampling.dataset_descriptor` 设为 `nep` 且未显式给出 `dataset_nep_file` 时，后续 FPS 由 `sampling.fps_pot` 控制：
- `fps_pot: nep89` 使用仓库自带的 `resources/nep89_20250409.txt`
- `fps_pot: self` 使用当前迭代 `00.nep/task.000000/nep.txt`
- `sampling.fps_pca_plot: true` 时，会在每次 FPS 采样后额外输出 PCA 覆盖图；图直接复用本次 FPS 已经算好的结构级 descriptor，不会重复算描述符
- `shock.pot` 可在 `mattersim` 和 `nep` 间切换

`nep-fps` 提供独立的 farthest point sampling 工具，例如：

```bash
nep-fps dump.xyz --number 2000
nep-fps dump.xyz --r2 0.95
nep-fps dump.xyz --number 500 --reference train.xyz
nep-fps dump.xyz --descriptor nep --model resources/nep89_20250409.txt --pca-plot fps_pca.png
```

更多采样和后端参数说明见源码内配置模板与相关模块。

## 其他工具

### `nep-identify`

用于基于邻居关系识别结构或轨迹中的分子组成。

### `nep-product`

用于根据元素组成求解可能的爆轰产物分布，并生成 Packmol 初始结构。

## 相关文档

- [docs/job_submission_guide.md](/workplace/liuzf/code/BRIDGE/docs/job_submission_guide.md)
- [src/nepactive/native_nep/README.md](/workplace/liuzf/code/BRIDGE/src/nepactive/native_nep/README.md)
- [src/nepactive/README_product.md](/workplace/liuzf/code/BRIDGE/src/nepactive/README_product.md)
- [examples/in_full_config.yaml](/workplace/liuzf/code/BRIDGE/examples/in_full_config.yaml)

## 许可证与引用

本仓库中的部分采样思路、NEP backend 集成方向及相关实现参考了 `NepTrainKit`：

- 仓库：`https://github.com/aboys-cb/NepTrainKit`
- 许可证：`GPL-3.0-or-later`

如果你继续分发或修改相关参考/移植代码，需要遵守对应许可证要求。学术使用时也建议一并引用 NepTrainKit。
