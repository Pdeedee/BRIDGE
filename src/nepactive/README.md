# nepactive

`nepactive` 是一个用于主动学习训练与冲击波测试的工作流脚本集合。

## 命令入口

推荐从包根目录执行：

```bash
python -m nepactive.main [command]
```

可用子命令：

- `remote`：扫描 `in.yaml` 中 `remote.dirs` 并远程提交
- `mktask`：从结构文件批量生成 VASP `task.*`
- `reset`：重置 `task_finished` / `task_failed` 标记
- `shock`：执行冲击波测试
- `OB`：执行 OB 流程
- `hod`：计算爆热
- 不带子命令：执行默认主动学习流程（`nepactive` 主流程）

## 势函数使用规则（重要）

### 1) 默认主动学习流程（不带子命令）

- 训练与模型偏差分析按 NEP 路径运行。
- 主流程内的 `shock_vel_test()` 仍固定使用当前迭代生成的 NEP（`iter.xxx/00.nep/.../nep.txt`）。

补充：

- `nep.in` 的 header 现在会从多个 `structure_files` 自动识别元素，并生成 `type N ...`。
- 如果你想手动覆盖元素或顺序，仍然可以在 `in.yaml` 里显式设置，例如：

```yaml
nep_in_header: "type 4 H C N O"
```

### 2) `nepactive shock`

- 使用 `in.yaml` 中 `shock.pot` 指定的势函数类型，不再强制改成 `nep`。
- `shock.pot: mattersim`：走 MatterSim 路径。
- `shock.pot: nep`：走 NEP 路径；若未设置 `shock.nep`，默认使用当前目录 `nep.txt`。

示例：

```yaml
shock:
  pot: "mattersim"   # 或 "nep"
  # nep: "/path/to/nep.txt"  # pot=nep 时可显式指定
```

## 数据采样与 Deviation

当前代码里，初始 `init` 数据切分、`max_candidate` 截断、以及 MatterSim 标注后的 `iter_train/iter_test` 切分，已经支持用 FPS 代替随机采样。

推荐把主动学习采样相关配置集中写在 `sampling:` 下：

```yaml
sampling:
  method: relative
  time_step: 0.2
  needed_frames: 10000
  max_candidate: 1000
  max_reference_points: 10000

  uncertainty_threshold: [0.3, 1]
  uncertainty_level: 1
  uncertainty_mode: max
  energy_threshold: 1

  deviation_backend: auto                 # auto / native / gpu / cpu
  enable_molecule_analysis: false
  molecule_analysis_backend: ase         # 当前仅支持 ase

  init_method: fps
  init_descriptor: soap
  init_min_dist: 0.0

  dataset_method: fps
  dataset_descriptor: structural         # structural / nep
  dataset_min_dist: 0.0
  dataset_backend: auto
  # dataset_nep_file: /path/to/nep.txt   # 当 dataset_descriptor: nep 时需要

  general:
    ensembles: [nphugo_scr, nvt]
    structure_id: [[0, 1], [0, 1]]
    pressure: [60, 35, 10]
    pperiod: 2000
    temperature: [3000]
```

说明：

- `sampling.dataset_descriptor: structural`：使用内置结构指纹做 FPS，不依赖 `nep.txt`。
- `sampling.dataset_descriptor: nep`：使用本地 NEP native backend 计算描述符；这条路径需要 `sampling.dataset_nep_file`。
- `sampling.init_descriptor: soap`：初始 `init` 数据的 FPS 描述符，依赖 `dscribe`，不依赖 `nep.txt`。
- `sampling.dataset_backend: auto`：优先用 native NEP 描述符，失败时回退到 `structural` 指纹 FPS。
- `sampling.max_reference_points`：dataset-aware FPS 之前，先从已有 reference 集里随机截断到这个上限，避免距离矩阵过大。
- `sampling.deviation_backend: auto`：优先用 GPU native backend，失败时回退到 CPU native backend。
- `sampling.enable_molecule_analysis: false`：默认不做分子识别；开启后会额外分析 `dump.xyz`，后处理会明显变慢。
- 一旦轨迹在第 `i` 帧出现异常，候选池只保留 `i` 之前的结构；异常帧及其后的结构不会参与后续 FPS。
- 全局 `candidate.xyz` 合并和坏任务重跑合并前，会再按 `shortest_d` 过滤异常最近邻距离结构。
- 如果你想强制暴露错误，就显式写 `gpu/cpu/native`，不要用 `auto`。

## 本地编译 NEP CPU/GPU 扩展

`nepactive` 已内置一份本地 [native_nep](/workplace/liuzf/code/nepactive/src/nepactive/native_nep) 源码和编译脚本，不需要再走 `NepTrainKit` 的上层程序。

这是可选组件，不应该阻塞主库安装。你可以先装主库，再按需单独编译 native backend。

在当前目录执行：

```bash
python build_native_nep.py build_ext --inplace
```

成功后会在当前包目录生成：

- `nep_cpu*.so`
- `nep_gpu*.so`

然后可以在 `in.yaml` 里把：

```yaml
sampling:
  deviation_backend: auto
```

或更明确地写成：

```yaml
sampling:
  deviation_backend: gpu
```

## 其他文档

- 产物求解与结构生成说明见 `README_product.md`。
