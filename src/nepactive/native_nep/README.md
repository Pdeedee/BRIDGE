# native_nep

这里放的是 `nepactive` 自带的可选本地 NEP CPU/GPU backend 源码。

目的很简单：

- editable install 时尽量自动尝试编译
- GPU 工具链缺失时不阻塞主库安装
- 需要更快的 native backend 时，也可以单独手工编译

## 安装时自动编译

在仓库根目录执行：

```bash
uv pip install -e .
# 或
pip install -e .
```

默认行为：

- CPU backend：默认尝试编译
- GPU backend：检测到 `nvcc` 或 `CUDA_HOME`/`CUDA_PATH` 后自动尝试编译
- 没有 CUDA 工具链时：跳过 GPU backend，不让安装失败
- GPU 构建会输出当前 `.cu` 的编译进度、`gencode` 来源以及 `nvcc` 线程数

可以显式控制：

```bash
NEP_NATIVE_BUILD=none uv pip install -e .
NEP_NATIVE_BUILD=cpu uv pip install -e .
NEP_NATIVE_BUILD=gpu uv pip install -e .
NEP_NATIVE_BUILD=all uv pip install -e .
```

如果 GPU 编译很慢，优先显式指定目标架构，避免默认 fatbin 太宽：

```bash
CUDAARCHS=89 uv pip install -e .
NEP_GPU_GENCODE="89" uv pip install -e .
```

如果是在没有驱动的登录节点编译，`nvidia-smi` 可能无法返回 compute capability，这时脚本会回退到默认架构集。建议直接指定：

```bash
CUDAARCHS=89 NEP_GPU_NVCC_THREADS=0 uv pip install -e .
```

## 手工编译

在 `src/nepactive` 目录执行：

```bash
python build_native_nep.py
```

这条命令默认等价于：

```bash
python build_native_nep.py build_ext --inplace
```

成功后会在 `src/nepactive/` 下生成：

- `nep_cpu*.so`
- `nep_gpu*.so`

如果已有 `.so` 且源码没有变化，脚本会跳过重复编译。需要强制重编译时：

```bash
NEP_NATIVE_REBUILD=1 python build_native_nep.py
```

## 使用建议

- 如果已经手工编好本地扩展，`deviation_backend: auto`
- 如果你就是要强制 GPU native，`deviation_backend: gpu`
- 如果你就是要强制 CPU native，`deviation_backend: cpu`

## 关于 `nep.txt`

如果你选择的是 `dataset_sampling_descriptor: nep`，那就必须提供 `dataset_sampling_nep_file`。

原因不是实现偷懒，而是 NEP 描述符本身就依赖模型文件里的信息，例如：

- 元素顺序
- 截断半径
- 描述符维度
- 模型内部参数定义

所以“NEP 描述符”不是一个脱离模型文件就能独立定义的无参特征。

如果你不想依赖 `nep.txt`，就用：

```yaml
dataset_sampling_descriptor: structural
```

这条路径使用的是 `nepactive` 内置结构指纹。
