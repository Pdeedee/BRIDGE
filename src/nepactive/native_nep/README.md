# native_nep

这里放的是 `nepactive` 自带的可选本地 NEP CPU/GPU backend 源码。

目的很简单：

- 主库安装不依赖它
- 编译失败不影响主流程安装
- 用户需要更快的 native backend 时，再单独手工编译

## 手工编译

在 `nepactive` 根目录执行：

```bash
python build_native_nep.py build_ext --inplace
```

成功后会在当前目录生成：

- `nep_cpu*.so`
- `nep_gpu*.so`

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
