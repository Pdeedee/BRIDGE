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

## 其他文档

- 产物求解与结构生成说明见 `README_product.md`。

