# 作业提交系统使用指南

NEPactive 支持灵活的作业提交系统，可以使用 SLURM、PBS、SGE 等调度系统提交作业，也可以直接执行。

## 快速开始

### 1. 配置作业提交系统

在 `in.yaml` 中添加 `job_system` 配置：

```yaml
job_system:
  mode: "local"          # 使用本地调度系统
  scheduler: "slurm"     # 调度系统类型
  check_interval: 30     # 作业状态检查间隔（秒）

  # GPU 任务脚本头
  gpu_header:
    - "#!/bin/bash"
    - "#SBATCH --job-name=nepactive-gpu"
    - "#SBATCH --partition=gpu"
    - "#SBATCH --gres=gpu:1"
    - "#SBATCH --time=24:00:00"
    - ""
    - "module load cuda/11.8"
    - "conda activate nepactive"
    - "export PATH=/path/to/gpumd/bin:$PATH"

  # CPU 任务脚本头
  cpu_header:
    - "#!/bin/bash"
    - "#SBATCH --job-name=nepactive-cpu"
    - "#SBATCH --partition=cpu"
    - "#SBATCH --ntasks=4"
    - "#SBATCH --time=12:00:00"
    - ""
    - "module load python/3.9"
```

### 2. 运行

```bash
nepactive
```

程序会自动：
1. 生成作业脚本（包含你提供的 header）
2. 使用 sbatch/qsub 提交作业
3. 监控作业状态
4. 等待作业完成后继续下一步

## 支持的调度系统

### SLURM

```yaml
job_system:
  scheduler: "slurm"

  gpu_header:
    - "#!/bin/bash"
    - "#SBATCH --job-name=my-job"
    - "#SBATCH --partition=gpu"
    - "#SBATCH --nodes=1"
    - "#SBATCH --ntasks-per-node=1"
    - "#SBATCH --gres=gpu:1"
    - "#SBATCH --time=24:00:00"
    - "#SBATCH --output=job_%j.out"
    - "#SBATCH --error=job_%j.err"
    - ""
    - "# 你的环境设置"
    - "module load cuda"
    - "source activate myenv"
```

**常用 SLURM 指令：**
- `--partition`: 分区名称
- `--nodes`: 节点数
- `--ntasks`: 任务数
- `--gres=gpu:N`: 请求 N 个 GPU
- `--time`: 最大运行时间
- `--mem`: 内存限制

### PBS/Torque

```yaml
job_system:
  scheduler: "pbs"  # 或 "torque"

  gpu_header:
    - "#!/bin/bash"
    - "#PBS -N my-job"
    - "#PBS -q gpu"
    - "#PBS -l nodes=1:ppn=4:gpus=1"
    - "#PBS -l walltime=24:00:00"
    - "#PBS -o job_${PBS_JOBID}.out"
    - "#PBS -e job_${PBS_JOBID}.err"
    - "#PBS -V"
    - ""
    - "cd $PBS_O_WORKDIR"
    - ""
    - "# 你的环境设置"
    - "module load cuda"
    - "source activate myenv"
```

**常用 PBS 指令：**
- `-N`: 作业名称
- `-q`: 队列名称
- `-l nodes=N:ppn=M:gpus=K`: N 个节点，每节点 M 个核心，K 个 GPU
- `-l walltime=HH:MM:SS`: 最大运行时间
- `-V`: 导出当前环境变量

### SGE

```yaml
job_system:
  scheduler: "sge"

  gpu_header:
    - "#!/bin/bash"
    - "#$ -N my-job"
    - "#$ -q gpu.q"
    - "#$ -pe smp 4"
    - "#$ -l gpu=1"
    - "#$ -l h_rt=24:00:00"
    - "#$ -o job_$JOB_ID.out"
    - "#$ -e job_$JOB_ID.err"
    - "#$ -cwd"
    - "#$ -V"
    - ""
    - "# 你的环境设置"
    - "module load cuda"
    - "source activate myenv"
```

**常用 SGE 指令：**
- `-N`: 作业名称
- `-q`: 队列名称
- `-pe smp N`: 请求 N 个核心
- `-l gpu=N`: 请求 N 个 GPU
- `-l h_rt=HH:MM:SS`: 最大运行时间
- `-cwd`: 在当前目录执行

### 直接执行（不使用调度系统）

```yaml
job_system:
  mode: "direct"
  scheduler: "direct"

  gpu_header:
    - "#!/bin/bash"
    - ""
    - "# 设置环境"
    - "source activate myenv"
    - "export PATH=/path/to/gpumd/bin:$PATH"
    - "export CUDA_VISIBLE_DEVICES=0"
```

## 配置参数说明

### job_system

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `mode` | 作业提交模式 | "local", "direct" | "local" |
| `scheduler` | 调度系统类型 | "slurm", "pbs", "torque", "sge", "direct" | "slurm" |
| `check_interval` | 作业状态检查间隔（秒） | 整数 | 30 |
| `gpu_header` | GPU 任务脚本头 | 字符串列表 | - |
| `cpu_header` | CPU 任务脚本头 | 字符串列表 | - |

### gpu_header 和 cpu_header

这两个参数定义了作业脚本的头部，包括：
1. **调度系统指令**：如 `#SBATCH`, `#PBS`, `#$` 等
2. **环境设置**：加载模块、激活虚拟环境等
3. **路径设置**：设置可执行文件路径

**区别：**
- `gpu_header`: 用于需要 GPU 的任务（NEP 训练、GPUMD 模拟）
- `cpu_header`: 用于 CPU 任务（数据处理、分析等）

## 使用场景

### 场景 1: 单机多 GPU

```yaml
job_system:
  mode: "direct"
  scheduler: "direct"

  gpu_header:
    - "#!/bin/bash"
    - "source activate nepactive"
    - "export PATH=/usr/local/gpumd/bin:$PATH"

gpu_available: [0, 1, 2, 3]
task_per_gpu: 1
```

直接在本地执行，使用 4 个 GPU。

### 场景 2: SLURM 集群

```yaml
job_system:
  mode: "local"
  scheduler: "slurm"
  check_interval: 60

  gpu_header:
    - "#!/bin/bash"
    - "#SBATCH --job-name=nep-train"
    - "#SBATCH --partition=gpu"
    - "#SBATCH --gres=gpu:1"
    - "#SBATCH --time=48:00:00"
    - "#SBATCH --mem=32G"
    - ""
    - "module purge"
    - "module load cuda/11.8 gcc/9.3.0"
    - "source /home/user/miniconda3/bin/activate nepactive"
    - "export PATH=/home/user/gpumd/bin:$PATH"

gpu_available: [0, 1, 2, 3]
task_per_gpu: 1
```

每个 GPU 任务会提交一个独立的 SLURM 作业。

### 场景 3: PBS 集群

```yaml
job_system:
  mode: "local"
  scheduler: "pbs"
  check_interval: 60

  gpu_header:
    - "#!/bin/bash"
    - "#PBS -N nep-train"
    - "#PBS -q gpu"
    - "#PBS -l nodes=1:ppn=4:gpus=1"
    - "#PBS -l walltime=48:00:00"
    - "#PBS -l mem=32gb"
    - "#PBS -V"
    - ""
    - "cd $PBS_O_WORKDIR"
    - "module load cuda/11.8"
    - "source activate nepactive"

gpu_available: [0, 1, 2, 3]
```

## 环境设置最佳实践

### 1. 模块加载

```yaml
gpu_header:
  - "#!/bin/bash"
  - "#SBATCH ..."
  - ""
  - "# 清理环境"
  - "module purge"
  - ""
  - "# 加载必要模块"
  - "module load cuda/11.8"
  - "module load gcc/9.3.0"
  - "module load python/3.9"
```

### 2. Conda 环境

```yaml
gpu_header:
  - "#!/bin/bash"
  - "#SBATCH ..."
  - ""
  - "# 初始化 Conda"
  - "source /path/to/miniconda3/bin/activate"
  - "conda activate nepactive"
  - ""
  - "# 或使用 module"
  - "# module load anaconda3"
  - "# conda activate nepactive"
```

### 3. 虚拟环境

```yaml
gpu_header:
  - "#!/bin/bash"
  - "#SBATCH ..."
  - ""
  - "# 激活虚拟环境"
  - "source /path/to/venv/bin/activate"
```

### 4. 路径设置

```yaml
gpu_header:
  - "#!/bin/bash"
  - "#SBATCH ..."
  - ""
  - "# 设置 GPUMD 路径"
  - "export PATH=/path/to/gpumd/bin:$PATH"
  - ""
  - "# 设置库路径"
  - "export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH"
  - ""
  - "# 设置 CUDA"
  - "export CUDA_HOME=/usr/local/cuda"
  - "export PATH=$CUDA_HOME/bin:$PATH"
```

## 故障排除

### 问题 1: 作业提交失败

**错误信息：**
```
Failed to submit SLURM job: sbatch: command not found
```

**解决方案：**
- 检查调度系统是否安装
- 确认 `scheduler` 参数设置正确
- 尝试使用 `mode: "direct"` 直接执行

### 问题 2: 环境变量未加载

**症状：** 作业运行时找不到 GPUMD 或其他命令

**解决方案：**
1. 在 `gpu_header` 中添加完整的环境设置
2. 使用绝对路径
3. 添加 `-V` (PBS) 或类似选项导出环境变量

### 问题 3: 作业一直处于 pending 状态

**可能原因：**
- 资源不足（GPU/内存/节点）
- 队列限制
- 优先级问题

**解决方案：**
1. 检查集群资源使用情况
2. 调整资源请求（减少 GPU 数量、内存等）
3. 更改队列或分区

### 问题 4: 作业运行但没有输出

**解决方案：**
1. 检查输出文件路径设置
2. 确认工作目录正确
3. 添加调试信息：
```yaml
gpu_header:
  - "#!/bin/bash"
  - "#SBATCH ..."
  - ""
  - "# 调试信息"
  - "echo 'Job started at:' $(date)"
  - "echo 'Working directory:' $(pwd)"
  - "echo 'GPU info:' $(nvidia-smi)"
  - ""
  - "# 你的命令"
```

## 高级用法

### 自定义提交命令

如果需要自定义提交命令（例如添加额外参数）：

```yaml
job_system:
  scheduler: "slurm"
  submit_command: "sbatch --account=myproject"
  status_command: "squeue"
  cancel_command: "scancel"
```

### 作业依赖

对于需要作业依赖的场景，可以在 header 中添加：

```yaml
gpu_header:
  - "#!/bin/bash"
  - "#SBATCH --dependency=afterok:12345"  # 等待作业 12345 完成
  - "#SBATCH ..."
```

### 数组作业

对于大量相似任务，可以使用数组作业：

```yaml
gpu_header:
  - "#!/bin/bash"
  - "#SBATCH --array=0-9"  # 10 个数组任务
  - "#SBATCH ..."
  - ""
  - "# 使用 $SLURM_ARRAY_TASK_ID"
```

## 完整配置示例

```yaml
# NEPactive 完整配置示例

project_name: "VN5_Training"
yaml_synchro: true

# 作业提交系统
job_system:
  mode: "local"
  scheduler: "slurm"
  check_interval: 60

  gpu_header:
    - "#!/bin/bash"
    - "#SBATCH --job-name=nepactive-gpu"
    - "#SBATCH --partition=gpu"
    - "#SBATCH --nodes=1"
    - "#SBATCH --ntasks-per-node=1"
    - "#SBATCH --gres=gpu:1"
    - "#SBATCH --time=48:00:00"
    - "#SBATCH --mem=32G"
    - "#SBATCH --output=job_%j.out"
    - "#SBATCH --error=job_%j.err"
    - ""
    - "# 环境设置"
    - "module purge"
    - "module load cuda/11.8"
    - "module load gcc/9.3.0"
    - ""
    - "source /home/user/miniconda3/bin/activate"
    - "conda activate nepactive"
    - ""
    - "export PATH=/home/user/gpumd/bin:$PATH"
    - "export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS"

  cpu_header:
    - "#!/bin/bash"
    - "#SBATCH --job-name=nepactive-cpu"
    - "#SBATCH --partition=cpu"
    - "#SBATCH --ntasks=8"
    - "#SBATCH --time=12:00:00"
    - "#SBATCH --mem=16G"
    - ""
    - "module load python/3.9"
    - "source activate nepactive"

# NEP 训练参数
nep_in_header: "type 2 V N"
accuracy: 1
pot_number: 4
train_steps: 5000

# GPU 配置
gpu_available: [0, 1, 2, 3]
task_per_gpu: 1

# 其他参数...
```

---

**提示：** 首次使用时，建议先用 `mode: "direct"` 测试，确认环境配置正确后再切换到调度系统。
