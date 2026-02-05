#!/bin/bash
#SBATCH --job-name=nepactive-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# 加载环境模块
module purge
module load cuda/11.8
module load gcc/9.3.0

# 激活 Python 环境
source /path/to/conda/bin/activate
conda activate nepactive

# 设置 GPUMD 路径
export PATH=/path/to/gpumd/bin:$PATH

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
