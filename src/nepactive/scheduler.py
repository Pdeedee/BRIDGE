"""
作业调度器模块 - 本地作业提交管理
支持 SLURM, PBS, SGE 等调度系统
"""

import os
import subprocess
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from nepactive import dlog


class JobScheduler(ABC):
    """作业调度器基类"""

    def __init__(self, config: Dict):
        self.config = config
        self.header_script = self._load_header(config)

    def _load_header(self, config: Dict) -> List[str]:
        """
        加载脚本头部
        支持两种方式：
        1. header: 多行字符串（推荐，使用 YAML 的 | 符号）
        2. header: 字符串列表（兼容旧格式）
        """
        if "header" in config:
            header = config["header"]
            # 如果是字符串，按行分割
            if isinstance(header, str):
                return header.splitlines()
            # 如果是列表，直接返回
            elif isinstance(header, list):
                return header
            else:
                dlog.warning(f"Invalid header format: {type(header)}, using empty header")
                return []

        # 兼容旧的 header_script 参数
        if "header_script" in config:
            header_script = config["header_script"]
            if isinstance(header_script, str):
                return header_script.splitlines()
            elif isinstance(header_script, list):
                return header_script

        return []

    @abstractmethod
    def submit_job(self, script_path: str, work_dir: str) -> str:
        """
        提交作业

        Args:
            script_path: 脚本文件路径
            work_dir: 工作目录

        Returns:
            job_id: 作业ID
        """
        pass

    @abstractmethod
    def check_job_status(self, job_id: str) -> str:
        """
        检查作业状态

        Args:
            job_id: 作业ID

        Returns:
            status: 'running', 'completed', 'failed', 'pending'
        """
        pass

    @abstractmethod
    def cancel_job(self, job_id: str):
        """取消作业"""
        pass

    def write_script(self, script_path: str, commands: List[str], work_dir: str = None):
        """
        写入作业脚本

        Args:
            script_path: 脚本文件路径
            commands: 要执行的命令列表
            work_dir: 工作目录
        """
        with open(script_path, 'w') as f:
            # 写入 header
            for line in self.header_script:
                f.write(line + '\n')

            f.write('\n')

            # 切换到工作目录
            if work_dir:
                f.write(f'cd {work_dir}\n\n')

            # 写入命令
            for cmd in commands:
                f.write(cmd + '\n')

        # 添加执行权限
        os.chmod(script_path, 0o755)
        dlog.info(f"Created job script: {script_path}")


class SLURMScheduler(JobScheduler):
    """SLURM 调度器"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.submit_command = config.get("submit_command", "sbatch")
        self.status_command = config.get("status_command", "squeue")
        self.cancel_command = config.get("cancel_command", "scancel")

    def submit_job(self, script_path: str, work_dir: str) -> str:
        """提交 SLURM 作业"""
        cmd = [self.submit_command, script_path]

        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=True
            )

            # 解析作业ID: "Submitted batch job 12345"
            output = result.stdout.strip()
            job_id = output.split()[-1]
            dlog.info(f"Submitted SLURM job: {job_id}")
            return job_id

        except subprocess.CalledProcessError as e:
            dlog.error(f"Failed to submit SLURM job: {e.stderr}")
            raise

    def check_job_status(self, job_id: str) -> str:
        """检查 SLURM 作业状态"""
        cmd = [self.status_command, "-j", job_id, "-h", "-o", "%T"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            status = result.stdout.strip()

            # SLURM 状态映射
            status_map = {
                'PENDING': 'pending',
                'RUNNING': 'running',
                'COMPLETED': 'completed',
                'FAILED': 'failed',
                'CANCELLED': 'failed',
                'TIMEOUT': 'failed',
                'NODE_FAIL': 'failed',
            }

            return status_map.get(status, 'unknown')

        except subprocess.CalledProcessError:
            # 作业不存在，可能已完成
            return 'completed'

    def cancel_job(self, job_id: str):
        """取消 SLURM 作业"""
        cmd = [self.cancel_command, job_id]

        try:
            subprocess.run(cmd, check=True)
            dlog.info(f"Cancelled SLURM job: {job_id}")
        except subprocess.CalledProcessError as e:
            dlog.error(f"Failed to cancel SLURM job {job_id}: {e}")


class PBSScheduler(JobScheduler):
    """PBS/Torque 调度器"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.submit_command = config.get("submit_command", "qsub")
        self.status_command = config.get("status_command", "qstat")
        self.cancel_command = config.get("cancel_command", "qdel")

    def submit_job(self, script_path: str, work_dir: str) -> str:
        """提交 PBS 作业"""
        cmd = [self.submit_command, script_path]

        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=True
            )

            # PBS 返回作业ID
            job_id = result.stdout.strip()
            dlog.info(f"Submitted PBS job: {job_id}")
            return job_id

        except subprocess.CalledProcessError as e:
            dlog.error(f"Failed to submit PBS job: {e.stderr}")
            raise

    def check_job_status(self, job_id: str) -> str:
        """检查 PBS 作业状态"""
        cmd = [self.status_command, job_id]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # 解析 qstat 输出
            lines = result.stdout.strip().split('\n')
            if len(lines) > 2:
                status_line = lines[2]
                status = status_line.split()[4]

                # PBS 状态映射
                status_map = {
                    'Q': 'pending',
                    'R': 'running',
                    'C': 'completed',
                    'E': 'failed',
                    'H': 'pending',
                }

                return status_map.get(status, 'unknown')

            return 'completed'

        except subprocess.CalledProcessError:
            return 'completed'

    def cancel_job(self, job_id: str):
        """取消 PBS 作业"""
        cmd = [self.cancel_command, job_id]

        try:
            subprocess.run(cmd, check=True)
            dlog.info(f"Cancelled PBS job: {job_id}")
        except subprocess.CalledProcessError as e:
            dlog.error(f"Failed to cancel PBS job {job_id}: {e}")


class SGEScheduler(JobScheduler):
    """SGE (Sun Grid Engine) 调度器"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.submit_command = config.get("submit_command", "qsub")
        self.status_command = config.get("status_command", "qstat")
        self.cancel_command = config.get("cancel_command", "qdel")

    def submit_job(self, script_path: str, work_dir: str) -> str:
        """提交 SGE 作业"""
        cmd = [self.submit_command, script_path]

        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=True
            )

            # 解析作业ID: "Your job 12345 ..."
            output = result.stdout.strip()
            job_id = output.split()[2]
            dlog.info(f"Submitted SGE job: {job_id}")
            return job_id

        except subprocess.CalledProcessError as e:
            dlog.error(f"Failed to submit SGE job: {e.stderr}")
            raise

    def check_job_status(self, job_id: str) -> str:
        """检查 SGE 作业状态"""
        cmd = [self.status_command, "-j", job_id]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # 解析 qstat 输出
            lines = result.stdout.strip().split('\n')
            if len(lines) > 2:
                status_line = lines[2]
                status = status_line.split()[4]

                # SGE 状态映射
                status_map = {
                    'qw': 'pending',
                    'r': 'running',
                    't': 'running',
                    'Eqw': 'failed',
                }

                return status_map.get(status, 'unknown')

            return 'completed'

        except subprocess.CalledProcessError:
            return 'completed'

    def cancel_job(self, job_id: str):
        """取消 SGE 作业"""
        cmd = [self.cancel_command, job_id]

        try:
            subprocess.run(cmd, check=True)
            dlog.info(f"Cancelled SGE job: {job_id}")
        except subprocess.CalledProcessError as e:
            dlog.error(f"Failed to cancel SGE job {job_id}: {e}")


class DirectScheduler(JobScheduler):
    """直接执行调度器（不使用作业调度系统）"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.processes = {}

    def submit_job(self, script_path: str, work_dir: str) -> str:
        """直接执行脚本"""
        try:
            process = subprocess.Popen(
                ['/bin/bash', script_path],
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            job_id = str(process.pid)
            self.processes[job_id] = process
            dlog.info(f"Started direct execution: PID {job_id}")
            return job_id

        except Exception as e:
            dlog.error(f"Failed to execute script: {e}")
            raise

    def check_job_status(self, job_id: str) -> str:
        """检查进程状态"""
        if job_id in self.processes:
            process = self.processes[job_id]
            if process.poll() is None:
                return 'running'
            elif process.returncode == 0:
                return 'completed'
            else:
                return 'failed'

        return 'completed'

    def cancel_job(self, job_id: str):
        """终止进程"""
        if job_id in self.processes:
            process = self.processes[job_id]
            process.terminate()
            dlog.info(f"Terminated process: PID {job_id}")


def create_scheduler(config: Dict) -> JobScheduler:
    """
    创建调度器实例

    Args:
        config: 调度器配置

    Returns:
        scheduler: 调度器实例
    """
    scheduler_type = config.get("scheduler", "slurm").lower()

    scheduler_map = {
        "slurm": SLURMScheduler,
        "pbs": PBSScheduler,
        "torque": PBSScheduler,
        "sge": SGEScheduler,
        "direct": DirectScheduler,
    }

    if scheduler_type not in scheduler_map:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    scheduler_class = scheduler_map[scheduler_type]
    return scheduler_class(config)


class JobManager:
    """作业管理器 - 管理多个作业的提交和监控"""

    def __init__(self, scheduler: JobScheduler):
        self.scheduler = scheduler
        self.jobs = {}  # {job_id: {"script": path, "work_dir": dir, "status": status}}

    def submit(self, script_path: str, work_dir: str, job_name: str = None) -> str:
        """提交作业"""
        job_id = self.scheduler.submit_job(script_path, work_dir)

        self.jobs[job_id] = {
            "script": script_path,
            "work_dir": work_dir,
            "name": job_name or os.path.basename(script_path),
            "status": "pending",
            "submit_time": time.time()
        }

        return job_id

    def wait_for_jobs(self, job_ids: List[str], check_interval: int = 30):
        """等待作业完成"""
        pending_jobs = set(job_ids)

        while pending_jobs:
            for job_id in list(pending_jobs):
                status = self.scheduler.check_job_status(job_id)
                self.jobs[job_id]["status"] = status

                if status in ['completed', 'failed']:
                    pending_jobs.remove(job_id)
                    job_name = self.jobs[job_id]["name"]

                    if status == 'completed':
                        dlog.info(f"Job {job_name} ({job_id}) completed")
                    else:
                        dlog.error(f"Job {job_name} ({job_id}) failed")

            if pending_jobs:
                dlog.info(f"Waiting for {len(pending_jobs)} jobs to complete...")
                time.sleep(check_interval)

    def cancel_all(self):
        """取消所有作业"""
        for job_id in self.jobs:
            if self.jobs[job_id]["status"] in ['pending', 'running']:
                self.scheduler.cancel_job(job_id)

    def get_status_summary(self) -> Dict[str, int]:
        """获取作业状态统计"""
        summary = {
            'pending': 0,
            'running': 0,
            'completed': 0,
            'failed': 0,
        }

        for job_info in self.jobs.values():
            status = job_info["status"]
            if status in summary:
                summary[status] += 1

        return summary
