# -*- coding: utf-8 -*-
#ase 读取 traj文件，在当前目录下创建名为task.{number}的文件夹，保存每一帧到其中
from ase.io import read,write
import shlex
import os
import shutil
from nepactive import dlog
import math
import paramiko
import subprocess
from typing import TYPE_CHECKING, Callable, Optional, Type, Union, Tuple, List, Any
import tarfile
import time
import socket
from glob import glob
import warnings

import pathlib
import uuid
import json
from enum import IntEnum
from hashlib import sha1
import random
import numpy as np

from pymatgen.io.vasp.sets import MPStaticSet
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

def traj2tasks(traj_file: str, incar_settings: dict = None, frames: int = 0,
               output_dir: str = None, start_index: int = 0) -> int:
    """
    读取轨迹文件，为每一帧生成完整的 VASP 输入（POSCAR + INCAR + POTCAR）。
    使用 pymatgen MPStaticSet 自动生成，通过 incar_settings 覆盖参数。
    默认关闭自旋 (ISPIN=1)。

    Parameters
    ----------
    output_dir : str, optional
        task 文件夹的输出目录，默认为 traj_file 所在目录。
    start_index : int
        task 编号起始值，用于同目录多文件时避免编号冲突。

    Returns
    -------
    int
        本次生成的 task 数量。
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(traj_file)) or "."
    os.makedirs(output_dir, exist_ok=True)

    traj = read(traj_file, index=':')
    if frames == 0:
        frames = len(traj)
    index = random.sample(range(len(traj)), frames)
    index.sort()
    index = np.array(index, dtype="int32")
    sorted_traj = [traj[i] for i in index]
    np.savetxt(os.path.join(output_dir, "index.txt"), index)

    # 默认关自旋
    settings = {"ISPIN": 1}
    if incar_settings:
        settings.update(incar_settings)

    for i, frame in enumerate(sorted_traj):
        folder_name = os.path.join(output_dir, f'task.{start_index + i:06d}')
        os.makedirs(folder_name, exist_ok=True)
        # 全部由pymatgen生成，保证POSCAR/POTCAR元素顺序一致
        struct = AseAtomsAdaptor.get_structure(frame)
        # pymatgen按元素排序，同种元素聚在一起
        struct = struct.get_sorted_structure()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vis = MPStaticSet(struct, user_incar_settings=settings)
            vis.incar.write_file(os.path.join(folder_name, "INCAR"))
            vis.poscar.write_file(os.path.join(folder_name, "POSCAR"))
            vis.potcar.write_file(os.path.join(folder_name, "POTCAR"))
        dlog.info(f"prepared {folder_name}")
    return len(sorted_traj)

class JobStatus(IntEnum):
    unsubmitted = 1
    waiting = 2
    running = 3
    terminated = 4
    finished = 5
    completing = 6
    unknown = 100

def rsync(
        from_file: str,
        to_file: str,
        port: int = 22,
        additional_args: Optional[List[str]] = None,
        key_filename: Optional[str] = None,
        timeout: Union[int, float] = 10,
    ):
        """Call rsync to transfer files."""
        ssh_cmd = [
            "ssh",
            "-o", "ConnectTimeout=" + str(timeout),
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-p", str(port),
            "-q",
        ]
        if key_filename is not None:
            ssh_cmd.extend(["-i", key_filename])
        cmd = [
            "rsync", "-az",
            "-e", " ".join(ssh_cmd),
            "-q",
            from_file,
            to_file,
        ]
        if additional_args:
            cmd.extend(additional_args)
        ret, out, err = run_cmd_with_all_output(cmd, shell=False)
        if ret != 0:
            raise RuntimeError(f"Failed to run {cmd}: {err}")

def run_cmd_with_all_output(cmd, shell=True):
        with subprocess.Popen(
            cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ) as proc:
            out, err = proc.communicate()
            ret = proc.returncode
        return (ret, out, err)

class RetrySignal(Exception):
    """Exception to give a signal to retry the function."""

def retry(
    max_retry: int = 3,
    sleep: Union[int, float] = 60,
    catch_exception: Type[BaseException] = RetrySignal,
) -> Callable:
    """Retry the function until it succeeds or fails for certain times."""
    def decorator(func):
        assert max_retry > 0, "max_retry must be greater than 0"
        def wrapper(*args, **kwargs):
            current_retry = 0
            errors = []
            while max_retry is None or current_retry < max_retry:
                try:
                    return func(*args, **kwargs)
                except (catch_exception,) as e:
                    errors.append(e)
                    dlog.exception("Failed to run %s: %s", func.__name__, e)
                    dlog.warning("Sleep %s s and retry...", sleep)
                    time.sleep(sleep)
                    current_retry += 1
            else:
                raise RuntimeError(
                    "Failed to run %s for %d times" % (func.__name__, current_retry)
                ) from errors[-1]
        return wrapper
    return decorator

# ---------------------------------------------------------------------------
#  Remotetask — 每个 task 独立上传 / 提交 / 回传
# ---------------------------------------------------------------------------
class Remotetask:
    """每个 task 独立上传、独立 sbatch 提交、完成后立即回传。
    重新运行时自动跳过本地已有 task_finished 标记的 task。
    上传与提交合并在同一循环中，先提交的 task 在后续 task 上传期间即可开始排队/运行。
    """

    def __init__(self, idata: dict, work_dirs: Optional[List[str]] = None):
        """
        Parameters
        ----------
        idata : dict
            从 in.yaml 解析出的完整配置。remote 相关配置从 idata['remote'] 子段落读取。
        work_dirs : list[str], optional
            要扫描 task.* 的本地目录列表。不传时从 idata['remote']['dirs'] 读取，默认 ["."]。
        """
        self.idata: dict = idata
        rc: dict = idata.get('remote', {})
        self.rc = rc

        if work_dirs is not None:
            self.work_dirs = [os.path.abspath(d) for d in work_dirs]
        else:
            raw_dirs = rc.get('dirs', ["."])
            self.work_dirs = [os.path.abspath(d) for d in raw_dirs]

        self.project_name = idata.get('project_name')
        self.username = rc.get('ssh_username')
        self.hostname = rc.get('ssh_hostname')
        self.remotename = f"{self.username}@{self.hostname}"
        self.port = rc.get('ssh_port', 22)
        self.remote_root = rc.get('remote_root')
        self.key_filename = None
        self.ssh: paramiko.SSHClient = None
        self._sftp = None
        self._setup_connection = False
        self.execute_command = rc.get('ssh_execute_command')

        self._setup_ssh()
        self.sftp = self.get_sftp()

    # ---- property ----------------------------------------------------------

    @property
    def remote_dir(self):
        """远程项目目录: {remote_root}/{project_name}"""
        return f"{self.remote_root}/{self.project_name}"

    # ---- SSH 基础 ----------------------------------------------------------

    def _setup_ssh(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        self.key_filename = self.rc.get('ssh_key_filename')
        hostname = self.rc.get('ssh_hostname')
        port = self.rc.get('ssh_port', 22)
        username = self.rc.get('ssh_username')
        dlog.info(f"connecting to {hostname}:{port} as {username}")
        self.ssh.connect(
            hostname=hostname, port=port, username=username,
            timeout=30, compress=True,
            allow_agent=False, look_for_keys=True,
        )
        dlog.info("ssh connection established")
        if self.execute_command is not None:
            self.ssh.exec_command(self.execute_command)
        self._setup_connection = True

    def get_sftp(self):
        if self._sftp is None:
            assert self.ssh is not None
            self.ensure_alive()
            self._sftp = self.ssh.open_sftp()
        return self._sftp

    def ensure_alive(self, max_check=10, sleep_time=10):
        count = 1
        while not self._check_alive():
            if count == max_check:
                raise RuntimeError(
                    "cannot connect ssh after %d failures at interval %d s"
                    % (max_check, sleep_time)
                )
            dlog.info("connection check failed, try to reconnect to " + self.hostname)
            self._setup_ssh()
            count += 1
            time.sleep(sleep_time)

    def _check_alive(self):
        if self.ssh is None:
            return False
        try:
            transport = self.ssh.get_transport()
            assert transport is not None
            transport.send_ignore()
            return True
        except EOFError:
            return False

    def block_call(self, cmd):
        assert self.remote_root is not None
        self.ensure_alive()
        stdin, stdout, stderr = self.ssh.exec_command(
            f"cd {shlex.quote(self.remote_root)} ; " + cmd
        )
        exit_status = stdout.channel.recv_exit_status()
        return exit_status, stdin, stdout, stderr

    def block_checkcall(self, cmd, asynchronously=False):
        if asynchronously:
            cmd = f"nohup {cmd} >/dev/null &"
        exit_status, stdin, stdout, stderr = self.block_call(cmd)
        if exit_status != 0:
            raise RuntimeError(
                "Get error code %d in calling %s. message: %s"
                % (exit_status, cmd, stderr.read().decode("utf-8"))
            )
        return stdin, stdout, stderr

    def _rsync(self, from_f, to_f, send=True, additional_args=None):
        """简化的 rsync 封装，不再自动 mkdir。"""
        key_filename = self.key_filename
        timeout = self.rc.get('ssh_timeout', 10)
        if send:
            to_f = self.remotename + ":" + to_f
        else:
            from_f = self.remotename + ":" + from_f
        rsync(from_f, to_f, port=self.port,
              key_filename=key_filename, timeout=timeout,
              additional_args=additional_args)

    def _reload_config(self):
        """重新读取 in.yaml，刷新 remote 配置（group_numbers 等）。"""
        from nepactive import parse_yaml
        self.idata = parse_yaml("in.yaml")
        self.rc = self.idata.get('remote', {})

    def _scan_dir(self, work_dir: str) -> List[str]:
        """扫描单个目录下的 pending task（跳过 task_finished / task_failed）。"""
        all_tasks = sorted(glob(os.path.join(work_dir, "task.*")))
        all_tasks = [os.path.basename(t) for t in all_tasks]
        pending = []
        n_finished = n_failed = 0
        for t in all_tasks:
            task_dir = os.path.join(work_dir, t)
            if os.path.exists(os.path.join(task_dir, "task_finished")):
                n_finished += 1
            elif os.path.exists(os.path.join(task_dir, "task_failed")):
                n_failed += 1
            else:
                pending.append(t)
        dlog.info(
            f"[{os.path.basename(work_dir)}] total: {len(all_tasks)}, "
            f"finished: {n_finished}, failed: {n_failed}, pending: {len(pending)}"
        )
        return pending

    # ---- task 准备 ----------------------------------------------------------

    def setup(self):
        """扫描 work_dirs 下所有 task.* 文件夹，跳过已有 task_finished 或 task_failed 的。
        返回的 self.tasks 是 (work_dir, task_name) 的列表。
        """
        self.tasks: List[tuple] = []
        for work_dir in self.work_dirs:
            for t in self._scan_dir(work_dir):
                self.tasks.append((work_dir, t))

    def _remote_task_dir(self, work_dir: str, task_name: str) -> str:
        """远程 task 路径，用 work_dir 的 basename 做子目录避免冲突。"""
        dir_tag = os.path.basename(work_dir)
        return f"{self.remote_dir}/{dir_tag}/{task_name}"

    def _generate_sbatch(self, work_dir: str, task_name: str) -> str:
        """为单个 task 生成 sbatch 脚本内容。"""
        header_lines: list = self.rc.get('slurm_header_script', [])
        header = "\n".join(header_lines)
        fp_command = self.rc.get('fp_command')
        remote_task = self._remote_task_dir(work_dir, task_name)
        return (
            f"{header}\n"
            f"cd {remote_task}\n"
            f"{fp_command} 1>>fp.log 2>>fp.log\n"
            f"if test $? -eq 0; then touch task_finished; else exit 1; fi\n"
        )

    def write_remote_file(self, remote_path: str, content: str):
        """通过 SFTP 写入远程文件。"""
        self.ensure_alive()
        remote_path = pathlib.PurePath(remote_path).as_posix()
        with self.sftp.open(remote_path, "w") as fp:
            fp.write(content)

    # ---- 上传 / 提交 / 状态 / 下载 ------------------------------------------

    def _upload_task(self, work_dir: str, task_name: str):
        """rsync 上传单个 task 文件夹 + sbatch 脚本到远程。"""
        local_path = os.path.join(work_dir, task_name) + "/"
        remote_path = self._remote_task_dir(work_dir, task_name)
        self.block_checkcall(f"mkdir -p {remote_path}")
        self._rsync(from_f=local_path, to_f=remote_path + "/", send=True)
        sbatch_content = self._generate_sbatch(work_dir, task_name)
        dir_tag = os.path.basename(work_dir)
        sub_path = f"{self.remote_dir}/{dir_tag}/{task_name}.sub"
        self.write_remote_file(sub_path, sbatch_content)
        dlog.info(f"uploaded {dir_tag}/{task_name}")

    @retry(max_retry=3, sleep=60, catch_exception=RetrySignal)
    def _submit_task(self, work_dir: str, task_name: str) -> str:
        """sbatch 提交单个 task，返回 job_id。"""
        dir_tag = os.path.basename(work_dir)
        sub_file = f"{dir_tag}/{task_name}.sub"
        command = f"cd {shlex.quote(self.remote_dir)} && sbatch {shlex.quote(sub_file)}"
        ret, stdin, stdout, stderr = self.block_call(command)
        if ret != 0:
            err_str = stderr.read().decode("utf-8")
            if "Socket timed out" in err_str or "Unable to contact slurm controller" in err_str or "Unexpected message received" in err_str:
                raise RetrySignal(f"submit error for {task_name}: {err_str}")
            if "Job violates accounting/QOS policy" in err_str or "Slurm temporarily unable to accept job" in err_str:
                return ""
            raise RuntimeError(f"sbatch failed for {task_name}: {err_str}")
        job_id = stdout.readlines()[0].split()[-1].strip()
        dlog.info(f"{task_name} submitted as job {job_id}")
        return job_id

    @retry(max_retry=3, sleep=60, catch_exception=RetrySignal)
    def check_status(self, job_id: str) -> JobStatus:
        """squeue 查询 job 状态。job 不在队列中时返回 terminated（需进一步检查 task_finished）。"""
        if not job_id:
            return JobStatus.unsubmitted
        command = 'squeue -h -o "%.18i %.2t" -j ' + job_id
        ret, stdin, stdout, stderr = self.block_call(command)
        if ret != 0:
            err_str = stderr.read().decode("utf-8")
            if "Invalid job id specified" in err_str:
                return JobStatus.terminated
            if "Socket timed out" in err_str or "Unable to contact slurm controller" in err_str or "Unexpected message received" in err_str:
                raise RetrySignal(f"squeue error: {err_str}")
            raise RuntimeError(f"squeue failed for {job_id}: {err_str}")
        status_line = stdout.read().decode("utf-8").strip()
        if not status_line:
            return JobStatus.terminated
        status_word = status_line.split()[-1]
        if status_word in ("PD", "CF", "S"):
            return JobStatus.waiting
        if status_word == "R":
            return JobStatus.running
        if status_word == "CG":
            return JobStatus.completing
        return JobStatus.terminated

    def _check_remote_finished(self, work_dir: str, task_name: str) -> bool:
        """检查远程 task 目录下是否存在 task_finished 标记。"""
        self.ensure_alive()
        remote_path = self._remote_task_dir(work_dir, task_name) + "/task_finished"
        try:
            self.sftp.stat(remote_path)
            return True
        except OSError:
            return False

    def _download_task(self, work_dir: str, task_name: str):
        """rsync 下载单个 task 结果，并在本地写入 task_finished 标记。"""
        remote_path = self._remote_task_dir(work_dir, task_name) + "/"
        local_path = os.path.join(work_dir, task_name) + "/"
        additional_args = [
            "--exclude=POTCAR", "--exclude=*.xml", "--exclude=XDATCAR",
        ]
        self._rsync(from_f=remote_path, to_f=local_path, send=False,
                     additional_args=additional_args)
        # 确保本地 task_finished 标记存在
        marker = os.path.join(work_dir, task_name, "task_finished")
        if not os.path.exists(marker):
            open(marker, "w").close()
        dlog.info(f"downloaded {task_name}")

    # ---- 主流程 ------------------------------------------------------------

    def run_submission(self):
        """主流程: 按目录逐个处理，每个目录跑完后 reload yaml 拿最新配置和目录列表。
        每次 _fill_queue 前也 reload yaml 以获取最新 group_numbers。
        """
        self._reload_config()
        self.block_checkcall(f"mkdir -p {self.remote_dir}")

        processed_dirs: set = set()
        failed_tasks: list = []

        while True:
            # 从最新 yaml 拿目录列表
            raw_dirs = self.rc.get('dirs', ["."])
            all_dirs = [os.path.abspath(d) for d in raw_dirs]
            # 找到下一个未处理的目录
            next_dir = None
            for d in all_dirs:
                if d not in processed_dirs:
                    next_dir = d
                    break
            if next_dir is None:
                break

            # 扫描该目录
            pending = self._scan_dir(next_dir)
            if not pending:
                dlog.info(f"[{os.path.basename(next_dir)}] nothing to do, skip")
                processed_dirs.add(next_dir)
                self._reload_config()
                continue

            queue = [(next_dir, t) for t in pending]
            active: dict = {}

            def _fill_queue():
                self._reload_config()
                max_c = self.rc.get('group_numbers', 10)
                while queue and len(active) < max_c:
                    key = queue.pop(0)
                    wd, tn = key
                    self._upload_task(wd, tn)
                    job_id = self._submit_task(wd, tn)
                    active[key] = job_id

            _fill_queue()
            dlog.info(
                f"[{os.path.basename(next_dir)}] submitted: {len(active)}, queued: {len(queue)}"
            )

            # 轮询该目录的 task
            while active:
                time.sleep(30)
                done_keys = []
                statuses: dict = {}

                for key, job_id in list(active.items()):
                    wd, tn = key
                    status = self.check_status(job_id)
                    statuses[key] = status

                    if status == JobStatus.terminated:
                        if self._check_remote_finished(wd, tn):
                            self._download_task(wd, tn)
                            done_keys.append(key)
                        else:
                            marker = os.path.join(wd, tn, "task_failed")
                            open(marker, "w").close()
                            failed_tasks.append(f"{wd}/{tn}")
                            dlog.warning(f"{tn} FAILED, marked task_failed")
                            done_keys.append(key)

                for k in done_keys:
                    del active[k]

                if done_keys:
                    _fill_queue()

                n_running = sum(
                    1 for s in statuses.values()
                    if s in (JobStatus.running, JobStatus.completing)
                )
                n_waiting = sum(1 for s in statuses.values() if s == JobStatus.waiting)
                dlog.info(
                    f"[{os.path.basename(next_dir)}] active: {len(active)}, "
                    f"running: {n_running}, waiting: {n_waiting}, queued: {len(queue)}"
                )

            processed_dirs.add(next_dir)
            dlog.info(f"[{os.path.basename(next_dir)}] done")
            # reload yaml 以发现新增目录
            self._reload_config()

        if failed_tasks:
            dlog.warning(f"{len(failed_tasks)} tasks failed:")
            for f in failed_tasks:
                dlog.warning(f"  {f}")
        else:
            dlog.info("all tasks completed successfully")

        clean = self.rc.get('clean', True)
        if clean:
            self.block_checkcall(f"rm -rf {self.remote_dir}")
            dlog.info(f"cleaned remote directory {self.remote_dir}")
