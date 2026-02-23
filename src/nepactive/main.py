# -*- coding: utf-8 -*-
from nepactive import dlog, parse_yaml
from nepactive.train import Nepactive
from nepactive.remote import Remotetask, traj2tasks
from nepactive.OB import OB
from nepactive.train_OB import Nepactive_OB
from nepactive.hod import batch_calculate_heat_of_detonation
import logging
import argparse
import yaml
import os


def _reset_tasks(dirs, failed_only=False):
    """清除 task 状态标记。failed_only=True 时只清 task_failed。"""
    from glob import glob as _glob
    targets = ["task_failed"] if failed_only else ["task_finished", "task_failed"]
    count = 0
    for d in dirs:
        d = os.path.abspath(d)
        for pattern in targets:
            for marker in _glob(os.path.join(d, f"task.*/{pattern}")):
                os.remove(marker)
                count += 1
                dlog.info(f"removed {marker}")
    dlog.info(f"reset {count} markers")


def main():
    parser = argparse.ArgumentParser(description="nepactive")
    sub = parser.add_subparsers(dest="command")

    # nepactive remote
    sub.add_parser("remote", help="扫描 in.yaml 中 remote_dirs 指定的目录并远程提交")

    # nepactive mktask [file1 file2 ...] [--frames N]
    p_mktask = sub.add_parser("mktask", help="从结构文件批量生成 VASP task 文件夹")
    p_mktask.add_argument("structure_files", nargs="*",
                          help="xyz/traj 结构文件路径，不指定则去 remote.dirs 各目录下找 candidate.traj")
    p_mktask.add_argument("--frames", type=int, default=0,
                          help="随机抽取的帧数，0 表示全部")

    # nepactive reset [failed]
    p_reset = sub.add_parser("reset", help="重置 task 状态标记")
    p_reset.add_argument("mode", nargs="?", default="all",
                         choices=["all", "failed"],
                         help="all=清除所有状态(默认), failed=只清除 task_failed")

    # nepactive shock / OB / hod
    sub.add_parser("shock", help="shock velocity calculation")
    sub.add_parser("OB", help="OB calculation")
    p_hod = sub.add_parser("hod", help="heat of detonation calculation")
    p_hod.add_argument("--gpu", type=int, default=0, help="GPU ID")

    args = parser.parse_args()
    idata: dict = parse_yaml("in.yaml")

    if args.command == "remote":
        task = Remotetask(idata=idata)
        task.run_submission()

    elif args.command == "mktask":
        incar_settings = idata.get('incar_settings', {})
        files = args.structure_files
        # 按输出目录分组累加编号，避免同目录编号冲突
        dir_offset: dict = {}  # output_dir -> next start_index
        if not files:
            # 无参数：去 remote.dirs 各目录下找 candidate.traj
            rc = idata.get('remote', {})
            raw_dirs = rc.get('dirs', ["."])
            for d in raw_dirs:
                d = os.path.abspath(d)
                traj_path = os.path.join(d, "candidate.traj")
                if os.path.exists(traj_path):
                    dlog.info(f"found {traj_path}")
                    si = dir_offset.get(d, 0)
                    n = traj2tasks(traj_file=traj_path, incar_settings=incar_settings,
                                   frames=args.frames, output_dir=d, start_index=si)
                    dir_offset[d] = si + n
                else:
                    dlog.warning(f"candidate.traj not found in {d}, skip")
        else:
            for f in files:
                f = os.path.abspath(f)
                if not os.path.exists(f):
                    dlog.warning(f"{f} not found, skip")
                    continue
                out_dir = os.path.dirname(f)
                si = dir_offset.get(out_dir, 0)
                n = traj2tasks(traj_file=f, incar_settings=incar_settings,
                               frames=args.frames, output_dir=out_dir, start_index=si)
                dir_offset[out_dir] = si + n

    elif args.command == "reset":
        rc = idata.get('remote', {})
        raw_dirs = rc.get('dirs', ["."])
        dirs = [os.path.abspath(d) for d in raw_dirs]
        _reset_tasks(dirs, failed_only=(args.mode == "failed"))
    elif args.command == "shock":
        task = Nepactive(idata=idata)
        task.shock()

    elif args.command == "OB":
        task = Nepactive_OB(idata=idata)
        task.run()

    elif args.command == "hod":
        base_dir = os.getcwd()
        dlog.info(f"Calculating heat of detonation for the latest iteration in: {base_dir}")
        from glob import glob
        shock_dirs = glob(os.path.join(base_dir, "iter.*/03.shock"))
        if not shock_dirs:
            dlog.error("No shock directories found")
            return
        shock_dirs.sort()
        latest_shock_dir = shock_dirs[-1]
        latest_iter = os.path.dirname(latest_shock_dir)
        dlog.info(f"Processing latest iteration with shock data: {latest_iter}")
        nep_path = os.path.join(latest_iter, "00.nep", "task.000000", "nep.txt")
        if not os.path.exists(nep_path):
            dlog.error(f"NEP file not found: {nep_path}")
            return
        from nepactive.hod import calculate_heat_of_detonation
        job_system = idata.get("job_system", None)
        try:
            Q_release = calculate_heat_of_detonation(latest_shock_dir, nep_path, args.gpu, job_system)
            dlog.info(f"Heat of detonation: {Q_release:.2f} kJ/mol")
        except Exception as e:
            dlog.error(f"Failed to calculate heat of detonation: {e}")
            import traceback
            traceback.print_exc()

    else:
        # 默认：主动学习流程
        task = Nepactive(idata=idata)
        task.run()

if __name__ == "__main__":
    main()
