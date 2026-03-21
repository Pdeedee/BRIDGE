__version__ = "0.0.1"

import logging
import os
import yaml


ROOT_PATH = __path__[0]
NAME = "nepactive"
SHORT_CMD = "nepactive"
dlog = logging.getLogger(__name__)
dlog.setLevel(logging.INFO)
dlogf = logging.FileHandler(os.getcwd() + os.sep + SHORT_CMD + ".log", delay=True)
dlogf_formatter = logging.Formatter("%(asctime)s - %(levelname)s : %(message)s")
# dlogf_formatter = logging.Formatter('%(asctime)s - %(name)s - [%(filename)s:%(funcName)s - %(lineno)d ] - %(levelname)s  %(message)s')
dlogf.setFormatter(dlogf_formatter)
dlog.addHandler(dlogf)
logging.basicConfig(level=logging.WARNING)

def _migrate_stable_config(data: dict) -> dict:
    """将旧 stable: 配置自动拆分为 init: + shock:"""
    if "stable" in data and "init" not in data and "shock" not in data:
        s = data["stable"]
        data["init"] = {
            "structure": s.get("structure", "POSCAR"),
            "struc_num": s.get("struc_num", 1),
            "pressure": s.get("pressure", [20, 40, 60, 80]),
            "temperature": s.get("temperature", [3000]),
            "steps": s.get("steps", 40000),
            "time_step": s.get("time_step", 0.2),
            "dump_freq": s.get("dump_freq", 10),
            "tau_t": s.get("tau_t", 100),
            "tau_p": s.get("tau_p", 2000),
            "pmode": s.get("pmode", "iso"),
            "tfreq": s.get("tfreq"),
            "pfreq": s.get("pfreq"),
            "original_make": s.get("original_make", True),
        }
        data["shock"] = {
            "structure": s.get("structure", "POSCAR"),
            "pressure_list": s.get("shock_pressure_list", [20, 25, 30, 35, 40, 45, 50, 55]),
            "steps": s.get("shock_steps", 600000),
            "time_step": s.get("time_step", 0.2),
            "dump_freq": s.get("dump_freq", 10),
            "analyze_range": s.get("analyze_range", [0.5, 1]),
            "pot": s.get("shock_pot", "nep"),
            "tau_t": s.get("tau_t", 100),
            "tau_p": s.get("tau_p", 2000),
            "pmode": s.get("pmode", "iso"),
            "original_make": s.get("original_make", False),
        }
        dlog.info("Migrated old 'stable:' config to 'init:' + 'shock:'")
    return data


def parse_yaml(file):
    with open(file, encoding='utf-8') as f:
        data = yaml.safe_load(f)
    data = _migrate_stable_config(data)
    return data
