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

def parse_yaml(file):
    with open(file, encoding='utf-8') as f:
        data = yaml.safe_load(f)
        # data = yaml.safe_load("in.yaml")
    # if os.path.exists(f"{os.path.dirname(file)}/nostrucnum"):
    #     data["stable"]["struc_num"] = 0
    #     data["structure_files"] = ["POSCAR"]
    #     data["model_devi_general"][0]["structure_id"] = [[0]]
    # else:
    #     data["stable"]["struc_num"] = 1
    #     data["structure_files"] = ["POSCAR","init/struc.000/structure/POSCAR"]
    #     data["model_devi_general"][0]["structure_id"] = [[0,1]]
    return data