from __future__ import annotations

import pickle
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nepactive.nep_backend import NativeNepCalculator


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: _native_guard_worker.py REQUEST_PKL RESPONSE_PKL", file=sys.stderr)
        return 2

    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    payload = pickle.loads(request_path.read_bytes())

    calculator = NativeNepCalculator(
        model_file=payload["model_file"],
        backend=payload["backend"],
    )
    structures = payload["structures"]
    task = payload["task"]

    if task == "calculate":
        result = calculator.calculate(structures, mean_virial=True)
    elif task == "descriptor":
        result = calculator.get_structures_descriptor(structures)
    else:
        raise ValueError(f"Unsupported native task: {task}")

    response_path.write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
