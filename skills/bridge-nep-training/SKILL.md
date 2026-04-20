---
name: bridge-nep-training
description: Use this skill when working in the BRIDGE/nepactive repository on NEP potential training, active-learning training loops, in.yaml training configuration, nep.in generation, training dataset assembly, or training-stage debugging. This skill is for training and dataset flow only; do not use it for shock velocity, HOD, or other blast-speed testing unless the user explicitly asks for those paths.
---

# BRIDGE NEP Training

This skill is for the BRIDGE repository's NEP training path. Use it when the task is about:

- adjusting `in.yaml` for training
- understanding or changing `00.nep` behavior
- debugging why `nep` training failed
- tracing how `train.xyz` and `test.xyz` are assembled
- explaining or modifying the active-learning loop around training

Do not expand into `shock`, HOD, or OB workflows unless the user explicitly asks for them.

## First pass

1. Confirm the task is in this repo and read only the training-relevant files:
   - [README.md](../../README.md)
   - [src/nepactive/main.py](../../src/nepactive/main.py)
   - [src/nepactive/train.py](../../src/nepactive/train.py)
   - [src/nepactive/template.py](../../src/nepactive/template.py)
2. Read [references/training-reference.md](references/training-reference.md).
3. Inspect the user's current project state before changing anything:
   - `in.yaml`
   - `record.nep`
   - `init/iter_train.xyz`, `init/iter_test.xyz`
   - latest `iter.*/00.nep`
   - latest `iter.*/02.label`

## Installation and entry

- Preferred editable install is `uv pip install -e .` from the repo root.
- After installation, prefer invoking the workflow through the installed CLI: `nepactive`.
- Use `python -m nepactive.main` only as a fallback when the CLI entry point is unavailable or when debugging packaging issues.

## Repo-specific mental model

- Main entry is `nepactive`, which calls `Nepactive.run()` from [src/nepactive/main.py](../../src/nepactive/main.py).
- If `record.nep` does not exist, the code first builds the initial dataset under `init/`.
- The iterative loop is fixed in [src/nepactive/train.py](../../src/nepactive/train.py): `00.nep -> 01.sampling/01.gpumd -> 02.label`.
- Training work lives in `iter.XXXXXX/00.nep/`.
- Each potential is a separate `task.XXXXXX` directory under `00.nep`.

## Training workflow rules

### Dataset assembly

- Initial train/test data come from `init/iter_train.xyz` and `init/iter_test.xyz`.
- Extra initial labeled data may be appended from `init.train_data` and `init.test_data`.
- For later iterations, training data are merged from prior `iter.XXXXXX/02.label/iter_train.xyz` and `iter_test.xyz`.
- `make_nep_train()` writes merged files to `iter.XXXXXX/00.nep/dataset/train.xyz` and `dataset/test.xyz`.

### `nep.in` generation

- If the project root has `nep.in`, or `in.yaml` sets `nep_template`, that file is used as the template.
- Even with a custom template, `generation` is always overwritten by the current round's `ini_train_steps` or `train_steps`.
- Without a custom template, the built-in template from [src/nepactive/template.py](../../src/nepactive/template.py) is used.

### Element order and inheritance

- When `pot_inherit: true`, treat `nep_in_header` as effectively mandatory for stable continuation.
- If `nep_in_header` is omitted, the code auto-infers it from `structure_files`.
- When inheritance is enabled, an auto-inferred header change between iterations raises an error instead of silently continuing.
- If the user reports restart mismatch, first compare current and previous `task.XXXXXX/nep.in`.

### Parallel training

- `pot_number` controls how many `nep` tasks are launched each iteration.
- `gpu_available` must be present and non-empty.
- Training launches `nep` directly, one process per task, with `CUDA_VISIBLE_DEVICES` assigned round-robin from `gpu_available`.
- Each task logs to `iter.XXXXXX/00.nep/task.XXXXXX/log`.

## How to debug training tasks

Check these in order:

1. Configuration sanity in `in.yaml`.
2. Whether the expected dataset files actually exist.
3. Whether `iter.XXXXXX/00.nep/dataset/train.xyz` and `dataset/test.xyz` were built.
4. Whether each `task.XXXXXX` has `train.xyz`, `test.xyz`, `nep.in`, and optionally `nep.restart`.
5. The failing task's `log`.
6. Whether the issue is really training, or an upstream init / sampling / label failure surfacing at training time.

## Modification guidance

- For changes to training file generation, inspect `_build_nep_in_content`, `_resolve_nep_in_header`, and `run_nep_train`.
- For changes to dataset composition, inspect `_resolve_init_dataset_files` and `make_nep_train`.
- For changes to loop ordering or rerun behavior, inspect `run`, `make_loop_train`, and `record.nep` handling.
- Preserve the repository's current separation: training in `00.nep`, sampling in `01.sampling` or legacy `01.gpumd`, labeling in `02.label`.

## Output style

- When answering, tie explanations to concrete repo paths and current iteration directories.
- When diagnosing, name the exact file or directory that should exist next.
- If the user asks for a fix, prefer editing the actual training path instead of suggesting abstract configuration ideas.
