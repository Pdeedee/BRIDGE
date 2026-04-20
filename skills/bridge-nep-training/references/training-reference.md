# BRIDGE Training Reference

## Entry points

- Preferred install command: `uv pip install -e .`
- Preferred runtime command: `nepactive`
- Fallback source entry: `python -m nepactive.main`
- CLI entry: [src/nepactive/main.py](../../src/nepactive/main.py)
- Main workflow: [src/nepactive/train.py](../../src/nepactive/train.py)
- Built-in `nep.in` template: [src/nepactive/template.py](../../src/nepactive/template.py)
- User-facing config notes: [README.md](../../README.md)

## Fixed stage order

Each iteration in [src/nepactive/train.py](../../src/nepactive/train.py) runs:

1. `make_nep_train`
2. `run_nep_train`
3. `post_nep_train`
4. `make_sampling`
5. `run_model_devi`
6. `post_sampling_run`
7. `make_label_task`
8. `run_label_task`
9. `post_label_task`

Training-only work normally touches:

- `init/`
- `iter.XXXXXX/00.nep/`
- `iter.XXXXXX/02.label/`
- `record.nep`

## Important directories and files

- `init/init.xyz`: raw initial extracted frames
- `init/iter_train.xyz`: initial training split
- `init/iter_test.xyz`: initial test split
- `iter.XXXXXX/00.nep/dataset/train.xyz`: merged train set used for this round
- `iter.XXXXXX/00.nep/dataset/test.xyz`: merged test set used for this round
- `iter.XXXXXX/00.nep/task.XXXXXX/nep.in`: actual input used by `nep`
- `iter.XXXXXX/00.nep/task.XXXXXX/nep.restart`: copied from previous round when inheritance is enabled
- `iter.XXXXXX/00.nep/task.XXXXXX/nep.txt`: trained potential output
- `iter.XXXXXX/00.nep/task.XXXXXX/log`: training log
- `iter.XXXXXX/02.label/iter_train.xyz`: newly labeled train additions
- `iter.XXXXXX/02.label/iter_test.xyz`: newly labeled test additions

## Config keys that matter most for training

- `ini_frames`: number of initial frames to keep
- `training_ratio`: train/test split ratio
- `ini_train_steps`: generation count for iteration 0
- `train_steps`: generation count for later iterations
- `pot_number`: number of parallel NEP models
- `pot_inherit`: whether to copy previous `nep.restart`
- `nep_in_header`: fixed element order for built-in template generation
- `nep_template`: explicit custom template path
- `gpu_available`: GPU list for launching `nep`
- `sampling.init_method`, `sampling.init_descriptor`: initial split method
- `sampling.dataset_method`, `sampling.dataset_descriptor`: later split method
- `init.train_data`, `init.test_data`: extra labeled data to append

## Behavior details worth remembering

- Root-level `nep.in` is treated as a user template even if `nep_template` is not set.
- A custom template does not protect its `generation` line; the code overwrites it.
- `pot_number` is capped at 4 by `make_nep_train()`.
- `gpu_available` missing or empty is a hard error in `run_nep_train()`.
- `sampling.general` is required by the main workflow even when the immediate problem looks like training.
- The repo accepts both `01.sampling` and legacy `01.gpumd` paths in some helper logic.

## Common failure signatures

- `No files found to extract frames from.`:
  initial `init/**/task.*/*.traj` were not produced, so training never had a valid starting dataset.
- `No files found to merge.`:
  `make_nep_train()` found neither `init/iter_train.xyz` nor prior label outputs.
- `gpu_available is missing or empty in in.yaml`:
  training launch cannot assign GPUs.
- `Auto-inferred nep_in_header changed while pot_inherit=True`:
  element order drifted across iterations; set `nep_in_header` explicitly.
- failing `nep` subprocess with only task-local log:
  inspect `iter.XXXXXX/00.nep/task.XXXXXX/log` first; do not assume the Python orchestration is at fault.

## Good inspection order

1. `in.yaml`
2. `record.nep`
3. `init/iter_train.xyz` and `init/iter_test.xyz`
4. latest `iter.XXXXXX/00.nep/dataset/`
5. failing `iter.XXXXXX/00.nep/task.XXXXXX/log`
6. prior iteration's `nep.in` and `nep.restart` when inheritance is enabled
