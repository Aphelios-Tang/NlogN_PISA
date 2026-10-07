# Training

Training uses the `xmixers` models and `pisa` operators.
Run commands from the repository root.

## Setup

On a connected machine with Python 3.11 and Git, run:

```bash
bash flame_script/env/setup_env.sh
```

The script creates `PISA/.env` and installs the pinned dependencies. Pass another
environment directory as the first argument to change its location. Both launchers
activate `.env` by default; set `PISA_ENV_DIR` for a custom location and `HF_HOME`
for the prepared dataset cache. Installation uses the network; training runs
offline. See the root [README](../README.md#installation).

## Train

Edit data paths and select an experiment in `launch_pt.sh` (PT) or
`launch_cpt.sh` (CPT). Configurations are in `flame/configs/pisa/pt/`
and `flame/configs/pisa/cpt/`. See the root [README](../README.md) for commands.

PT supports optional varlen attention and retains document-transition targets.
CPT uses document isolation and masks those targets. `PISA_SOURCE_CHECKPOINT` starts CPT
from PT model weights and Adam moments with a new schedule and data position;
resuming an existing run restores its full training state.
DDP and FSDP2 are supported; tensor, pipeline and context parallelism are not.

## CPT data

The default input is `train.zarr` with shape `[rows, 65536]`, plus:

```text
document-boundaries/train-row-offsets.npy    # int64, CSR offsets, rows + 1
document-boundaries/train-segment-ends.npy   # int32, document ends within each row
document-boundaries/train-manifest.json      # format_version=1, split, rows, row_tokens, segments
```

`PISA_DOCUMENT_START` and `PISA_DOCUMENT_COUNT` select the row window.
The training sequence count must match that window; `PISA_ALLOW_PARTIAL_DATA=true`
allows a shorter run. Resume with the same data window and training plan.

## Logs and evaluation

Logs go to `flame/logs`, checkpoints to `flame/ckpt/flame/`, and evaluation
results to `flame/eval`. Logs include loss, learning rate, gradient norm,
throughput, memory and elapsed/remaining time. WandB defaults to offline mode.

`PISA_POST_TRAIN` selects `none`, `convert`, or `eval` (default).
Evaluation requires locally available task datasets. `setup_env.sh` installs
lm-eval public commit `0c134ee944d97998013eaff6f4e76d1b9fa87ecd` plus three task
adaptations; its reported version remains 0.4.9.1. See
[Evaluation](../README.md#evaluation) for
source verification, the scoring convention and a complete evaluation command.
Evaluation saves the package versions and per-sample responses alongside results.

See [LICENSE](LICENSE), [NOTICE](NOTICE) and [LICENSE.torchtitan](LICENSE.torchtitan).
