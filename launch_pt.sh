#!/usr/bin/env bash
set -euo pipefail

# PT: keep one experiment command active.

# Environment
PISA_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${PISA_ENV_DIR:-${PISA_REPO_DIR}/.env}/bin/activate"

export HF_HUB_OFFLINE=1 \
       HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1
export WANDB_MODE=${WANDB_MODE:-offline}

export PISA_WORK_DIR="${PISA_WORK_DIR:-${PISA_REPO_DIR}}"
export FLAME_CODE_PATH="${FLAME_CODE_PATH:-${PISA_REPO_DIR}/flame}"
export PYTHONPATH="${PISA_REPO_DIR}/pisa:${PISA_REPO_DIR}/xmixers:${FLAME_CODE_PATH}:${PYTHONPATH:-}"
export PISA_STAGE=pt

# Paths and checkpoint resume
# export PISA_DATA_PATH=/path/to/pt-dataset
# export PISA_TOKENIZER_PATH=/path/to/tokenizer
# export PISA_OUTPUT_DIR=/path/to/run
# export PISA_LOAD_STEP=-1
# export PISA_POST_TRAIN=none

export NPROC_PER_NODE="${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-${CLUSTER_WORKER_GPU:-${CLUSTER_WORKER_GPU_PER_NODE:-8}}}}"
script="${PISA_REPO_DIR}/flame_script/flame_task/script.sh"

# TP DEBUG CONFIG_DIR CONFIG_NAME BATCH ACCUM SEQ VARLEN TAG STEPS
# EVAL_BATCH GPU_LABEL LR SAVE_INTERVAL DATASET CONTEXT AC_MODE [TASKS]

# PT / 4K
config_dir=pisa/pt

# BSA
# bash "$script" 1 false "$config_dir/bsa" bsa_418m 8 2 4096 false bsa_418m 100000 8 gpu 3e-4 2048 100b -1 none

# PISA1 / First order
# bash "$script" 1 false "$config_dir/pisa_1" pisa1_418m 8 2 4096 false pisa1_418m 100000 8 gpu 3e-4 2048 100b -1 none

# PISA2 / Second order
# bash "$script" 1 false "$config_dir/pisa_2" pisa2_418m 8 2 4096 false pisa2_418m 100000 8 gpu 3e-4 2048 100b -1 none

# PISA / LSE
bash "$script" 1 false "$config_dir/pisa" pisa_418m 8 2 4096 false pisa_418m 100000 8 gpu 3e-4 2048 100b -1 none

# NSA / Learned compression
# bash "$script" 1 false "$config_dir/nsa" nsa_418m 8 2 4096 false nsa_418m 100000 8 gpu 3e-4 2048 100b -1 none

# PT varlen: document attention with cross-document targets
# bash "$script" 1 false "$config_dir/pisa" pisa_418m 1 16 4096 true pisa_418m_varlen 100000 8 gpu 3e-4 2048 100b -1 none

# PT / 90M / 10B tokens (single node, 8 GPUs)
# bash "$script" 1 false pisa/pt/pisa pisa_90m 32 1 2048 false pisa_90m_10b 20000 1 gpu 3e-4 2000 10b -1 none
