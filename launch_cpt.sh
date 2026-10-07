#!/usr/bin/env bash
set -euo pipefail

# CPT: keep one experiment command active.

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
export PISA_STAGE=cpt

# Data, tokenizer and PT checkpoint paths
# export PISA_DATA_PATH=/path/to/prepared-cpt-zarr
# export PISA_TOKENIZER_PATH=/path/to/tokenizer
# export PISA_SOURCE_CHECKPOINT=/path/to/PT/checkpoint/step-100000
# export PISA_DOCUMENT_COUNT=80000
# export PISA_OUTPUT_DIR=/path/to/cpt-run
# export PISA_LOAD_STEP=-1
# export PISA_POST_TRAIN=none

export NPROC_PER_NODE="${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-${CLUSTER_WORKER_GPU:-${CLUSTER_WORKER_GPU_PER_NODE:-8}}}}"
script="${PISA_REPO_DIR}/flame_script/flame_task/script.sh"

# TP DEBUG CONFIG_DIR CONFIG_NAME BATCH ACCUM SEQ VARLEN TAG STEPS
# EVAL_BATCH GPU_LABEL LR SAVE_INTERVAL DATASET CONTEXT AC_MODE [TASKS]

# CPT / 16K
config_dir=pisa/cpt

# BSA / K32
# bash "$script" 1 false "$config_dir/bsa" bsa_418m_k32 4 1 16384 true bsa_cpt 10000 1 gpu 3e-5 1000 10b 16384 none

# PISA1 / K32
# bash "$script" 1 false "$config_dir/pisa_1" pisa1_418m_k32 4 1 16384 true pisa1_cpt 10000 1 gpu 3e-5 1000 10b 16384 none

# PISA2 / K32
# bash "$script" 1 false "$config_dir/pisa_2" pisa2_418m_k32 4 1 16384 true pisa2_cpt 10000 1 gpu 3e-5 1000 10b 16384 none

# PISA / LSE: select a Top-K setting.
bash "$script" 1 false "$config_dir/pisa" pisa_418m_k8 4 1 16384 true pisa_cpt_k8 10000 1 gpu 3e-5 1000 10b 16384 none

# bash "$script" 1 false "$config_dir/pisa" pisa_418m_k16 4 1 16384 true pisa_cpt_k16 10000 1 gpu 3e-5 1000 10b 16384 none

# bash "$script" 1 false "$config_dir/pisa" pisa_418m_k32 4 1 16384 true pisa_cpt_k32 10000 1 gpu 3e-5 1000 10b 16384 none

# NSA / K32
# bash "$script" 1 false "$config_dir/nsa" nsa_418m_k32 4 1 16384 true nsa_cpt 10000 1 gpu 3e-5 1000 10b 16384 none
