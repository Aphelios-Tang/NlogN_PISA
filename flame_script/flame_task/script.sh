#!/usr/bin/env bash
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Paths
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"
code_dir="${FLAME_CODE_PATH:-${repo_dir}/flame}"
export PISA_WORK_DIR="${PISA_WORK_DIR:-${repo_dir}}"

# TP DEBUG CONFIG_DIR CONFIG_NAME BATCH ACCUM SEQ VARLEN TAG STEPS
# EVAL_BATCH GPU_LABEL LR SAVE_INTERVAL DATASET CONTEXT AC_MODE [TASKS]
training_tensor_parallel_degree=$1
debug=$2
config_dir=$3
config_name=$4
batch_size=$5
gradient_accumulation_steps=$6
seq_len=${7:-2048}
varlen=${8:-false}
model_type=$9
step=${10}
eval_batch_size=${11:-$batch_size}
gpu=${12:-gpu}

# PT and CPT defaults
post_train=${PISA_POST_TRAIN:-eval}
case "$post_train" in none|convert|eval) ;; *) echo "PISA_POST_TRAIN must be none, convert or eval." >&2; exit 2 ;; esac
stage=${PISA_STAGE:-pt}
case "$stage" in
  pt)
    default_lr=3e-4
    default_scheduler=wsd
    default_compile=true
    export PISA_DATA_FORMAT=${PISA_DATA_FORMAT:-text}
    ;;
  cpt)
    default_lr=3e-5
    default_scheduler=cosine
    default_compile=false
    export PISA_DATA_FORMAT=${PISA_DATA_FORMAT:-zarr}
    if [[ "$varlen" != true ]]; then
      echo 'CPT requires varlen=true.' >&2
      exit 2
    fi
    ;;
  *) echo "PISA_STAGE must be pt or cpt, got: $stage" >&2; exit 2 ;;
esac
case "$debug:$varlen" in true:true|true:false|false:true|false:false) ;; *) echo 'DEBUG and VARLEN must be true or false.' >&2; exit 2 ;; esac
lr=${13:-$default_lr}
checkpoint_interval=${14:-2048}
dataname=${15:-10b}
context_len=${16:--1}
mode=${17:-none}
tasks=${18:-wikitext,lambada_openai,boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,fda,swde,squad_completion}

# Data and model
case "$dataname" in
  10b) ckpt_suffix=ckpt; wandb_project=flame_project; dataset=${DATA_10B_PATH:-} ;;
  100b) ckpt_suffix=ckpt_100b; wandb_project=flame_project_100b; dataset=${DATA_100B_PATH:-} ;;
  350b) ckpt_suffix=ckpt_350b; wandb_project=flame_project_350b; dataset=${DATA_350B_PATH:-} ;;
  *) echo "Unsupported dataname: $dataname (use 10b, 100b or 350b)." >&2; exit 2 ;;
esac
dataset=${PISA_DATA_PATH:-$dataset}
: "${dataset:?Set PISA_DATA_PATH or the corresponding DATA_10B_PATH/DATA_100B_PATH/DATA_350B_PATH}"
tokenizer=${PISA_TOKENIZER_PATH:-openai-community/gpt2}
ckpt_dir=${PISA_CKPT_DIR:-${PISA_WORK_DIR}/flame/ckpt/flame/${ckpt_suffix}}
config=${PISA_MODEL_CONFIG:-${code_dir}/configs/${config_dir}/${config_name}.json}
if [[ ! -f "$config" ]]; then
  echo "Missing model config: $config" >&2
  exit 2
fi

# Run name
warmup_steps=${PISA_WARMUP_STEPS:-1000}
eps=${PISA_ADAM_EPS:-1e-8}
training_seed=${TRAINING_SEED:-42}
wandb_name=${config_name}_b${batch_size}_g${gradient_accumulation_steps}_seq${seq_len}_step${step}_warmup${warmup_steps}_lr${lr}_eps${eps}_${gpu}
[[ "$varlen" == true ]] && wandb_name=${wandb_name}_ctx${context_len}
[[ "$training_seed" != 42 ]] && wandb_name=${wandb_name}_seed${training_seed}
if [[ "$stage" == cpt ]]; then
  wandb_name=${wandb_name}_cpt
  wandb_project=${wandb_project}_cpt
fi
if [[ "$debug" == true ]]; then
  wandb_name=${wandb_name}_debug
  wandb_project=${wandb_project}_debug
fi
job_dump_folder=${PISA_OUTPUT_DIR:-${ckpt_dir}/${model_type}/${wandb_name}}
export PISA_OUTPUT_DIR="$job_dump_folder"
export PISA_CKPT_DIR="$ckpt_dir"
export PISA_STAGE="$stage"
export PISA_DEBUG="$debug"
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} WANDB_MODE=${WANDB_MODE:-offline}

# Training
bash "${script_dir}/train.sh" \
  "job_config_file=${code_dir}/train.toml" \
  "job_dump_folder=${job_dump_folder}" \
  "model_config=${config}" \
  "model_tokenizer_path=${tokenizer}" \
  "model_source=xmixers" \
  "optimizer_name=AdamW" \
  "optimizer_lr=${lr}" \
  "optimizer_min_lr_ratio=0.1" \
  "optimizer_scheduler=${PISA_SCHEDULER:-$default_scheduler}" \
  "eps=${eps}" \
  "training_batch_size=${batch_size}" \
  "training_seq_len=${seq_len}" \
  "training_context_len=${context_len}" \
  "training_warmup_steps=${warmup_steps}" \
  "training_gradient_accumulation_steps=${gradient_accumulation_steps}" \
  "training_steps=${step}" \
  "training_max_norm=1.0" \
  "training_skip_nan_inf=true" \
  "training_dataset=${dataset}" \
  "training_dataset_name=${DATASET_NAME:-default}" \
  "training_dataset_split=train" \
  "training_streaming=${PISA_STREAMING:-true}" \
  "training_num_workers=${PISA_NUM_WORKERS:-32}" \
  "training_prefetch_factor=2" \
  "training_seed=${training_seed}" \
  "training_compile=${PISA_COMPILE:-$default_compile}" \
  "training_data_parallel_shard_degree=${PISA_SHARD_DEGREE:--1}" \
  "training_data_parallel_replicate_degree=${PISA_REPLICATE_DEGREE:-1}" \
  "training_debug=false" \
  "checkpoint_interval=${checkpoint_interval}" \
  "checkpoint_load_step=${PISA_LOAD_STEP:--1}" \
  "metrics_log_freq=${PISA_LOG_FREQ:-100}" \
  "disable_loss_parallel=false" \
  "mode=${mode}" \
  "selective_ac_option=2" \
  "training_tensor_parallel_degree=${training_tensor_parallel_degree}" \
  "training_cache_size_limit=64" \
  "training_compile_fullgraph=false" \
  "debug=${debug}" \
  "varlen=${varlen}" \
  "wandb_project=${WANDB_PROJECT:-$wandb_project}" \
  "wandb_name=${WANDB_NAME:-$wandb_name}"

# Checkpoint export and evaluation
[[ "$post_train" == none ]] && exit 0
if [[ ! -s "${job_dump_folder}/checkpoint/step-${step}/.metadata" ]]; then
  echo "Training finished without checkpoint metadata: ${job_dump_folder}/checkpoint/step-${step}/.metadata" >&2; exit 1
fi
model_max_length=$seq_len
[[ "$context_len" != -1 ]] && model_max_length=$context_len
bash "${script_dir}/convert_and_eval.sh" "$model_type" "$wandb_name" "${config_dir}/${config_name}.json" "step-${step}" "$tokenizer" "$model_max_length" "$eval_batch_size" "$dataname" "$tasks"
