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
code_path=${FLAME_CODE_PATH:-${repo_dir}/flame}
exp_path=${PISA_WORK_DIR:-${repo_dir}}/flame

# Distributed launch
LOG_RANK=${LOG_RANK:-0}
NNODES=${NNODES:-${PET_NNODES:-${CLUSTER_WORKER_NUM:-1}}}
NPROC_PER_NODE=${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-${CLUSTER_WORKER_GPU:-${CLUSTER_WORKER_GPU_PER_NODE:-1}}}}
NODE_RANK=${NODE_RANK:-${PET_NODE_RANK:-${CLUSTER_ID:-0}}}
MASTER_ADDR=${MASTER_ADDR:-${PET_MASTER_ADDR:-${CLUSTER_WORKER_0_HOST:-${CLUSTER_EXECUTOR_0_HOST:-127.0.0.1}}}}
cluster_port=${CLUSTER_WORKER_0_PORT:-${CLUSTER_EXECUTOR_0_PORT:-}}
MASTER_PORT=${MASTER_PORT:-${PET_MASTER_PORT:-${cluster_port%%,*}}}
MASTER_PORT=${MASTER_PORT:-29500}

# Command-line overrides
for arg in "$@"; do
  [[ "$arg" =~ ^[a-zA-Z_][a-zA-Z0-9_]*= ]] || { echo "Expected key=value, got: $arg" >&2; exit 2; }
  key=${arg%%=*}
  value=${arg#*=}
  case "$key" in
    job_*|model_*|optimizer_*|training_*|checkpoint_*|metrics_*|disable_loss_parallel|mode|selective_ac_option|eps|debug|varlen|wandb_project|wandb_name) ;;
    *) echo "Unsupported Flame launch parameter: $key" >&2; exit 2 ;;
  esac
  printf -v "$key" '%s' "$value"
done

# Training options
training_skip_nan_inf=${training_skip_nan_inf:-false}
training_streaming=${training_streaming:-false}
training_compile=${training_compile:-false}
training_debug=${training_debug:-false}
disable_loss_parallel=${disable_loss_parallel:-false}
training_compile_fullgraph=${training_compile_fullgraph:-false}
training_compile_whole_model=${training_compile_whole_model:-false}
varlen=${varlen:-false}
debug=${debug:-false}
wandb_project=${wandb_project:-debug}
wandb_name=${wandb_name:-debug}

# Training arguments
params=(
  --job.config_file "$job_config_file"
  --job.dump_folder "$job_dump_folder"
  --model.config "$model_config"
  --model.tokenizer_path "$model_tokenizer_path"
  --model.model_source "$model_source"
  --optimizer.name "$optimizer_name"
  --optimizer.lr "$optimizer_lr"
  --optimizer.min_lr_ratio "$optimizer_min_lr_ratio"
  --optimizer.scheduler "$optimizer_scheduler"
  --optimizer.eps "$eps"
  --training.batch_size "$training_batch_size"
  --training.seq_len "$training_seq_len"
  --training.context_len "$training_context_len"
  --training.warmup_steps "$training_warmup_steps"
  --training.gradient_accumulation_steps "$training_gradient_accumulation_steps"
  --training.steps "$training_steps"
  --training.max_norm "$training_max_norm"
  --training.dataset "$training_dataset"
  --training.dataset_name "$training_dataset_name"
  --training.dataset_split "$training_dataset_split"
  --training.num_workers "$training_num_workers"
  --training.prefetch_factor "$training_prefetch_factor"
  --training.seed "$training_seed"
  --training.dataset_format "${PISA_DATA_FORMAT:-text}"
  --training.data_parallel_shard_degree "$training_data_parallel_shard_degree"
  --training.data_parallel_replicate_degree "$training_data_parallel_replicate_degree"
  --training.cache_size_limit "$training_cache_size_limit"
  --training.tensor_parallel_degree "$training_tensor_parallel_degree"
  --checkpoint.interval "$checkpoint_interval"
  --checkpoint.load_step "$checkpoint_load_step"
  --activation_checkpoint.mode "$mode"
  --activation_checkpoint.selective_ac_option "$selective_ac_option"
  --metrics.log_freq "$metrics_log_freq"
  --training.stage "${PISA_STAGE:-pt}"
)

# Optional arguments
[[ -n "${PISA_SOURCE_CHECKPOINT:-}" ]] && params+=(--checkpoint.init_from "$PISA_SOURCE_CHECKPOINT")
[[ -n "${PISA_DOCUMENT_START:-}" ]] && params+=(--training.document_start "$PISA_DOCUMENT_START")
[[ -n "${PISA_DOCUMENT_COUNT:-}" ]] && params+=(--training.document_count "$PISA_DOCUMENT_COUNT")
[[ "${PISA_ALLOW_PARTIAL_DATA:-false}" == true ]] && params+=(--training.allow_partial_data)
[[ "${training_skip_nan_inf}" == true ]] && params+=(--training.skip_nan_inf)
[[ "${training_streaming}" == true ]] && params+=(--training.streaming)
[[ "${training_compile}" == true ]] && params+=(--training.compile)
[[ "${training_debug}" == true ]] && params+=(--training.debug)
[[ "${training_compile_fullgraph}" == true ]] && params+=(--training.compile_fullgraph)
[[ "${training_compile_whole_model}" == true ]] && params+=(--training.compile_whole_model)
[[ "${disable_loss_parallel}" == true ]] && params+=(--training.disable_loss_parallel)
[[ "${varlen}" == true ]] && params+=(--training.varlen)

# Logs
echo "debug:           $debug"
echo "wandb_project:   $wandb_project"
echo "wandb_name:      $wandb_name"
logger_dir=${exp_path}/logs
[[ "$debug" == true ]] && logger_dir=${exp_path}/logs_test
log_node=
(( NNODES > 1 )) && log_node=-node${NODE_RANK}
LOG_FILE=${logger_dir}/$(date +%Y%m%d-%H:%M:%S)-${wandb_name}${log_node}.log

export PYTHONPATH="${repo_dir}/pisa:${repo_dir}/xmixers:${code_path}:${PYTHONPATH:-}"
export WANDB_PROJECT="$wandb_project" WANDB_NAME="$wandb_name"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
echo "log_file:       $LOG_FILE"
mkdir -p "$(dirname -- "$LOG_FILE")"

# Launch
if [[ "$debug" == true ]]; then
  python -m torch.distributed.run \
    --standalone \
    --nproc-per-node "$NPROC_PER_NODE" \
    --local-ranks-filter "$LOG_RANK" \
    "${code_path}/train.py" \
    "${params[@]}" 2>&1 | tee -a "$LOG_FILE"
elif [[ "$debug" == false ]]; then
  python -m torch.distributed.run \
    --nproc-per-node "$NPROC_PER_NODE" \
    --nnodes "$NNODES" \
    --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    --local-ranks-filter "$LOG_RANK" \
    "${code_path}/train.py" \
    "${params[@]}" 2>&1 | tee -a "$LOG_FILE"
else
  echo "debug must be true or false, got: $debug" >&2
  exit 2
fi
echo 'RUNNING DONE!'
