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
code_dir=${FLAME_CODE_PATH:-${repo_dir}/flame}
PISA_WORK_DIR=${PISA_WORK_DIR:-${repo_dir}}

# MODEL_TYPE MODEL_NAME CONFIG STEP TOKENIZER [MAX_LENGTH BATCH DATASET TASKS]
model_type=$1
model_name=$2
config=$3
step=$4
tokenizer=$5
model_max_length=${6:-2048}
batch_size=${7:-32}
dataname=${8:-10b}
tasks=${9:-wikitext,lambada_openai,boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,fda,swde,squad_completion}
case "$dataname" in
  10b) suffix=ckpt ;;
  100b) suffix=ckpt_100b ;;
  350b) suffix=ckpt_350b ;;
  *) echo "Unsupported dataname: $dataname" >&2; exit 2 ;;
esac

ckpt_dir=${PISA_CKPT_DIR:-${PISA_WORK_DIR}/flame/ckpt/flame/${suffix}}
dump_folder=${PISA_OUTPUT_DIR:-${ckpt_dir}/${model_type}/${model_name}}
checkpoint=${dump_folder}/checkpoint/${step}
model_path=${checkpoint}-hf
export PYTHONPATH="${repo_dir}/pisa:${repo_dir}/xmixers:${code_dir}:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# Only node zero exports and evaluates the checkpoint.
[[ "${NODE_RANK:-${PET_NODE_RANK:-${CLUSTER_ID:-0}}}" != 0 ]] && exit 0

# Export
logger_dir=${PISA_WORK_DIR}/flame/logs_transform
mkdir -p "$logger_dir"

python "${code_dir}/script/convert_dcp_to_hf.py" \
  --checkpoint "$checkpoint" \
  --path "$model_path" \
  --config "${PISA_MODEL_CONFIG:-${code_dir}/configs/${config}}" \
  --model_max_length "$model_max_length" \
  --tokenizer "$tokenizer" \
  --stage "${PISA_STAGE:-pt}" \
  2>&1 | tee "${logger_dir}/$(date +%Y%m%d-%H:%M:%S)-${model_name}.log"
[[ "${PISA_POST_TRAIN:-eval}" == convert ]] && exit 0

# Evaluation
eval_dir=${PISA_WORK_DIR}/flame/eval
eval_harness=${EVAL_HARNESS:-${repo_dir}/xmixers/evals/harness.py}

mkdir -p "${eval_dir}/logs" "${eval_dir}/output/${model_name}"
python -m pip freeze > "${eval_dir}/output/${model_name}/environment.txt"
python -m torch.distributed.run \
  --nnodes 1 --node-rank 0 \
  --master-addr 127.0.0.1 --master-port "${MASTER_PORT:-29500}" \
  --nproc-per-node "${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-${CLUSTER_WORKER_GPU:-${CLUSTER_WORKER_GPU_PER_NODE:-1}}}}" \
  "$eval_harness" \
  --model hf \
  --model_args "pretrained=${model_path},dtype=bfloat16" \
  --tasks "$tasks" \
  --output_path "${eval_dir}/output/${model_name}" \
  --trust_remote_code \
  --batch_size "$batch_size" \
  --log_samples \
  2>&1 | tee "${eval_dir}/logs/$(date +%Y%m%d-%H:%M:%S)_${model_name}.log"
