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
export PISA_WORK_DIR="${PISA_WORK_DIR:-${repo_dir}}"
export NLTK_DATA="${NLTK_DATA:-${PISA_WORK_DIR}/data}"
export HF_HOME="${HF_HOME:-${PISA_WORK_DIR}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${PISA_WORK_DIR}/.cache/torch}"
work_dir=${PISA_WORK_DIR:-${repo_dir}}
# Training arguments; position 18 selects the RULER tasks.
debug=$2
config_name=$4
batch_size=$5
accumulation=$6
seq_len=$7
varlen=$8
model_type=$9
steps=${10}
eval_batch_size=${11:-$batch_size}
gpu=${12:-gpu}
stage=${PISA_STAGE:-pt}
case "$stage" in pt) default_lr=3e-4 ;; cpt) default_lr=3e-5 ;; *) echo 'PISA_STAGE must be pt or cpt.' >&2; exit 2 ;; esac
lr=${13:-$default_lr}
dataname=${15:-10b}
context_len=${16:--1}
tasks=${18:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue}
case "$debug:$varlen" in true:true|true:false|false:true|false:false) ;; *) echo 'DEBUG and VARLEN must be true or false.' >&2; exit 2 ;; esac
case "$dataname" in
  10b) suffix=ckpt ;;
  100b) suffix=ckpt_100b ;;
  350b) suffix=ckpt_350b ;;
  *) echo 'DATASET must be 10b, 100b or 350b.' >&2; exit 2 ;;
esac

# Checkpoint
ckpt_dir=${PISA_CKPT_DIR:-${work_dir}/flame/ckpt/flame/${suffix}}
warmup_steps=${PISA_WARMUP_STEPS:-1000}
eps=${PISA_ADAM_EPS:-1e-8}
seed=${TRAINING_SEED:-42}
model_name=${config_name}_b${batch_size}_g${accumulation}_seq${seq_len}_step${steps}_warmup${warmup_steps}_lr${lr}_eps${eps}_${gpu}
[[ "$varlen" == true ]] && model_name=${model_name}_ctx${context_len}
[[ "$seed" != 42 ]] && model_name=${model_name}_seed${seed}
[[ "$stage" == cpt ]] && model_name=${model_name}_cpt
[[ "$debug" == true ]] && model_name=${model_name}_debug
dump_folder=${PISA_OUTPUT_DIR:-${ckpt_dir}/${model_type}/${model_name}}
model_path=${PISA_HF_MODEL_PATH:-${dump_folder}/checkpoint/step-${steps}-hf}
eval_dir=${work_dir}/flame/eval
export PYTHONPATH="${repo_dir}/pisa:${repo_dir}/xmixers:${FLAME_CODE_PATH:-${repo_dir}/flame}:${PYTHONPATH:-}"
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export SQUAD_FILE_PATH=${SQUAD_FILE_PATH:-${work_dir}/data/dev-v2.0.json}
export HOTPOT_FILE_PATH=${HOTPOT_FILE_PATH:-${work_dir}/data/hotpot_dev_distractor_v1.json}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
[[ "${NODE_RANK:-${PET_NODE_RANK:-${CLUSTER_ID:-0}}}" != 0 ]] && exit 0

# Evaluation
metadata=${PISA_RULER_METADATA:-'{"max_seq_lengths":[1024,2048,4096,8192,16384]}'}
mkdir -p "${eval_dir}/logs"
python -m torch.distributed.run \
  --standalone \
  --nproc-per-node "${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-${CLUSTER_WORKER_GPU:-${CLUSTER_WORKER_GPU_PER_NODE:-1}}}}" \
  "${EVAL_HARNESS:-${repo_dir}/xmixers/evals/harness.py}" \
  --model hf \
  --model_args "pretrained=${model_path},dtype=bfloat16,max_length=${PISA_RULER_MAX_LENGTH:-32768},parallelize=True" \
  --metadata "$metadata" \
  --tasks "$tasks" \
  --output_path "${eval_dir}/output/${model_name}" \
  --gen_kwargs 'temperature=0.0,do_sample=False' \
  --batch_size "$eval_batch_size" \
  2>&1 | tee "${eval_dir}/logs/$(date +%Y%m%d-%H:%M:%S)_${model_name}_ruler.log"
