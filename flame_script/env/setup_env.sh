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

# Environment path (default: PISA/.env)
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_dir="$(realpath -m "${1:-${PISA_ENV_DIR:-$repo_dir/.env}}")"

# Python environment
python3.11 -m venv "$env_dir"
source "$env_dir/bin/activate"
cd "$repo_dir"

# Dependencies
pip install -r flame/requirements.txt

# lm-eval
lm_eval_commit=0c134ee944d97998013eaff6f4e76d1b9fa87ecd
lm_eval_source="$env_dir/src/lm-evaluation-harness-pisa"

git init "$lm_eval_source"
git -C "$lm_eval_source" fetch https://github.com/EleutherAI/lm-evaluation-harness.git "$lm_eval_commit"
git -C "$lm_eval_source" checkout --detach "$lm_eval_commit"

# FDA, SQuAD and SIQA task changes
patch_file="$repo_dir/flame_script/env/lm_eval_tasks.patch"
git -C "$lm_eval_source" apply --reverse --check "$patch_file" 2>/dev/null ||
  git -C "$lm_eval_source" apply "$patch_file"

pip install --no-deps --no-build-isolation --force-reinstall -e "$lm_eval_source"

# PISA packages
pip install --no-deps --no-build-isolation -e ./pisa -e ./xmixers -e ./flame

pip check

printf '\nEnvironment: %s\n' "$env_dir"
printf 'Activate: source "%s/bin/activate"\n' "$env_dir"
