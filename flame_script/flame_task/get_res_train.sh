#!/usr/bin/env bash
# Summarize training logs as CSV.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"
code_dir=${FLAME_CODE_PATH:-${repo_dir}/flame}
if [[ $# -eq 0 ]]; then
  set -- --log_dir "${PISA_LOG_DIR:-${PISA_WORK_DIR:-${repo_dir}}/flame/logs}"
fi

export PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE:-1}
exec python "${code_dir}/script/get_training_stat.py" "$@"
