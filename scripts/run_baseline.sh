#!/usr/bin/env bash
set -euo pipefail
ROOT=/work/users/y/x/yxyang/ICRL/ICRL
export HF_HOME="$ROOT/models"
export PYTHONPATH="${PYTHONPATH:-}"
CSV_PATH="$ROOT/results/baseline/metrics.csv"
CONFIG="$ROOT/configs/baseline.yaml"

# fetch default batch from YAML
BATCH=$(python - <<PY
import yaml, sys
cfg=yaml.safe_load(open("$CONFIG"))
print(cfg.get('batch', 4))
PY
)

single_run() {
  local MODEL="$1"; local TASK="$2"
  local OUTDIR="$ROOT/results/baseline/${TASK}_${MODEL}"
  echo "[RUN] $MODEL  |  $TASK  | batch=$BATCH"
  PYTHONPATH="$ROOT/src:${PYTHONPATH}" \
    python -m baseline.baseline_runner \
      --model_path "$ROOT/models/$MODEL" \
      --task_dir   "$ROOT/data/$TASK" \
      --output_dir "$OUTDIR" \
      --csv_path   "$CSV_PATH" \
      --batch "$BATCH"
}

if [[ "${1:-}" == "all" ]]; then
  readarray -t MODELS   < <(python - <<PY
import yaml, sys; print(*yaml.safe_load(open("$CONFIG"))['models'], sep="\n")
PY
)
  readarray -t DATASETS < <(python - <<PY
import yaml, sys; print(*yaml.safe_load(open("$CONFIG"))['datasets'], sep="\n")
PY
)
  for m in "${MODELS[@]}"; do
    for d in "${DATASETS[@]}"; do
      single_run "$m" "$d"
    done
  done
else
  [[ $# -eq 2 ]] || { echo "Usage: $0 MODEL DATASET | $0 all"; exit 1; }
  single_run "$1" "$2"
fi