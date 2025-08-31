set -e
ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")" 
export HF_HOME="$ROOT/models"          # optional central cache
export HUGGINGFACE_HUB_TOKEN="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"  # <-- your token

PY_SCRIPT="$ROOT/models/download.py"

if [[ $# -gt 0 ]]; then
  python "$PY_SCRIPT" "$@"
else
  python "$PY_SCRIPT"
fi