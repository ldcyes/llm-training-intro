#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="llm-intro"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${CONDA_BIN:-}" ]; then
  :
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  CONDA_BIN="$HOME/miniconda3/bin/conda"
elif [ -x "$HOME/anaconda3/bin/conda" ]; then
  CONDA_BIN="$HOME/anaconda3/bin/conda"
else
  CONDA_BIN=""
fi

if [ ! -x "$CONDA_BIN" ]; then
  echo "conda not found"
  echo "Set CONDA_BIN=/path/to/conda and run again."
  exit 1
fi

cd "$WORKDIR"
"$CONDA_BIN" run -n "$ENV_NAME" python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"
"$CONDA_BIN" run -n "$ENV_NAME" jupyter lab
