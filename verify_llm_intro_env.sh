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

"$CONDA_BIN" run -n "$ENV_NAME" python -c "
import sys
import torch
import matplotlib
import numpy

sys.path.insert(0, '$WORKDIR/llm_training_stages')
sys.path.insert(0, '$WORKDIR/attention_design_metrics')

from tiny_llm_training import CharTokenizer, TinyGPT, collect_all_texts
from attention_metrics_torch import make_causal_mask, graph_diameter

tok = CharTokenizer(collect_all_texts())
model = TinyGPT(tok.vocab_size, 32, d_model=32, n_heads=4, n_layers=1)

print('python:', sys.executable)
print('torch:', torch.__version__)
print('torch_file:', torch.__file__)
print('cuda_available:', torch.cuda.is_available())
print('matplotlib:', matplotlib.__version__)
print('numpy:', numpy.__version__)
print('tiny_llm_params:', sum(p.numel() for p in model.parameters()))
print('attention_diameter:', graph_diameter(make_causal_mask(8))['diameter'])
print('env_check: OK')
"
