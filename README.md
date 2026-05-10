# LLM Training Intro

Introductory learning materials for:

- LLM training stages: pre-train, SFT, and post-RL/DPO.
- Transformer and attention design concepts.
- PyTorch/Jupyter experiments for small local learning tasks.
- Printable Chinese, English, and bilingual PDF handouts.

Author: `liangdacheng / 梁达成`

## Contents

- `llm_training_stages/`
  - `llm_pretrain_sft_postrl_tutorial.ipynb`
  - `tiny_llm_training.py`
- `attention_design_metrics/`
  - `attention_metrics_tutorial.ipynb`
  - `attention_metrics_torch.py`
- `printable_tutorials/`
  - Chinese PDF: `llm_attention_training_tutorial_zh.pdf`
  - English PDF: `llm_attention_training_tutorial_en.pdf`
  - Bilingual PDF: `llm_attention_training_tutorial_bilingual.pdf`
- `environment.yml`
  - Conda environment for the intro notebooks.

## Quick Start

Create the environment:

```bash
conda env create -f environment.yml
```

Verify it:

```bash
./verify_llm_intro_env.sh
```

Start Jupyter:

```bash
./start_llm_intro_jupyter.sh
```

Select the kernel named `Python (llm-intro)`.

## Notes

The notebooks are intentionally small. They are designed to teach training mechanics and metrics, not to produce a useful assistant model.
