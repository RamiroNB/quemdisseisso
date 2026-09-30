#!/usr/bin/env bash
# Per-participant summaries (speaker-segmented task) and the naive whole-transcript baseline. GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

LLAMA="OCARANDU_MODEL_REPO=meta-llama/Llama-3.1-8B-Instruct OCARANDU_ACT_DIR=activations_llama_31_8b_instruct OCARANDU_GEN_LAYER=14"
QWEN="OCARANDU_MODEL_REPO=Qwen/Qwen3-4B-Instruct-2507 OCARANDU_ACT_DIR=activations_qwen3_4b_instruct_2507 OCARANDU_GEN_LAYER=21"

# steering dose chosen on the development hearings by a fixed rule (longcontext_pick_alpha.py)
for spec in "llama_31_8b_instruct:$LLAMA" "qwen3_4b_instruct_2507:$QWEN"; do
  tag=${spec%%:*}; envs=${spec#*:}
  env $envs OCARANDU_LC_SPLIT=dev OCARANDU_LC_N=100 OCARANDU_LC_PER_HEARING=2 OCARANDU_LC_TAG=lc_dev \
    OCARANDU_LC_CONDITIONS="baseline,halluc:0.5,halluc:0.75,halluc:1.0,halluc:1.25,random0:1.0,random1:1.0,random2:1.0,random0:0.5,random1:0.5" \
    uv run python scripts/longcontext_speaker_generation.py
done
uv run python scripts/longcontext_rescore.py llama_31_8b_instruct_lc_dev qwen3_4b_instruct_2507_lc_dev
uv run python scripts/longcontext_pick_alpha.py llama_31_8b_instruct_lc_dev
uv run python scripts/longcontext_pick_alpha.py qwen3_4b_instruct_2507_lc_dev

# cross-fit: direction fit on one half of the hearings, summaries of the other half, then swapped.
# Baseline, steered and random-direction arms; the token-probe and corruption steps of 03 read all three.
for spec in "llama_31_8b_instruct:$LLAMA" "qwen3_4b_instruct_2507:$QWEN"; do
  tag=${spec%%:*}; envs=${spec#*:}
  for fold in A B; do
    env $envs OCARANDU_LC_FOLD=$fold OCARANDU_LC_SPLIT=all OCARANDU_LC_N=100000 OCARANDU_LC_PER_HEARING=100 \
      OCARANDU_LC_TAG=lc_xfit$fold OCARANDU_LC_CONDITIONS="baseline,halluc:1.0,random0:1.0" \
      uv run python scripts/longcontext_speaker_generation.py
  done

  # naive pipeline: whole transcript in, "Name: opinion" lines out, all 206 hearings
  env $envs OCARANDU_FT_SPLIT=everything OCARANDU_FT_N=206 OCARANDU_FT_CTX_TOKENS=48000 OCARANDU_FT_MAXNEW=1200 \
    OCARANDU_FT_TAG=ft_all_baseline OCARANDU_FT_CONDITIONS=baseline \
    uv run python scripts/fulltranscript_generation.py
  uv run python scripts/fulltranscript_lines.py ${tag}_ft_all_baseline
done
