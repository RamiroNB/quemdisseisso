#!/usr/bin/env bash
# Audit: metadata use with speaker tags restored, credibility of a printed title, omission from the
# published article. Needs 00 and 01. GPU for the model runs, CPU for the analyses.
set -euo pipefail
cd "$(dirname "$0")/.."

# metadata use: does the model confirm a swapped attribution when the evidence carries speaker tags?
OCARANDU_TAGGED=1 uv run python scripts/identity_swap.py            # Llama-2 (default model, layer 14)
OCARANDU_TAGGED=1 OCARANDU_MODEL_REPO=meta-llama/Llama-3.1-8B-Instruct OCARANDU_ACT_DIR=activations_llama_31_8b_instruct \
  OCARANDU_PROBE_LAYER=13 OCARANDU_SAVE_LAYERS=6,8,10,12,13,14,16,20,24 uv run python scripts/identity_swap.py
OCARANDU_TAGGED=1 OCARANDU_MODEL_REPO=Qwen/Qwen2.5-7B-Instruct OCARANDU_ACT_DIR=activations_qwen25_7b_instruct \
  OCARANDU_PROBE_LAYER=18 OCARANDU_SAVE_LAYERS=6,8,10,12,14,16,18,20,24,18 uv run python scripts/identity_swap.py
for t in llama_2_7b_chat_hf_tagged llama_31_8b_instruct_tagged qwen25_7b_instruct_tagged; do
  uv run python scripts/identity_swap_analysis.py $t 0.5
done

# generation: statements invented for an absent speaker, plain prompt vs abstention instruction
OCARANDU_MODEL_REPO=meta-llama/Llama-3.1-8B-Instruct OCARANDU_ACT_DIR=activations_llama_31_8b_instruct OCARANDU_GEN_LAYER=14 \
  OCARANDU_GEN_N=200 OCARANDU_GEN_CONDITIONS=baseline,misattr:-0.5,misattr:-1.0,halluc:-0.5,random:-0.5 \
  uv run python scripts/attribution_generation.py
OCARANDU_MODEL_REPO=Qwen/Qwen2.5-7B-Instruct OCARANDU_ACT_DIR=activations_qwen25_7b_instruct OCARANDU_GEN_LAYER=18 \
  OCARANDU_GEN_N=200 OCARANDU_GEN_CONDITIONS=baseline,misattr:-0.5,misattr:-0.25,halluc:-0.5,random:-0.5 \
  uv run python scripts/attribution_generation.py
uv run python scripts/attribution_generation_stats.py llama_31_8b_instruct
uv run python scripts/attribution_generation_stats.py qwen25_7b_instruct

# credibility: only the printed title of a false attribution changes
for spec in "meta-llama/Llama-3.1-8B-Instruct:10,12,14,16,18,20" "Qwen/Qwen2.5-7B-Instruct:12,14,16,18,20,24" \
            "NousResearch/Llama-2-7b-chat-hf:10,12,14,16,18,20" "Qwen/Qwen3-4B-Instruct-2507:12,16,18,21,24,28"; do
  repo=${spec%%:*}; layers=${spec#*:}
  tag=$(python3 -c "print('${repo##*/}'.lower().replace('.','').replace('-','_'))")
  OCARANDU_MODEL_REPO=$repo OCARANDU_TAGGED=1 OCARANDU_TITLE_AXIS=role OCARANDU_SAVE_LAYERS=$layers OCARANDU_TITLE_N=500 \
    uv run python scripts/title_swap.py
  uv run python scripts/title_swap_analysis.py ${tag}_tagged 5000
done
uv run python scripts/title_swap_decisions.py llama_31_8b_instruct_tagged qwen25_7b_instruct_tagged \
  llama_2_7b_chat_hf_tagged qwen3_4b_instruct_2507_tagged
for spec in "meta-llama/Llama-3.1-8B-Instruct:10,12,14,16,18,20" "Qwen/Qwen2.5-7B-Instruct:12,14,16,18,20,24"; do
  repo=${spec%%:*}; layers=${spec#*:}
  tag=$(python3 -c "print('${repo##*/}'.lower().replace('.','').replace('-','_'))")
  for axis in party gender; do
    OCARANDU_MODEL_REPO=$repo OCARANDU_TAGGED=1 OCARANDU_TITLE_AXIS=$axis OCARANDU_SAVE_LAYERS=$layers OCARANDU_TITLE_N=500 \
      uv run python scripts/title_swap.py
    uv run python scripts/title_axis_analysis.py ${tag}_tagged_${axis} 5000
  done
done

# omission from the published article: floor-holders, then the labels and tests
uv run python scripts/extract_lds_audit.py
uv run python scripts/lds_audit_stage2.py llama_31_8b_instruct
uv run python scripts/lds_omission_relabel.py llama_31_8b_instruct
