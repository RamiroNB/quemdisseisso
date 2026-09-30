#!/usr/bin/env bash
# Token-level probe (next content word absent from the source) and rule-based corruptions of the
# models' own opinions. Needs the cross-fit runs of 02. GPU for the re-reads, CPU for the stats.
set -euo pipefail
cd "$(dirname "$0")/.."

LLAMA="OCARANDU_MODEL_REPO=meta-llama/Llama-3.1-8B-Instruct OCARANDU_ACT_DIR=activations_llama_31_8b_instruct OCARANDU_GEN_LAYER=14"
QWEN="OCARANDU_MODEL_REPO=Qwen/Qwen3-4B-Instruct-2507 OCARANDU_ACT_DIR=activations_qwen3_4b_instruct_2507 OCARANDU_GEN_LAYER=21"

# token-level probe; lc_stream_stats.py also saves the trigger probe (lc_stream/<model>/probe_tokens_novel.npz)
env $LLAMA OCARANDU_LS_LAYERS=10,14,20 uv run python scripts/lc_stream_extract.py
uv run python scripts/lc_stream_stats.py llama_31_8b_instruct
env $QWEN OCARANDU_LS_LAYERS=14,21,28 uv run python scripts/lc_stream_extract.py
uv run python scripts/lc_stream_stats.py qwen3_4b_instruct_2507

# corruptions: every generated opinion and its corrupted twins, re-read in the probe's format
env $LLAMA uv run python scripts/probe_fresh_corruptions.py
env $LLAMA uv run python scripts/probe_fresh_stats.py llama_31_8b_instruct
env $QWEN OCARANDU_PF_LAYERS=15,18,21,24,27 uv run python scripts/probe_fresh_corruptions.py
env $QWEN uv run python scripts/probe_fresh_stats.py qwen3_4b_instruct_2507

# NLI on the same pairs
OCARANDU_NLI_PF_TAG=llama_31_8b_instruct uv run python scripts/nli_verifier.py corruptions
