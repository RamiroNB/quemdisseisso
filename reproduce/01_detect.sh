#!/usr/bin/env bash
# Opinion-level probe on the benchmark's human labels, zero-shot transfer from English,
# RAGTruth external test, NLI baseline. GPU for the extractions, CPU for the rest.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="NousResearch/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-4B-Instruct-2507"

# opinion-level probe: activations of every labelled opinion, then the per-layer probe (GroupKFold by hearing)
for repo in $MODELS; do
  OCARANDU_MODEL_REPO=$repo uv run python scripts/extract_publichearingbr_activations.py
done
for act in activations activations_llama_31_8b_instruct activations_qwen25_7b_instruct activations_qwen3_4b_instruct_2507; do
  OCARANDU_ACT_DIR=$act uv run python scripts/run_publichearingbr_probe.py
done

# RAGTruth sentence activations (needs vendor/ragtruth, see README)
for repo in NousResearch/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct; do
  OCARANDU_MODEL_REPO=$repo uv run python scripts/extract_ragtruth_sentence_mean.py
done

# zero-shot transfer: probe trained on English RAGTruth, applied to the Portuguese labels
uv run python scripts/crosslingual_probe.py llama_2_7b_chat_hf activations
uv run python scripts/crosslingual_probe.py llama_31_8b_instruct
uv run python scripts/crosslingual_probe.py qwen25_7b_instruct

# RAGTruth summarisation, official test split
for t in llama_31_8b_instruct llama_2_7b_chat_hf qwen25_7b_instruct; do
  uv run python scripts/ragtruth_external_eval.py $t
done
uv run python scripts/ragtruth_external_eval.py --report

# NLI cross-encoder against the benchmark's human labels
uv run python scripts/nli_verifier.py benchmark
