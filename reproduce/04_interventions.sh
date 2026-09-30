#!/usr/bin/env bash
# Retrieve-and-regenerate triggered by the token probe, and the Llama-3.1 summaries of the blind round
# whose baseline arm routing_fresh.py uses. Needs 02 and 03 (the probe is
# lc_stream/<model>/probe_tokens_novel.npz). GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

LLAMA="OCARANDU_MODEL_REPO=meta-llama/Llama-3.1-8B-Instruct OCARANDU_ACT_DIR=activations_llama_31_8b_instruct OCARANDU_GEN_LAYER=14"
QWEN="OCARANDU_MODEL_REPO=Qwen/Qwen3-4B-Instruct-2507 OCARANDU_ACT_DIR=activations_qwen3_4b_instruct_2507 OCARANDU_GEN_LAYER=21"
FOLD_A="OCARANDU_LC_FOLD=A OCARANDU_LC_SPLIT=all OCARANDU_LC_PER_HEARING=100"
T=llama_31_8b_instruct

# second arm of the blind round synth3_gcad500 (probe-gated context-aware decoding, 500 fold-A participants)
env $LLAMA $FOLD_A OCARANDU_LC_N=500 OCARANDU_LC_TAG=gcad500 \
  OCARANDU_LC_GATE_PROBE=data/publichearingbr/lc_stream/$T/probe_tokens_novel.npz \
  OCARANDU_LC_CONDITIONS="gcad:1.0@0.5" uv run python scripts/longcontext_speaker_generation.py

# retrieve-and-regenerate. The threshold is fit on fold-A participants 200-219 and never re-fit;
# evaluation on 0-199 and 220-499. rag:none (no trigger) reproduces the baseline byte for byte.
env $LLAMA $FOLD_A OCARANDU_LC_N=220 OCARANDU_RAG_N_EVAL=200 OCARANDU_RAG_N_CAL=20 \
  OCARANDU_RAG_ARMS=probe,none uv run python scripts/lc_retrieve_regen.py
env $LLAMA $FOLD_A OCARANDU_LC_N=100000 OCARANDU_RAG_EVAL_FROM=220 OCARANDU_RAG_N_EVAL=280 \
  OCARANDU_RAG_CAL_FROM=200 OCARANDU_RAG_N_CAL=20 OCARANDU_RAG_TAG=${T}_rag_ext \
  OCARANDU_RAG_ARMS=probe uv run python scripts/lc_retrieve_regen.py
uv run python - <<'PY'
# merge 0-199 and 220-499 into one run; rag:none is byte-identical to the lc_xfitA baseline and is dropped
import json
from pathlib import Path
B = Path("data/publichearingbr/longcontext_generation")
out = B / "llama_31_8b_instruct_rag480"
out.mkdir(exist_ok=True)
seen = set()
with open(out / "generations.jsonl", "w", encoding="utf-8") as f:
    for tag in ("llama_31_8b_instruct_rag", "llama_31_8b_instruct_rag_ext"):
        for line in open(B / tag / "generations.jsonl", encoding="utf-8"):
            r = json.loads(line)
            k = (r["condition"], r["sample_id"], r["speaker"])
            if k in seen or r["condition"] == "rag:none":
                continue
            seen.add(k)
            f.write(line if line.endswith("\n") else line + "\n")
PY

Q="$QWEN $FOLD_A OCARANDU_LC_N=100000 OCARANDU_RAG_CAL_FROM=200 OCARANDU_RAG_N_CAL=20 \
   OCARANDU_RAG_TAG=qwen3_4b_instruct_2507_rag480 OCARANDU_RAG_ARMS=probe"
env $Q OCARANDU_RAG_EVAL_FROM=0 OCARANDU_RAG_N_EVAL=200 uv run python scripts/lc_retrieve_regen.py
env $Q OCARANDU_RAG_EVAL_FROM=220 OCARANDU_RAG_N_EVAL=280 uv run python scripts/lc_retrieve_regen.py
