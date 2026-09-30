#!/usr/bin/env bash
# Blind paired judgement and routing. `build` writes the rater sets; each set's header holds the full
# rating instructions. One rater per set reads only sets/set-NN.md and writes raw/set-NN.jsonl, one JSON
# object per item: {"id", "support": sustentada|nao_sustentada|nao_da_para_dizer, "form_issue": 0|1,
# "comment"}. The key linking items to arms goes to <round>_key/, which raters never read.
# `analyse` reads the raters' files. Needs 02 and 04. CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
STEP=${1:-build}   # build | analyse

if [ "$STEP" = build ]; then
  B="uv run python scripts/build_synth3_sets.py"
  $B --arms baseline=llama_31_8b_instruct_lc_xfitA:baseline gcad=llama_31_8b_instruct_gcad500:gcad:1.0@0.5 \
     --name synth3_gcad500 --blocks-per-set 20 --calib 4 --replicate 20 --seed 20260926
  $B --arms baseline=llama_31_8b_instruct_lc_xfitA:baseline rag=llama_31_8b_instruct_rag480:rag:probe \
     --name synth3_rag480 --blocks-per-set 20 --calib 4 --replicate 20 --seed 20260928
  $B --arms baseline=qwen3_4b_instruct_2507_lc_xfitA:baseline rag=qwen3_4b_instruct_2507_rag480:rag:probe \
     --name synth3_qwen_rag480 --blocks-per-set 20
  exit 0
fi

# retrieve-and-regenerate: rates by arm, exact McNemar on participants, calibration against the
# benchmark's labels, kappa. The Qwen3-4B sets were rated twice, by Sonnet (raw/) and by Opus (raw_opus/).
OCARANDU_SYNTH3_NAME=synth3_rag480 uv run python scripts/synth3_results.py
OCARANDU_SYNTH3_NAME=synth3_qwen_rag480 uv run python scripts/synth3_results.py
OCARANDU_SYNTH3_NAME=synth3_qwen_rag480 OCARANDU_SYNTH3_RAW=raw_opus uv run python scripts/synth3_results.py

# routing: withhold what a detector ranks highest, on the benchmark's human labels and on the
# baseline arm of synth3_gcad500 (Llama-3.1's own summaries)
uv run python scripts/mitigation_headroom.py
uv run python scripts/routing_fresh.py
