# Quem disse isso?

Code for the paper *Quem disse isso? Catching What Open Language Models Put in Speakers' Mouths When
Summarising Brazilian Public Hearings* (1º Desafio de Dados do Instituto Kunumi, 2026), and for the
portal at <https://ramironb.github.io/quemdisseisso.io/>.

Every number in the paper is produced by a script in `scripts/`. `reproduce/` holds the commands, in
the order they were run. Results and data are not versioned. The scripts write them to `data/`.

## Setup

```bash
uv sync                                          # Python 3.11, locked environment
uv run python -m spacy download en_core_web_sm   # sentence splitting for RAGTruth
export HF_HOME=/path/to/hf_cache
huggingface-cli download unicamp-dl/PublicHearingBR --repo-type dataset \
  --revision 2f84a44bc34df483e25c987f0ff86caad0ab3433
git clone https://github.com/ParticleMedia/RAGTruth vendor/ragtruth
```

If PublicHearingBR is stored somewhere other than `$HF_HOME`, set `OCARANDU_PHBR_DIR` to the folder that holds
`PublicHearingBR_NLI.jsonl` and `PublicHearingBR_LDS.jsonl`.

Models: `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen2.5-7B-Instruct` and
`NousResearch/Llama-2-7b-chat-hf` (an ungated copy of `meta-llama/Llama-2-7b-chat-hf`). Everything runs on one 48 GB
GPU. Select it with `CUDA_VISIBLE_DEVICES`; the scripts use `OCARANDU_DEVICE`, which defaults to `cuda:0`. Probes and
statistics run on CPU.

## Layout

| path | contents |
|---|---|
| `src/ocarandu/` | dataset loaders, probe utilities |
| `scripts/` | one script per experiment or analysis; they import each other, so keep them in one folder |
| `reproduce/` | `00`–`07`: the commands behind the paper, in order |
| `app/portal/` | the portal, a static site; its data comes from `scripts/export_portal_data.py` |
| `tests/` | `uv run pytest` |

## Paper to code

| paper | scripts | step |
|---|---|---|
| Dataset: speaker headers restored to the evidence | `build_tagged_evidence.py` | 00 |
| Opinion-level probe, content-word coverage, NLI, zero-shot from English | `extract_publichearingbr_activations.py`, `run_publichearingbr_probe.py`, `extract_ragtruth_sentence_mean.py`, `crosslingual_probe.py`, `nli_verifier.py` | 01 |
| RAGTruth summarisation test | `extract_ragtruth_sentence_mean.py`, `ragtruth_external_eval.py` | 01 |
| Per-participant summaries (cross-fit; baseline, steered and random-direction arms) | `longcontext_speaker_generation.py`, `longcontext_rescore.py`, `longcontext_pick_alpha.py` | 02 |
| Naive whole-transcript pipeline | `fulltranscript_generation.py`, `fulltranscript_lines.py` | 02 |
| Rule-based corruptions | `probe_fresh_corruptions.py`, `probe_fresh_stats.py`, `nli_verifier.py` | 03 |
| Token-level probe, one token ahead | `lc_stream_extract.py`, `lc_stream_stats.py` | 03 |
| Retrieve-and-regenerate | `lc_retrieve_regen.py` | 04 |
| Blind paired judgement | `build_synth3_sets.py`, `synth3_results.py` | 05 |
| Routing | `mitigation_headroom.py`, `routing_fresh.py` | 05 |
| Metadata use | `identity_swap.py`, `identity_swap_analysis.py`, `attribution_generation.py`, `attribution_generation_stats.py` | 06 |
| Credibility of a title | `title_swap.py`, `title_swap_analysis.py`, `title_swap_decisions.py`, `title_axis_analysis.py` | 06 |
| Omission from the article | `derive_speaker_attributes.py`, `extract_lds_audit.py`, `lds_audit_stage2.py`, `lds_omission_relabel.py` | 00, 06 |
| Portal | `export_demo_success.py`, `export_portal_data.py` | 07 |

## Notes

- Splits are by hearing. Decoding is greedy, and random directions use fixed seeds.
- The blind raters were LLM agents (Claude Sonnet; Claude Opus for the second rating of the Qwen3-4B sets), one per
  set. Each read only its set file, whose header holds the rating instructions, and the key linking items to arms was
  kept in a directory the raters never read.

## License

MIT, see `LICENSE`.
