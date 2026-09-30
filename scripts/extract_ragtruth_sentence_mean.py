"""Sentence-mean residual activations on RAGTruth summarisation, for any model.

One teacher-forced pass per RAGTruth Summary response (prompt and source article as
the user turn, the response as the assistant turn; inputs cut at 6,000 tokens). For
each spaCy sentence of the response it saves the mean residual vector over the
sentence's tokens at a few layers, its filler / content-bearing class, and its gold
label: 1 if at least half of some human hallucination span's characters fall inside
the sentence (ocarandu.data.sentence_labels.sentence_gold_label). Responses of all six
RAGTruth generators are used (the label is about the text, not who wrote it), with the
official train/test split kept on each row.

Usage: CUDA_VISIBLE_DEVICES=0 OCARANDU_MODEL_REPO=<hf repo> uv run python scripts/extract_ragtruth_sentence_mean.py
Env: OCARANDU_MODEL_REPO (default meta-llama/Llama-3.1-8B-Instruct), OCARANDU_DEVICE (default
cuda:0), OCARANDU_RT_LAYERS (default 8,10,12,14,16,18,20), OCARANDU_RT_MAX (responses, 0 = all).
Output: data/ragtruth_sentence/<model_tag>/: sent_mean.npy (float16 [N, n_layers, hidden]; only
the first meta.json n_rows rows are filled), index.jsonl, meta.json, done.txt. Resumable.
"""
import json
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np
import torch

from ocarandu.data.ragtruth import load_summarization_subset
from ocarandu.data.sentence_labels import segment_sentences, sentence_gold_label

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = os.environ.get("OCARANDU_MODEL_REPO", "meta-llama/Llama-3.1-8B-Instruct")
DEVICE = os.environ.get("OCARANDU_DEVICE", "cuda:0")
MODEL_TAG = REPO_ID.split("/")[-1].lower().replace(".", "").replace("-", "_")
OUT_DIR = ROOT / "data" / "ragtruth_sentence" / MODEL_TAG
LAYERS = [int(x) for x in os.environ.get("OCARANDU_RT_LAYERS", "8,10,12,14,16,18,20").split(",")]
MAX_EXAMPLES = int(os.environ.get("OCARANDU_RT_MAX", "0"))  # 0 = all


def chat_ids(tok, user, response):
    if tok.chat_template:
        text_ctx = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
    else:
        text_ctx = f"<s>[INST] {user.strip()} [/INST]"
    ctx = tok(text_ctx, add_special_tokens=False)["input_ids"]
    enc = tok(response, add_special_tokens=False, return_offsets_mapping=True)
    return ctx + enc["input_ids"], len(ctx), enc["offset_mapping"]


def main():
    from nnsight import LanguageModel

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = load_summarization_subset()
    if MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]
    # pre-segment so the memmap can be sized exactly
    # sentence_gold_label takes (start, end) tuples
    seg = [(ex, segment_sentences(ex.response), tuple((sp.start, sp.end) for sp in ex.spans)) for ex in examples]
    n_sent = sum(len(s) for _, s, _ in seg)
    print(f"{len(examples)} responses -> {n_sent} sentences; layers {LAYERS}", file=sys.stderr, flush=True)

    lm = LanguageModel(REPO_ID, device_map=DEVICE, dispatch=True, torch_dtype=torch.bfloat16)
    tok = lm.tokenizer
    H = lm.config.hidden_size
    acts_path = OUT_DIR / "sent_mean.npy"
    done_path = OUT_DIR / "done.txt"
    start = int(done_path.read_text()) if done_path.exists() and acts_path.exists() else 0
    acts = (np.lib.format.open_memmap(acts_path, mode="r+") if start
            else np.lib.format.open_memmap(acts_path, mode="w+", dtype=np.float16, shape=(n_sent, len(LAYERS), H)))
    index_path = OUT_DIR / "index.jsonl"
    if start:
        kept = [json.loads(l) for l in open(index_path)]
        kept = [k for k in kept if k["response_i"] < start]
        with index_path.open("w") as f:
            for k in kept:
                f.write(json.dumps(k) + "\n")
        row = len(kept)
    else:
        row = 0
    f = index_path.open("a")

    for ri, (ex, sents, spans) in enumerate(seg):
        if ri < start:
            continue
        user = f"{ex.prompt}\n\n{ex.source_info}"
        ids, n_ctx, offsets = chat_ids(tok, user, ex.response)
        if len(ids) > 6000:
            ids = ids[:6000]  # very long sources: sentences past the cut get no row (below)
        outs = {}
        with torch.no_grad(), lm.trace(torch.tensor([ids], device=lm.device)):
            for li, layer in enumerate(LAYERS):
                outs[li] = lm.model.layers[layer].output[0].save()
        hs = [(outs[li][0] if outs[li].dim() == 3 else outs[li]).detach().float().cpu().numpy() for li in range(len(LAYERS))]
        for si, sent in enumerate(sents):
            tok_idx = [n_ctx + j for j, (a, b) in enumerate(offsets) if a < sent.end and b > sent.start]
            tok_idx = [t for t in tok_idx if t < len(ids)]
            if not tok_idx:
                continue
            for li in range(len(LAYERS)):
                acts[row, li] = hs[li][tok_idx].mean(0)
            f.write(json.dumps({"row": row, "response_i": ri, "response_id": ex.response_id, "source_id": ex.source_id,
                                "gen_model": ex.model, "split": ex.split, "sent_i": si,
                                "label": sentence_gold_label(sent, spans), "filler": sent.filler_label,
                                "n_tokens": len(tok_idx)}) + "\n")
            row += 1
        if (ri + 1) % 100 == 0:
            f.flush(); acts.flush(); done_path.write_text(str(ri + 1))
            print(f"  {ri + 1}/{len(seg)} responses, {row} sentences", file=sys.stderr, flush=True)
    f.close(); acts.flush(); done_path.write_text(str(len(seg)))
    json.dump({"layers": LAYERS, "model": REPO_ID, "n_rows": row, "n_alloc": n_sent}, open(OUT_DIR / "meta.json", "w"))
    print(f"wrote {OUT_DIR} ({row} sentence rows)")


if __name__ == "__main__":
    main()
