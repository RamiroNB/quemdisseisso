"""Stage 1 of the LDS audit: residual-stream embeddings of transcript turns and article sentences.

For each hearing of PublicHearingBR_LDS.jsonl (next to the NLI file; `transcricao` and `materia`, the
published Agência Câmara article), mean-pooled embeddings are saved for every transcript turn (speaker
from the HEADER regex of build_tagged_evidence.py; one independent forward pass per turn) and every
article sentence (heuristic PT-BR splitter; one forward pass per article, truncated at 3000 tokens, each
sentence pooled over its token span). The matching is done on CPU by lds_audit_stage2.py.

Usage: OCARANDU_MODEL_REPO=<hf repo> uv run python scripts/extract_lds_audit.py   (one GPU; resumable)
Env: OCARANDU_LDS_LAYERS (default 6,8,...,24), OCARANDU_LDS_MAX_TURN_CHARS (3000),
  OCARANDU_LDS_MAX_TOKENS (512 per turn), OCARANDU_LDS_MAX_TURNS_PER_HEARING (600),
  OCARANDU_LDS_MAX_HEARINGS (0 = all), OCARANDU_DEVICE (default cuda:0).
Output: data/publichearingbr/lds_audit/<model_tag>/: turn_mean.npy and sent_mean.npy
  ([rows, layers, hidden] fp16) with turn_index.jsonl / sent_index.jsonl, and the resume
  checkpoints done_turns.txt / done_sents.txt.
"""
import json
import os
import re
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tagged_evidence import HEADER, parse_header  # noqa: E402
from ocarandu.data.publichearingbr import DEFAULT_NLI_PATH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LDS_PATH = Path(str(DEFAULT_NLI_PATH).replace("_NLI", "_LDS"))
REPO_ID = os.environ.get("OCARANDU_MODEL_REPO", "meta-llama/Llama-3.1-8B-Instruct")
DEVICE = os.environ.get("OCARANDU_DEVICE", "cuda:0")
MODEL_TAG = REPO_ID.split("/")[-1].lower().replace(".", "").replace("-", "_")
OUT_DIR = ROOT / "data" / "publichearingbr" / "lds_audit" / MODEL_TAG
LAYERS = [int(x) for x in os.environ.get("OCARANDU_LDS_LAYERS", "6,8,10,12,14,16,18,20,22,24").split(",")]
MAX_TURN_CHARS = int(os.environ.get("OCARANDU_LDS_MAX_TURN_CHARS", "3000"))  # ~p95; longer turns truncated, not dropped
MAX_TOKENS = int(os.environ.get("OCARANDU_LDS_MAX_TOKENS", "512"))
MAX_TURNS_PER_HEARING = int(os.environ.get("OCARANDU_LDS_MAX_TURNS_PER_HEARING", "600"))  # bounds one outlier hearing with thousands of turns
MAX_HEARINGS = int(os.environ.get("OCARANDU_LDS_MAX_HEARINGS", "0"))  # 0 = all; small for smoke tests
PROGRESS_EVERY = 200

# PT-BR sentence heuristic: split at sentence-ending punctuation followed by whitespace
# and a capital, quote or parenthesis. No abbreviation list; adequate for news prose.
_SENT_END = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀ-Ú"“(])')


def split_materia_sentences(text):
    spans = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def hearing_turns(transcricao):
    """-> list of (speaker_name, chair_role, text) for each turn, in order."""
    tn = re.sub(r"\s+", " ", transcricao)
    heads = list(HEADER.finditer(tn))
    turns = []
    for i, m in enumerate(heads):
        name, meta = parse_header(m)
        end = heads[i + 1].start() if i + 1 < len(heads) else len(tn)
        text = tn[m.end():end].strip()
        if text:
            turns.append((name, meta["chair_role"], text))
    if len(turns) > MAX_TURNS_PER_HEARING:
        print(f"    capping {len(turns)} -> {MAX_TURNS_PER_HEARING} turns (outlier hearing)", file=sys.stderr)
        turns = turns[:MAX_TURNS_PER_HEARING]
    return turns


def load_hearings():
    hearings = [json.loads(l) for l in open(LDS_PATH)]
    return hearings[:MAX_HEARINGS] if MAX_HEARINGS else hearings


def extract_layers(lm, ids_1d):
    """One forward pass; returns [len(LAYERS), T, H] float32 tensor on CPU."""
    outs = {}
    with torch.no_grad(), lm.trace(torch.tensor([ids_1d], device=lm.device)):
        for li, layer in enumerate(LAYERS):
            outs[li] = lm.model.layers[layer].output[0].save()
    return torch.stack([(outs[li][0] if outs[li].dim() == 3 else outs[li]) for li in range(len(LAYERS))], dim=0).float().cpu()


def run_turns(lm, tok, hearings):
    done_path = OUT_DIR / "done_turns.txt"
    index_path = OUT_DIR / "turn_index.jsonl"
    # pre-segment so the memmap can be sized exactly
    all_turns = []  # (hearing_id, speaker, chair_role, text)
    for h in hearings:
        for speaker, chair_role, text in hearing_turns(h["transcricao"]):
            all_turns.append((h["id"], speaker, chair_role, text))
    n = len(all_turns)
    H = lm.config.hidden_size
    print(f"turns: {n} total across {len(hearings)} hearings; layers {LAYERS}", file=sys.stderr, flush=True)

    start = int(done_path.read_text()) if done_path.exists() else 0
    mean = (np.lib.format.open_memmap(OUT_DIR / "turn_mean.npy", mode="r+") if start
            else np.lib.format.open_memmap(OUT_DIR / "turn_mean.npy", mode="w+", dtype=np.float16, shape=(n, len(LAYERS), H)))
    if start and index_path.exists():
        kept = [l for l in index_path.read_text().splitlines() if l.strip() and json.loads(l)["row"] < start]
        index_path.write_text("\n".join(kept) + ("\n" if kept else ""))
        assert len(kept) == start, f"turn_index has {len(kept)} rows below checkpoint {start}"
    f = open(index_path, "a" if start else "w")

    for i in range(start, n):
        hearing_id, speaker, chair_role, text = all_turns[i]
        enc = tok(text[:MAX_TURN_CHARS], add_special_tokens=True, truncation=True, max_length=MAX_TOKENS)
        ids = enc["input_ids"]
        stacked = extract_layers(lm, ids)  # [L, T, H]
        mean[i] = stacked.mean(dim=1).numpy().astype(np.float16)
        f.write(json.dumps({
            "row": i, "hearing_id": hearing_id, "speaker": speaker, "chair_role": chair_role,
            "n_tokens": len(ids), "char_len": len(text), "truncated": len(text) > MAX_TURN_CHARS,
        }, ensure_ascii=False) + "\n")
        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == n:
            mean.flush(); f.flush()
            done_path.write_text(str(i + 1))
            print(f"  turns {i + 1}/{n}", file=sys.stderr, flush=True)
    f.close()


def run_sentences(lm, tok, hearings):
    done_path = OUT_DIR / "done_sents.txt"
    index_path = OUT_DIR / "sent_index.jsonl"
    # pre-segment so the memmap can be sized exactly
    seg = [(h, split_materia_sentences(h["materia"])) for h in hearings]
    n_sent = sum(len(s) for _, s in seg)
    H = lm.config.hidden_size
    print(f"materia sentences: {n_sent} total across {len(hearings)} hearings; layers {LAYERS}", file=sys.stderr, flush=True)

    start = int(done_path.read_text()) if done_path.exists() else 0
    mean = (np.lib.format.open_memmap(OUT_DIR / "sent_mean.npy", mode="r+") if start
            else np.lib.format.open_memmap(OUT_DIR / "sent_mean.npy", mode="w+", dtype=np.float16, shape=(n_sent, len(LAYERS), H)))
    if start and index_path.exists():
        kept = [l for l in index_path.read_text().splitlines() if l.strip() and json.loads(l)["row"] < start]
        index_path.write_text("\n".join(kept) + ("\n" if kept else ""))
        assert len(kept) == start, f"sent_index has {len(kept)} rows below checkpoint {start}"
    f = open(index_path, "a" if start else "w")

    row = 0
    for h, spans in seg:
        if row + len(spans) <= start:
            row += len(spans)
            continue
        enc = tok(h["materia"], add_special_tokens=True, truncation=True, max_length=3000, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        stacked = extract_layers(lm, ids)  # [L, T, H]
        for si, (s, e) in enumerate(spans):
            if row < start:
                row += 1
                continue
            tok_idx = [j for j, (a, b) in enumerate(offsets) if b > s and a < e and b > a]
            if not tok_idx:
                row += 1
                continue
            mean[row] = stacked[:, tok_idx, :].mean(dim=1).numpy().astype(np.float16)
            f.write(json.dumps({
                "row": row, "hearing_id": h["id"], "sent_i": si, "start": s, "end": e,
                "text": h["materia"][s:e],
            }, ensure_ascii=False) + "\n")
            row += 1
            if row % PROGRESS_EVERY == 0 or row == n_sent:
                mean.flush(); f.flush()
                done_path.write_text(str(row))
                print(f"  sentences {row}/{n_sent}", file=sys.stderr, flush=True)
    f.close()


def main():
    from nnsight import LanguageModel

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hearings = load_hearings()
    print(f"{len(hearings)} hearings; device={DEVICE}; model={REPO_ID}; out={OUT_DIR}", file=sys.stderr, flush=True)

    lm = LanguageModel(REPO_ID, device_map=DEVICE, dispatch=True, torch_dtype=torch.bfloat16)
    tok = lm.tokenizer

    run_turns(lm, tok, hearings)
    run_sentences(lm, tok, hearings)
    print(f"wrote {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
