"""Per-layer residual-stream activations for every PublicHearingBR NLI opinion.

Each opinion with four evidence chunks is teacher-forced after PROMPT (speaker's
name and title, the chunks) as the assistant turn. Two vectors per opinion, at
every decoder layer:
  opinion_mean  mean over the opinion's (response) tokens; the in-domain probe
                (run_publichearingbr_probe.py) and the steering directions are
                fit on it.
  name_last     activation at the last token of the speaker's name in the prompt
                (where Ferrando et al. 2025 read entity recognition).

Usage: CUDA_VISIBLE_DEVICES=0 OCARANDU_MODEL_REPO=<hf repo> uv run python scripts/extract_publichearingbr_activations.py
Env: OCARANDU_MODEL_REPO (default NousResearch/Llama-2-7b-chat-hf), OCARANDU_DEVICE
(default cuda:0), OCARANDU_TAGGED=1 to use the speaker-tagged chunks of build_tagged_evidence.py.
Output: data/publichearingbr/activations_<model_tag>[_tagged]/, where <model_tag> is the repo
name lowercased with "." dropped and "-" -> "_" (plain "activations[_tagged]" for Llama-2-7b-chat):
opinion_mean.npy and name_last.npy (float16 [N, n_layers, hidden]), index.jsonl (one row per
opinion), done.txt (rows completed; a rerun resumes from it).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from nnsight import LanguageModel

from ocarandu.data.publichearingbr import load_nli_opinions

# Default: NousResearch's ungated mirror of meta-llama/Llama-2-7b-chat-hf (same weights and tokenizer).
REPO_ID = os.environ.get("OCARANDU_MODEL_REPO", os.environ.get("OCARANDU_LLAMA2_REPO", "NousResearch/Llama-2-7b-chat-hf"))
_TAG = REPO_ID.split("/")[-1].lower().replace(".", "").replace("-", "_")
TAGGED = os.environ.get("OCARANDU_TAGGED", "0") == "1"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "publichearingbr" / (
    ("activations" if "llama_2_7b_chat" in _TAG else f"activations_{_TAG}") + ("_tagged" if TAGGED else ""))
TAGGED_PATH = Path(__file__).resolve().parents[1] / "data" / "publichearingbr" / "tagged_evidence.jsonl"
DEVICE = os.environ.get("OCARANDU_DEVICE", "cuda:0")
PROGRESS_EVERY = 100

PROMPT = (
    "Com base nos trechos de uma transcrição de audiência pública a seguir, "
    "avalie a seguinte opinião atribuída a {nome} ({cargo}).\n\n{chunks}"
)
if os.environ.get("OCARANDU_TAGGED", "0") == "1":
    PROMPT = (
        "Com base nos trechos de uma transcrição de audiência pública a seguir (cada trecho começa com o nome "
        "de quem estava com a palavra, entre colchetes), avalie a seguinte opinião atribuída a {nome} ({cargo}).\n\n{chunks}"
    )


def llama2_chat_text(user_content: str) -> str:
    """Llama-2-chat single-turn format of meta-llama's chat_template:
    BOS + '[INST] ' + stripped content + ' [/INST]'. Hard-coded because the
    mirror's tokenizer has no chat_template; the token ids match the original."""
    return f"<s>[INST] {user_content.strip()} [/INST]"


def chat_text(tokenizer, user_content: str) -> str:
    """Use the tokenizer's own chat template when it has one (Llama-3), else
    the hard-coded Llama-2 format (the mirror ships none)."""
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template([{"role": "user", "content": user_content}], add_generation_prompt=True, tokenize=False)
    return llama2_chat_text(user_content)


def build_ids(tokenizer, op):
    """Returns (full_ids [1, T], context_len, name_token_span (start, end_exclusive))."""
    user = PROMPT.format(nome=op.person_name, cargo=op.person_role, chunks="\n\n".join(op.context_chunks))
    text = chat_text(tokenizer, user)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    context_ids = torch.tensor([enc["input_ids"]])
    # locate the name: first occurrence after the anchor phrase
    anchor = "atribuída a "
    a = text.find(anchor)
    name_start = text.find(op.person_name, a + len(anchor)) if a >= 0 else -1
    if name_start < 0:
        raise ValueError(f"name not found in prompt text: {op.person_name!r}")
    name_end = name_start + len(op.person_name)
    tok_idx = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > name_start and s < name_end and e > s]
    if not tok_idx:
        raise ValueError(f"no tokens aligned to name: {op.person_name!r}")
    response_ids = torch.tensor([tokenizer(op.opinion, add_special_tokens=False)["input_ids"]])
    full_ids = torch.cat([context_ids, response_ids], dim=1)
    return full_ids, context_ids.shape[1], (tok_idx[0], tok_idx[-1] + 1)


def main():
    opinions = [o for o in load_nli_opinions() if len(o.context_chunks) == 4]
    if TAGGED:
        from dataclasses import replace as _replace
        tagged = {}
        for line in open(TAGGED_PATH):
            d = json.loads(line)
            tagged[(d["sample_id"], d["nome"], d["opinion"])] = tuple(d["tagged_chunks"])
        opinions = [_replace(o, context_chunks=tagged[(o.sample_id, o.person_name, o.opinion)]) for o in opinions
                    if (o.sample_id, o.person_name, o.opinion) in tagged]
        print(f"TAGGED evidence for {len(opinions)} opinions", file=sys.stderr)
    n = len(opinions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done_path = OUT_DIR / "done.txt"
    start = int(done_path.read_text()) if done_path.exists() else 0
    print(f"{n} opinions; resuming at {start}; device={DEVICE}; model={REPO_ID}; out={OUT_DIR}", file=sys.stderr)

    lm = LanguageModel(REPO_ID, device_map=DEVICE, dtype=torch.bfloat16)
    tokenizer = lm.tokenizer
    n_layers = len(lm.model.layers)
    hidden = lm.config.hidden_size

    # sanity: the literal "<s>" must be parsed as the BOS special token, not as text
    ids = tokenizer(chat_text(tokenizer, "olá"), add_special_tokens=False)["input_ids"]
    if tokenizer.bos_token_id is not None:
        assert ids[0] == tokenizer.bos_token_id, "BOS special token not parsed from the chat text"

    mode = "r+" if start > 0 else "w+"
    opinion_mean = np.lib.format.open_memmap(OUT_DIR / "opinion_mean.npy", mode=mode, dtype=np.float16, shape=(n, n_layers, hidden))
    name_last = np.lib.format.open_memmap(OUT_DIR / "name_last.npy", mode=mode, dtype=np.float16, shape=(n, n_layers, hidden))
    index_path = OUT_DIR / "index.jsonl"
    if start > 0 and index_path.exists():
        # rows written after the last flushed checkpoint would be duplicated on resume: drop them
        kept = [l for l in index_path.read_text().splitlines() if l.strip() and json.loads(l)["row"] < start]
        index_path.write_text("\n".join(kept) + ("\n" if kept else ""))
        assert len(kept) == start, f"index has {len(kept)} rows below checkpoint {start}"
    index_f = open(index_path, "a" if start > 0 else "w")

    for i in range(start, n):
        op = opinions[i]
        full_ids, context_len, (ns, ne) = build_ids(tokenizer, op)
        full_ids = full_ids.to(DEVICE)
        outs = {}
        with torch.no_grad(), lm.trace(full_ids):
            for l in range(n_layers):  # plain loop: nnsight's trace context does not see comprehension scopes
                outs[l] = lm.model.layers[l].output[0].save()
        stacked = torch.stack([outs[l][0] if outs[l].dim() == 3 else outs[l] for l in range(n_layers)], dim=0)  # [L, T, H]
        opinion_mean[i] = stacked[:, context_len:, :].float().mean(dim=1).cpu().numpy().astype(np.float16)
        name_last[i] = stacked[:, ne - 1, :].float().cpu().numpy().astype(np.float16)
        index_f.write(json.dumps({
            "row": i, "sample_id": op.sample_id, "nome": op.person_name, "cargo": op.person_role,
            "opinion": op.opinion, "label": int(op.is_hallucination),
            "n_context_tokens": int(context_len), "n_response_tokens": int(full_ids.shape[1] - context_len),
            "name_token_span": [ns, ne],
        }, ensure_ascii=False) + "\n")
        if (i + 1) % PROGRESS_EVERY == 0 or i + 1 == n:
            opinion_mean.flush(); name_last.flush(); index_f.flush()
            done_path.write_text(str(i + 1))
            print(f"  {i + 1}/{n}", file=sys.stderr)
    index_f.close()
    print(f"wrote {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
