"""Teacher-forced re-read of long-context generations: residual vectors and next-token statistics.

Each stored generation of longcontext_speaker_generation.py is re-read by the unsteered model after its
prompt (rebuilt from build_items() with the plain prompt, so the OCARANDU_LC_* settings must match the
generation run). Labels are exact by construction:
  token  each content word (>=4 letters, not a stop word, framing verbs excluded): novel = absent from
         the prompt's content words; each number: novel = not among the prompt's numbers. The vector is
         read at the position before the word's first token.
  line   at the position before each parsed opinion: referee containment (ungrounded = < 0.3) and whether
         the opinion has a novel content word or number.
  item   at the last prompt token: whole-output labels (any ungrounded opinion, fabricated number).
Each read position also stores next-token entropy, max probability, the probability of the token written,
and the copy mass (probability on tokens that occur in the participant's turns or name).

Usage: OCARANDU_MODEL_REPO=<hf repo> uv run python scripts/lc_stream_extract.py   (one GPU)
Env: OCARANDU_LS_TAGS (default <model_tag>_lc_xfitA,<model_tag>_lc_xfitB), OCARANDU_LS_CONDITIONS
  (default baseline,halluc:1.0,random0:1.0), OCARANDU_LS_LAYERS (default 10,14,20),
  OCARANDU_LS_MAX_ROWS (0 = all), OCARANDU_LS_OUT (output suffix), OCARANDU_DEVICE (default cuda:0).
Output: data/publichearingbr/lc_stream/<model_tag>[_<out>]/: tokens / lines / items .npy
  ([rows, layers, hidden] fp16) with matching .jsonl rows, and meta.json. The .npy files are
  over-allocated; only the first n_tokens / n_lines rows given in meta.json are valid.
"""
import json
import math
import os
import re
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")
# build_items must enumerate every matched participant, so each stored row finds its prompt
os.environ.setdefault("OCARANDU_LC_N", "100000")
os.environ.setdefault("OCARANDU_LC_PER_HEARING", "100")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attribution_generation import content_words, norm  # noqa: E402
from content_hallucination_generation import DEVICE, MODEL_TAG, NUMBER_RE, REPO_ID, ROOT  # noqa: E402
from longcontext_speaker_generation import LINE_PREFIX, PROMPTS, build_items, parse_opinions  # noqa: E402
from attribution_generation import containment  # noqa: E402

LC_DIR = ROOT / "data" / "publichearingbr" / "longcontext_generation"
TAGS = os.environ.get("OCARANDU_LS_TAGS", f"{MODEL_TAG}_lc_xfitA,{MODEL_TAG}_lc_xfitB").split(",")
CONDITIONS = os.environ.get("OCARANDU_LS_CONDITIONS", "baseline,halluc:1.0,random0:1.0").split(",")
LAYERS = [int(x) for x in os.environ.get("OCARANDU_LS_LAYERS", "10,14,20").split(",")]
MAX_ROWS = int(os.environ.get("OCARANDU_LS_MAX_ROWS", "0"))
OUT_TAG = os.environ.get("OCARANDU_LS_OUT", "lc_stream")
OUT_DIR = ROOT / "data" / "publichearingbr" / "lc_stream" / (MODEL_TAG + ("" if OUT_TAG == "lc_stream" else "_" + OUT_TAG))
WORD_RE = re.compile(r"\S+")
MARKER_RE = re.compile(r"^\(?\d+[.)]$")  # "1." / "2)" list markers are not numbers
# Novelty is judged against the whole prompt (turns + instruction), so the words of the
# instruction, the name and the cargo count as in-source.
# FRAMING: third-person reporting verbs of the summary register. They are never in the
# source, so they are excluded from the novelty label; rows keep a `framing` flag.
FRAMING = set("""acredita acha afirma alega alerta aponta apoia argumenta avalia cita cobra comenta compara concorda
conclui condena considera constata contesta critica declara defende demonstra denuncia descreve destaca diz discorda
elogia enfatiza entende esclarece espera evidencia exige explica expressa frisa indica informa insiste julga lamenta
lembra manifesta menciona mostra nega observa opina pede pondera propoe propõe questiona reafirma reclama recomenda
reconhece reforca reforça registra reitera relata relembra ressalta salienta sinaliza solicita sublinha sugere sustenta
teme valoriza vê ve expressou apontou afirmou defendeu destacou ressaltou considerou criticou sugeriu propos propôs
questionou reconheceu lembrou observou avaliou argumentou alertou apoiou elogiou manifestou mencionou concordou""".split())


def chat_ids(tok, user):
    text = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def word_tokens(tok, gen):
    """-> (token ids, [(word, first_token_index, char_start)]) for the generation, aligned by offsets."""
    enc = tok(gen, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    out = []
    for m in WORD_RE.finditer(gen):
        a, b = m.span()
        first = next((i for i, (s, e) in enumerate(offs) if e > a and s < b and e > s), None)
        if first is not None:
            out.append((m.group(0), first, a))
    return ids, out


def line_starts(gen):
    """-> [(opinion, char offset where its text begins, after the list marker)]."""
    ops = parse_opinions(gen)
    starts, pos = [], 0
    for line in gen.splitlines(keepends=True):
        m = LINE_PREFIX.match(line)
        body = line[m.end():] if m else line
        lead = len(body) - len(body.lstrip(' "“”'))
        clean = LINE_PREFIX.sub("", line).strip().strip('"“”')
        if clean in ops:
            starts.append((clean, pos + (m.end() if m else 0) + lead))
        pos += len(line)
    return starts


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(REPO_ID)
    rows = []
    for tag in TAGS:
        for line in open(LC_DIR / tag / "generations.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["condition"] in CONDITIONS:
                r["run"] = tag
                rows.append(r)
    if MAX_ROWS:
        rows = rows[:MAX_ROWS]
    print(f"{len(rows)} generations from {TAGS} ({CONDITIONS}); layers {LAYERS}; out {OUT_DIR}", file=sys.stderr, flush=True)

    hearings = {r["sample_id"] for r in rows}
    items = {(it["sample_id"], it["speaker"]): it for it in build_items(tok, hearings)}
    missing = [(r["sample_id"], r["speaker"]) for r in rows if (r["sample_id"], r["speaker"]) not in items]
    if missing:
        raise SystemExit(f"{len(missing)} generation rows have no rebuilt item, e.g. {missing[:3]}")

    model = AutoModelForCausalLM.from_pretrained(REPO_ID, dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    hidden = model.config.hidden_size
    vocab = model.lm_head.weight.shape[0]
    saved = {}

    def make_hook(li):
        def hook(module, inputs, output):
            saved[li] = (output[0] if isinstance(output, tuple) else output)[0].detach()
        return hook

    for li, layer in enumerate(LAYERS):
        model.model.layers[layer].register_forward_hook(make_hook(li))

    # upper bounds for the memmaps; the number of rows actually written goes to meta.json
    n_tok_max = sum(len(r["generation"].split()) for r in rows) + 10
    n_line_max = sum(max(1, r["n_opinions"]) for r in rows) + 10
    tokens = np.lib.format.open_memmap(OUT_DIR / "tokens.npy", mode="w+", dtype=np.float16, shape=(n_tok_max, len(LAYERS), hidden))
    lines_ = np.lib.format.open_memmap(OUT_DIR / "lines.npy", mode="w+", dtype=np.float16, shape=(n_line_max, len(LAYERS), hidden))
    items_ = np.lib.format.open_memmap(OUT_DIR / "items.npy", mode="w+", dtype=np.float16, shape=(len(rows), len(LAYERS), hidden))
    f_tok = open(OUT_DIR / "tokens.jsonl", "w", encoding="utf-8")
    f_line = open(OUT_DIR / "lines.jsonl", "w", encoding="utf-8")
    f_item = open(OUT_DIR / "items.jsonl", "w", encoding="utf-8")
    n_tok = n_line = 0

    for i, r in enumerate(rows):
        it = items[(r["sample_id"], r["speaker"])]
        user = PROMPTS["plain"].format(nome=it["nome"], cargo=it["cargo"], falas=it["falas"])
        p_ids = chat_ids(tok, user)
        gen = r["generation"]
        g_ids, words = word_tokens(tok, gen)
        if not g_ids:
            continue
        n_p, n_g = len(p_ids), len(g_ids)
        ids = torch.tensor([p_ids + g_ids], device=DEVICE)
        src_cw = content_words(user)  # the whole prompt: turns + instruction
        src_nums = {n.strip() for n in NUMBER_RE.findall(user)}  # the regex may keep a trailing space
        src_tok = torch.zeros(vocab, dtype=torch.bool, device=DEVICE)
        src_tok[torch.tensor(sorted(set(tok(it["falas"], add_special_tokens=False)["input_ids"] + tok(it["nome"], add_special_tokens=False)["input_ids"])), device=DEVICE)] = True
        with torch.no_grad():
            out = model(ids, logits_to_keep=n_g + 1)  # logits for positions n_p-1 .. n_p+n_g-1
            logp = torch.log_softmax(out.logits[0].float(), dim=-1)  # [n_g+1, vocab]
            prob = logp.exp()
            ent = -(prob * logp).sum(-1)
            maxp = prob.max(-1).values
            copy_mass = prob[:, src_tok].sum(-1)
            actual = torch.tensor(g_ids, device=DEVICE)
            p_actual = prob[:-1].gather(1, actual[:, None])[:, 0]  # position k predicts g_ids[k]
        # k-th logit row = position n_p-1+k, which predicts generated token k
        def stats_at(k):
            return dict(entropy=float(ent[k]), maxp=float(maxp[k]), copy_mass=float(copy_mass[k]),
                        p_actual=float(p_actual[k]) if k < n_g else None)
        def vec_at(pos, arr, idx):
            for li in range(len(LAYERS)):
                arr[idx, li] = saved[li][pos].float().cpu().numpy().astype(np.float16)

        base = {"row": i, "run": r["run"], "condition": r["condition"], "sample_id": r["sample_id"], "speaker": r["speaker"]}
        # token level: content words and numbers, vector at the position before the word's first token
        n_novel = n_content = 0
        for w, first, a in words:
            if MARKER_RE.match(w):
                continue
            cw = content_words(w)
            is_num = bool(NUMBER_RE.search(w))
            if not cw and not is_num:
                continue
            framing = bool(cw) and cw <= FRAMING
            novel_word = bool(cw) and not (cw <= src_cw) and not framing
            nums = {n.strip() for n in NUMBER_RE.findall(w)}
            novel_num = is_num and not (nums <= src_nums)
            if cw and not framing:
                n_content += 1
                n_novel += int(novel_word)
            vec_at(n_p + first - 1, tokens, n_tok)
            f_tok.write(json.dumps({**base, "idx": n_tok, "word": w, "char": a, "tok": first,
                                    "is_content": bool(cw), "framing": framing, "novel": novel_word,
                                    "is_number": is_num, "novel_number": novel_num,
                                    **stats_at(first)}, ensure_ascii=False) + "\n")
            n_tok += 1
        # line level: position before the first token of each opinion's text
        first_tok_of_char = {a: first for _, first, a in words}
        for op, start in line_starts(gen):
            first = first_tok_of_char.get(start)
            if first is None:
                # marker glued to the text ("1.Opinião"): take the token containing that char
                enc = tok(gen, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
                first = next((j for j, (s, e) in enumerate(enc) if s <= start < e), None)
                if first is None:
                    continue
            ocw = content_words(op) - FRAMING
            vec_at(n_p + first - 1, lines_, n_line)
            f_line.write(json.dumps({**base, "idx": n_line, "opinion": op, "tok": first,
                                     "containment": containment(op, r["chunks"]),
                                     "has_novel_word": not (ocw <= src_cw), "n_novel_words": len(ocw - src_cw),
                                     "n_content_words": len(ocw),
                                     "has_novel_number": not ({n.strip() for n in NUMBER_RE.findall(op)} <= src_nums),
                                     **stats_at(first)}, ensure_ascii=False) + "\n")
            n_line += 1
        # item level: last prompt token
        vec_at(n_p - 1, items_, i)
        f_item.write(json.dumps({**base, "idx": i, "n_prompt_tokens": n_p, "n_gen_tokens": n_g,
                                 "coverage": r["coverage"], "low_share": r["low_share"], "any_ungrounded": bool(r["low_share"]),
                                 "fabricated_number": r["fabricated_number"], "n_opinions": r["n_opinions"],
                                 "n_content_words": n_content, "n_novel_words": n_novel,
                                 "mean_copy_mass_gen": float(copy_mass[1:].mean()), "mean_entropy_gen": float(ent[1:].mean()),
                                 **stats_at(0)}, ensure_ascii=False) + "\n")
        if (i + 1) % 100 == 0:
            for fh in (f_tok, f_line, f_item):
                fh.flush()
            print(f"  {i + 1}/{len(rows)}  tokens {n_tok}  lines {n_line}", file=sys.stderr, flush=True)

    for fh in (f_tok, f_line, f_item):
        fh.close()
    tokens.flush(); lines_.flush(); items_.flush()
    json.dump({"model": REPO_ID, "tags": TAGS, "conditions": CONDITIONS, "layers": LAYERS,
               "n_tokens": n_tok, "n_lines": n_line, "n_items": len(rows), "hidden": hidden},
              open(OUT_DIR / "meta.json", "w"), indent=1)
    print(f"wrote {OUT_DIR}: {n_tok} token rows, {n_line} line rows, {len(rows)} item rows "
          f"(memmaps are sized {n_tok_max}/{n_line_max}; read n_tokens/n_lines from meta.json)", file=sys.stderr)


if __name__ == "__main__":
    main()
