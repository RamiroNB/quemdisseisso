"""Probe activations for generated opinions and rule-corrupted copies of them.

Every non-abstained opinion of the chosen generation runs and arms (longcontext_generation)
is kept with its evidence, and one rule corrupts the claim so the variant is unsupported
by construction:
  polarity  a negation inserted before / removed from a modal or copula, or an antonym
            swap (deve -> não deve, aumento -> redução); the content words are
            unchanged, so a content-word referee cannot see it.
  number    one number changed (another quantity, year or percentage).
  entity    one multi-word proper-noun span replaced by one from another hearing.
  scope     a universal qualifier appended (", em todo o país", ", sem exceção", ...).
  cross     the opinion replaced by one from another hearing, credited to this speaker.
Each variant is teacher-forced after the in-domain probe's prompt
(extract_publichearingbr_activations.PROMPT: name, cargo and, as chunks, the four 100-word
windows of the participant's turns that best contain the original opinion), so within a
pair only the claim differs. The mean residual over the claim tokens is saved at several
layers; rows.jsonl also has content-word containment (shown / all windows) and a
fabricated-number flag. Analysis: probe_fresh_stats.py.

Usage: CUDA_VISIBLE_DEVICES=0 uv run python scripts/probe_fresh_corruptions.py
Env: OCARANDU_MODEL_REPO (default meta-llama/Llama-3.1-8B-Instruct), OCARANDU_DEVICE (default cuda:0);
  OCARANDU_PF_TAGS        generation runs (default <model_tag>_lc_xfitA,<model_tag>_lc_xfitB)
  OCARANDU_PF_CONDITIONS  arms (default baseline,halluc:1.0,random0:1.0)
  OCARANDU_PF_LAYERS      default 10,12,14,16,18
  OCARANDU_PF_SHARE       share of originals that also get scope/cross variants (default 0.34)
  OCARANDU_PF_VARIANTS    "original" to skip the corruptions (default all)
  OCARANDU_PF_MAX_ROWS    cap on original opinions (0 = all)
  OCARANDU_PF_OUT         output tag (default probe_fresh)
Output: data/publichearingbr/probe_fresh/<model_tag>[_<out>]/{acts.npy,rows.jsonl,meta.json}
"""
import json
import os
import random
import re
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attribution_generation import containment, content_words  # noqa: E402
from content_hallucination_generation import DEVICE, MODEL_TAG, NUMBER_RE, REPO_ID, ROOT, fabricated_numbers  # noqa: E402
from extract_publichearingbr_activations import PROMPT as PROBE_PROMPT, chat_text  # noqa: E402

LC_DIR = ROOT / "data" / "publichearingbr" / "longcontext_generation"
TAGS = os.environ.get("OCARANDU_PF_TAGS", f"{MODEL_TAG}_lc_xfitA,{MODEL_TAG}_lc_xfitB").split(",")
CONDITIONS = os.environ.get("OCARANDU_PF_CONDITIONS", "baseline,halluc:1.0,random0:1.0").split(",")
LAYERS = [int(x) for x in os.environ.get("OCARANDU_PF_LAYERS", "10,12,14,16,18").split(",")]
SHARE = float(os.environ.get("OCARANDU_PF_SHARE", "0.34"))
MAX_ROWS = int(os.environ.get("OCARANDU_PF_MAX_ROWS", "0"))
OUT_TAG = os.environ.get("OCARANDU_PF_OUT", "probe_fresh")
ONLY_ORIGINAL = os.environ.get("OCARANDU_PF_VARIANTS", "all") == "original"
OUT_DIR = ROOT / "data" / "publichearingbr" / "probe_fresh" / (MODEL_TAG + ("" if OUT_TAG == "probe_fresh" else "_" + OUT_TAG))
SEED = 20260921
N_CHUNKS = 4

# --- polarity: negation of a modal/copula, or one antonym swap (both directions) ---
MODALS = "deve devem é são está estão pode podem precisa precisam tem têm há vai vão foi foram deveria deveriam poderia poderiam".split()
ANTONYMS = [("aumento", "redução"), ("aumentar", "reduzir"), ("aumenta", "reduz"), ("mais", "menos"), ("maior", "menor"),
            ("maiores", "menores"), ("favorável", "contrário"), ("favoráveis", "contrários"), ("positivo", "negativo"),
            ("positiva", "negativa"), ("necessário", "desnecessário"), ("necessária", "desnecessária"),
            ("possível", "impossível"), ("eficaz", "ineficaz"), ("eficazes", "ineficazes"), ("adequado", "inadequado"),
            ("adequada", "inadequada"), ("suficiente", "insuficiente"), ("suficientes", "insuficientes"), ("melhor", "pior"),
            ("melhorar", "piorar"), ("fortalecer", "enfraquecer"), ("ampliar", "restringir"), ("incluir", "excluir"),
            ("público", "privado"), ("pública", "privada"), ("públicos", "privados"), ("públicas", "privadas"),
            ("importante", "irrelevante"), ("essencial", "dispensável"), ("fundamental", "secundário"),
            ("urgente", "adiável"), ("apoia", "rejeita"), ("apoiar", "rejeitar"), ("aprovação", "rejeição"),
            ("aprovar", "rejeitar"), ("garantir", "negar"), ("proteger", "desproteger"), ("legal", "ilegal"),
            ("constitucional", "inconstitucional"), ("justo", "injusto"), ("justa", "injusta"), ("segura", "insegura"),
            ("seguro", "inseguro"), ("viável", "inviável"), ("compatível", "incompatível"), ("transparente", "opaco"),
            ("obrigatório", "opcional"), ("obrigatória", "opcional"), ("permanente", "temporário"), ("permitir", "proibir"),
            ("permite", "proíbe"), ("proibir", "permitir"), ("proíbe", "permite"), ("antes", "depois"), ("acima", "abaixo")]
ANT = {}
for a, b in ANTONYMS:
    ANT.setdefault(a, b)
    ANT.setdefault(b, a)
SCOPE_TAILS = [", em todo o país", ", sem exceção", ", para todos os brasileiros", ", em qualquer circunstância",
               ", de forma imediata e obrigatória", ", em todos os estados"]
CAP_SPAN = re.compile(r"\b(?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wáéíóúâêôãõç\-]+)(?:\s+(?:d[aeo]s?\s+)?[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wáéíóúâêôãõç\-]+)*")
PRONOUNS = {"Ele", "Ela", "Eles", "Elas", "O", "A", "Os", "As", "Não", "Segundo", "Para", "Também", "Além"}


def keep_case(src, rep):
    if src.isupper():
        return rep.upper()
    if src[:1].isupper():
        return rep[:1].upper() + rep[1:]
    return rep


def corrupt_polarity(text, rng):
    """One negation inserted before / removed from a modal or copula, or one antonym swap; string splicing, so
    hyphens and punctuation are untouched."""
    cands = []
    for m in re.finditer(r"\w+", text, re.UNICODE):
        tl = m.group(0).lower()
        if tl in MODALS:
            pre = text[:m.start()].rstrip()
            if pre.lower().endswith(("não", "nao")):
                cands.append(("drop", m))
            else:
                cands.append(("neg", m))
        elif tl in ANT and not m.group(0)[:1].isupper():  # never inside a proper noun ("Fazenda Pública")
            cands.append(("ant", m))
    if not cands:
        return None
    kind, m = rng.choice(cands)
    if kind == "drop":
        pre = text[:m.start()].rstrip()
        cut = len(pre) - 3  # "não" / "nao"
        return text[:cut].rstrip() + " " + text[m.start():], "não " + m.group(0) + "→" + m.group(0)
    if kind == "neg":
        return text[:m.start()] + "não " + text[m.start():], m.group(0) + "→não " + m.group(0)
    rep = keep_case(m.group(0), ANT[m.group(0).lower()])
    return text[:m.start()] + rep + text[m.end():], f"{m.group(0)}→{rep}"


def corrupt_number(text, rng):
    nums = []
    for m in NUMBER_RE.finditer(text):
        before = text[m.start() - 1] if m.start() else " "
        after = text[m.start() + len(m.group(0).rstrip())] if m.start() + len(m.group(0).rstrip()) < len(text) else " "
        if before in "-/" or before.isalpha() or after in "/-" or after.isalpha():
            continue
        nums.append(m)
    if not nums:
        return None
    m = rng.choice(nums)
    s = m.group(0).rstrip()
    end = m.start() + len(s)
    core = re.sub(r"[^\d,.]", "", s).rstrip(".,")
    try:
        val = float(core.replace(".", "").replace(",", ".")) if ("," in core or core.count(".") > 1 or len(core.split(".")[-1]) == 3) else float(core)
    except ValueError:
        return None
    if 1900 <= val <= 2100 and float(val).is_integer():
        new = int(val) + rng.choice([-7, -5, -3, -2, 2, 3, 5, 7])
        rep = str(new)
    elif val <= 12:
        rep = str(int(val) + rng.choice([2, 3, 4, 5])) if val.is_integer() else f"{val * 3:.1f}".replace(".", ",")
    else:
        f = rng.choice([0.5, 1.5, 2.0, 3.0])
        rep = str(int(round(val * f))) if val.is_integer() else f"{val * f:.1f}".replace(".", ",")
    if s.rstrip().endswith("%"):
        rep += "%"
    if rep == s:
        return None
    return text[:m.start()] + rep + text[end:], f"{s}→{rep}"


def proper_spans(text, nome):
    own = {w.lower() for w in nome.split()}
    spans = []
    for m in CAP_SPAN.finditer(text):
        if m.start() == 0:
            continue
        words = [w for w in m.group(0).split() if w.lower() not in ("da", "de", "do", "das", "dos")]
        if not words or words[0] in PRONOUNS or any(w.lower() in own for w in words):
            continue
        if len(words) < 2:  # "Supremo Tribunal Federal", "Imposto de Renda", person names; never a lone "Estado"
            continue
        spans.append(m)
    return spans


def corrupt_entity(text, nome, pool, rng):
    spans = proper_spans(text, nome)
    if not spans or not pool:
        return None
    m = rng.choice(spans)
    rep = rng.choice(pool)
    if rep.lower() == m.group(0).lower():
        return None
    return text[:m.start()] + rep + text[m.end():], f"{m.group(0)}→{rep}"


def corrupt_scope(text, rng):
    t = text.rstrip()
    end = t[-1] if t and t[-1] in ".!?" else ""
    body = t[:-1] if end else t
    tail = rng.choice(SCOPE_TAILS)
    return body + tail + end, tail


def corrupt_cross(nome, foreign, rng):
    op = rng.choice(foreign)
    m = re.match(r"^(?:[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wáéíóúâêôãõç\-]+\s+)+(?=\w)", op)
    if m and m.group(0).split()[0] not in PRONOUNS:
        op = nome + " " + op[m.end():]
    return op, "cross"


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag in TAGS:
        for line in open(LC_DIR / tag / "generations.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["condition"] in CONDITIONS and not r["abstained"]:
                r["run"] = tag
                rows.append(r)
    # pools for entity / cross corruptions, keyed by hearing so the donor is always another hearing
    ent_by_h, op_by_h = {}, {}
    for r in rows:
        if r["condition"] != "baseline":
            continue
        for op in r["opinions"]:
            op_by_h.setdefault(r["sample_id"], []).append(op)
            for m in proper_spans(op, r["nome"]):
                ent_by_h.setdefault(r["sample_id"], []).append(m.group(0))
    hearings = sorted(op_by_h)

    def foreign(sid, table):
        out = []
        for h in hearings:
            if h != sid and h in table:
                out.extend(table[h])
        return out

    # build the variant list first (CPU), then one GPU pass
    variants = []
    n_orig = 0
    for r in rows:
        windows = r["chunks"]
        for j, op in enumerate(r["opinions"]):
            if MAX_ROWS and n_orig >= MAX_ROWS:
                break
            n_orig += 1
            best = sorted(range(len(windows)), key=lambda i: -containment(op, [windows[i]]))[:N_CHUNKS]
            chunks = [windows[i] for i in sorted(best)]
            base = dict(run=r["run"], condition=r["condition"], sample_id=r["sample_id"], speaker=r["speaker"],
                        sample=r.get("sample", 0), nome=r["nome"], cargo=r["cargo"], opinion_idx=j, original=op,
                        chunk_idx=sorted(best))
            pair_id = len(variants)
            variants.append({**base, "pair_id": pair_id, "variant": "original", "text": op, "change": ""})
            if ONLY_ORIGINAL:
                continue
            outs = [("polarity", corrupt_polarity(op, rng)), ("number", corrupt_number(op, rng)),
                    ("entity", corrupt_entity(op, r["nome"], foreign(r["sample_id"], ent_by_h), rng))]
            if rng.random() < SHARE:
                outs.append(("scope", corrupt_scope(op, rng)))
                outs.append(("cross", corrupt_cross(r["nome"], foreign(r["sample_id"], op_by_h), rng)))
            for kind, res in outs:
                if res is None or res[0] == op:
                    continue
                variants.append({**base, "pair_id": pair_id, "variant": kind, "text": res[0], "change": res[1]})
    for v in variants:
        v["containment_all"] = containment(v["text"], [w for w in rows_windows(rows, v)])
        v["containment_shown"] = containment(v["text"], [rows_windows(rows, v)[i] for i in v["chunk_idx"]])
        v["fabricated_number"] = fabricated_numbers(v["text"], rows_windows(rows, v))
    from collections import Counter
    print(f"{n_orig} original opinions from {len(rows)} generations; variants: {Counter(v['variant'] for v in variants)}",
          file=sys.stderr, flush=True)

    tok = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForCausalLM.from_pretrained(REPO_ID, dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    hidden = model.config.hidden_size
    saved = {}

    def make_hook(li):
        def hook(module, inputs, output):
            saved[li] = (output[0] if isinstance(output, tuple) else output)[0].detach()
        return hook

    for li, layer in enumerate(LAYERS):
        model.model.layers[layer].register_forward_hook(make_hook(li))

    acts = np.lib.format.open_memmap(OUT_DIR / "acts.npy", mode="w+", dtype=np.float16, shape=(len(variants), len(LAYERS), hidden))
    f = open(OUT_DIR / "rows.jsonl", "w", encoding="utf-8")
    ctx_cache = {}
    for i, v in enumerate(variants):
        key = (v["run"], v["condition"], v["sample_id"], v["speaker"], v["sample"], v["opinion_idx"])
        if key not in ctx_cache:
            wins = rows_windows(rows, v)
            user = PROBE_PROMPT.format(nome=v["nome"], cargo=v["cargo"], chunks="\n\n".join(wins[k] for k in v["chunk_idx"]))
            ctx_cache = {key: tok(chat_text(tok, user), add_special_tokens=False)["input_ids"]}  # one entry: an opinion's variants are consecutive
        c_ids = ctx_cache[key]
        r_ids = tok(v["text"], add_special_tokens=False)["input_ids"]
        ids = torch.tensor([c_ids + r_ids], device=DEVICE)
        with torch.no_grad():
            model(ids, logits_to_keep=1)
        for li in range(len(LAYERS)):
            acts[i, li] = saved[li][len(c_ids):].float().mean(0).cpu().numpy().astype(np.float16)
        f.write(json.dumps({**v, "idx": i, "n_context_tokens": len(c_ids), "n_response_tokens": len(r_ids)}, ensure_ascii=False) + "\n")
        if (i + 1) % 500 == 0:
            f.flush()
            print(f"  {i + 1}/{len(variants)}", file=sys.stderr, flush=True)
    f.close()
    acts.flush()
    json.dump({"model": REPO_ID, "tags": TAGS, "conditions": CONDITIONS, "layers": LAYERS, "n": len(variants),
               "n_original": n_orig, "share_scope_cross": SHARE, "seed": SEED, "hidden": hidden},
              open(OUT_DIR / "meta.json", "w"), indent=1)
    print(f"wrote {OUT_DIR}: {len(variants)} rows", file=sys.stderr)


_WIN = {}


def rows_windows(rows, v):
    key = (v["run"], v["condition"], v["sample_id"], v["speaker"], v["sample"])
    if not _WIN:
        for r in rows:
            _WIN[(r["run"], r["condition"], r["sample_id"], r["speaker"], r.get("sample", 0))] = r["chunks"]
    return _WIN[key]


if __name__ == "__main__":
    main()
