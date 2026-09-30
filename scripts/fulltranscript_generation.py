"""Whole-transcript summarisation (the benchmark's original task), with optional steering.

The model reads a whole hearing transcript and lists the participants and the
opinions each expressed, the setting of the dataset's own GPT-4 baseline, scored
against the human `metadados`.

  prompt   = the transcript (its first OCARANDU_FT_CTX_TOKENS tokens) + "list the
             participants and, one line per opinion, `Name: opinion in one short
             third-person sentence`"
  referee  = every generated line is parsed into (name, opinion); the name is
             matched to a transcript speaker (header names, title-stripped, exact
             or token-subset match; among several matches, the one who spoke most).
             Then, with 100-word windows of each speaker's turns in the text the
             model saw:
               grounded        matched, containment in the named speaker's own
                               windows >= 0.3
               misattributed   matched, < 0.3 in the named speaker's windows but
                               >= 0.3 in another speaker's (right words, wrong speaker)
               unsupported     matched, < 0.3 everywhere
               unknown speaker no transcript speaker matches the name
             plus fabricated numbers and recall of the hearing's human gold: share
             of gold participants named, and share of gold opinions with >= 50% of
             their content words in an opinion attributed to the matching
             participant (soft: mean fraction).
Steering is applied at generated positions only, unless OCARANDU_FT_PROMPT_STEER=1.
Direction, layer and hearing hold-out come from content_hallucination_generation.py
(the hold-out, OCARANDU_CONTENT_HOLDOUT=1, is required).

Env: OCARANDU_MODEL_REPO, OCARANDU_ACT_DIR, OCARANDU_GEN_LAYER (shared with
content_hallucination_generation.py); OCARANDU_FT_SPLIT=dev|test|all|everything (dev;
"all" = dev + test, "everything" = every LDS hearing, baseline only), OCARANDU_FT_N (40),
OCARANDU_FT_CTX_TOKENS (40000), OCARANDU_FT_MAXNEW (700), OCARANDU_FT_CONDITIONS
(baseline,halluc:1.0,random0:1.0), OCARANDU_FT_TAG, OCARANDU_FT_HEARINGS (comma list of
hearing ids outside the direction-fit half), OCARANDU_FT_PROMPT_STEER (0).
Output: data/publichearingbr/fulltranscript_generation/<model_tag>[_<tag>]/generations.jsonl + summary.csv
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
from build_tagged_evidence import name_match, norm as norm_name  # noqa: E402
from content_hallucination_generation import (  # noqa: E402
    DEVICE, HOLDOUT, LAYER, MODEL_TAG, REPO_ID, ROOT, SEED,
    directions, fabricated_numbers, hearing_split, random_direction,
)
from extract_lds_audit import hearing_turns, load_hearings  # noqa: E402

SPLIT = os.environ.get("OCARANDU_FT_SPLIT", "dev")  # dev|test|all, or "everything" (all 206 LDS hearings, baseline only)
N_ITEMS = int(os.environ.get("OCARANDU_FT_N", "40"))
MAX_NEW = int(os.environ.get("OCARANDU_FT_MAXNEW", "700"))
CTX_TOKENS = int(os.environ.get("OCARANDU_FT_CTX_TOKENS", "40000"))
WINDOW_WORDS = 100
CONDITIONS = os.environ.get("OCARANDU_FT_CONDITIONS", "baseline,halluc:1.0,random0:1.0").split(",")
PROMPT_STEER = os.environ.get("OCARANDU_FT_PROMPT_STEER", "0") == "1"
OUT_DIR = ROOT / "data" / "publichearingbr" / "fulltranscript_generation" / (
    MODEL_TAG + (("_" + os.environ["OCARANDU_FT_TAG"]) if os.environ.get("OCARANDU_FT_TAG") else ""))

PROMPT = (
    "A seguir está a transcrição de uma audiência pública da Câmara dos Deputados.\n\n{transcricao}\n\n"
    "Com base na transcrição acima, liste os participantes que se manifestaram e as opiniões que cada um expressou. "
    "Escreva uma linha por opinião, exatamente no formato:\n"
    "Nome do participante: opinião em uma frase curta na terceira pessoa.\n"
    "Inclua até 4 opiniões por participante e não inclua nada além dessas linhas."
)
LINE = re.compile(r"^\s*(?:[-•*–]|\d+[.)])?\s*\**([^:\n]{3,90}?)\**\s*:\s*(.{15,})$")
TITLES = re.compile(r"(?i)^(o |a |os |as )?(sr\.?|sra\.?|senhor|senhora|deputad[oa]( federal)?|dep\.|presidente|relator[a]?|ministr[oa]|secretári[oa]|dr\.?|dra\.?|prof\.?|professor[a]?|vereador[a]?|senador[a]?)\s+")
PARTICLES = {"dos", "das", "del", "van", "von", "de", "da", "do", "e"}


def tokens_of(name):
    return {t for t in norm_name(name).split() if len(t) > 2 and t not in PARTICLES}


def clean_name(s):
    s = re.sub(r"\(.*?\)", "", s).strip(" -–*\"'")
    for _ in range(3):
        s = TITLES.sub("", s).strip()
    return s


def match_speaker(name, speakers):
    n = clean_name(name)
    if not n:
        return None
    hits = [s for s in speakers if name_match(n, s)]
    if not hits:
        tn = tokens_of(n)
        hits = [s for s in speakers if tn and (tn <= tokens_of(s) or tokens_of(s) <= tn)]
    if not hits:
        return None
    return max(hits, key=lambda s: speakers[s]["chars"])  # the one who spoke most


def windows_of(text, n=WINDOW_WORDS):
    w = text.split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)] or [text]


HEADER_LINE = re.compile(r"^\s*(?:\d+[.)]\s*)?\**\s*([^:\n*]{3,90}?)\s*\**\s*:\s*\**\s*$")
BULLET = re.compile(r"^\s*(?:[-•*–]|\d+[.)])\s*(.+?)\s*$")


def parse_lines(gen):
    """Parse `Name: opinion` one-liners and `**Name:**` headers followed by a
    bullet list into (name, opinion) pairs."""
    out, current = [], None
    for line in gen.splitlines():
        if not line.strip():
            continue
        h = HEADER_LINE.match(line)
        if h:
            current = h.group(1).strip()
            continue
        m = LINE.match(line)  # `1. Nome: opinião` -- numbered one-liners are the common case
        if m and len(clean_name(m.group(1)).split()) <= 6:
            name, op = m.group(1).strip(" *"), m.group(2).strip().strip('"“”')
            if len(op.split()) >= 4 and not re.match(r"(?i)^(formato|exemplo|nome do participante)", name):
                out.append((name, op))
            continue
        b = BULLET.match(line)
        if b and current:
            op = b.group(1).strip().strip('"“”*')
            if len(op.split()) >= 4:
                out.append((current, op))
    return out


def score(gen, speakers, gold):
    """speakers: {name: {'chars', 'windows'}}; gold: metadados envolvidos."""
    ops = []
    for name, op in parse_lines(gen):
        spk = match_speaker(name, speakers)
        cov_own = containment(op, speakers[spk]["windows"]) if spk else None
        best_other, cov_other = None, 0.0
        for s, d in speakers.items():
            if s != spk:
                c = containment(op, d["windows"])
                if c > cov_other:
                    best_other, cov_other = s, c
        if spk is None:
            cls = "unknown_speaker"
        elif cov_own >= 0.3:
            cls = "grounded"
        elif cov_other >= 0.3:
            cls = "misattributed"
        else:
            cls = "unsupported"
        ops.append(dict(name=name, speaker=spk, text=op, cov_own=cov_own, cov_other=cov_other,
                        best_other=best_other, cls=cls))
    n = len(ops)
    share = lambda c: (sum(o["cls"] == c for o in ops) / n) if n else None  # noqa: E731
    named = {o["speaker"] for o in ops if o["speaker"]}
    # gold recall: participants and opinions
    gp_hit, go_hit, go_soft = [], [], []
    for part in gold:
        spk = match_speaker(part["nome"], speakers)
        gp_hit.append(spk is not None and spk in named)
        mine = [o["text"] for o in ops if o["speaker"] and spk and o["speaker"] == spk]
        for g in part.get("opinioes", []):
            gw = content_words(g)
            if not gw:
                continue
            frac = max((len(gw & content_words(t)) / len(gw) for t in mine), default=0.0)
            go_hit.append(frac >= 0.5)
            go_soft.append(frac)
    all_windows = [w for d in speakers.values() for w in d["windows"]]
    own = [o["cov_own"] for o in ops if o["cov_own"] is not None]
    return dict(opinions=ops, n_opinions=n, n_participants=len(named),
                share_grounded=share("grounded"), share_misattributed=share("misattributed"),
                share_unsupported=share("unsupported"), share_unknown_speaker=share("unknown_speaker"),
                coverage=(float(np.mean(own)) if own else None), abstained=(n == 0),
                fabricated_number=fabricated_numbers(" ".join(o["text"] for o in ops), all_windows) if ops else False,
                gold_participant_recall=(float(np.mean(gp_hit)) if gp_hit else None),
                gold_opinion_recall=(float(np.mean(go_hit)) if go_hit else None),
                gold_opinion_recall_soft=(float(np.mean(go_soft)) if go_soft else None))


def build_items(tok, allowed):
    lds = {h["id"]: h for h in load_hearings()}
    order = sorted(allowed)
    random.Random(SEED).shuffle(order)
    items = []
    for hid in order:
        h = lds.get(hid)
        if h is None:
            continue
        ids = tok(h["transcricao"], add_special_tokens=False).input_ids
        truncated = len(ids) > CTX_TOKENS
        seen = tok.decode(ids[:CTX_TOKENS]) if truncated else h["transcricao"]
        speakers = {}
        for name, _, text in hearing_turns(seen):
            d = speakers.setdefault(name, {"chars": 0, "texts": []})
            d["chars"] += len(text)
            d["texts"].append(text)
        for d in speakers.values():
            d["windows"] = windows_of(" ".join(d.pop("texts")))
        gold = [p for p in h["metadados"].get("envolvidos", []) if p.get("opinioes")]
        items.append(dict(sample_id=hid, text=seen, n_tokens=min(len(ids), CTX_TOKENS), full_tokens=len(ids),
                          truncated=truncated, speakers=speakers, gold=gold))
        if len(items) >= N_ITEMS:
            break
    print(f"items: {len(items)} hearings; median seen tokens {int(np.median([i['n_tokens'] for i in items]))}; "
          f"truncated: {sum(i['truncated'] for i in items)}; speakers/hearing median "
          f"{int(np.median([len(i['speakers']) for i in items]))}; gold participants median "
          f"{int(np.median([len(i['gold']) for i in items]))}", file=sys.stderr, flush=True)
    return items


def main():
    if not HOLDOUT:
        raise SystemExit("requires the hearing hold-out")
    fit_hearings, eval_pool, dev_h, test_h = hearing_split()
    if SPLIT == "everything":
        # whole-corpus baseline: with no direction applied the fit half is not a
        # hold-out concern, so any steered arm is refused
        if CONDITIONS != ["baseline"]:
            raise SystemExit("OCARANDU_FT_SPLIT=everything is baseline-only (fit hearings included)")
        allowed = sorted(h["id"] for h in load_hearings())
        print(f"everything: {len(allowed)} hearings, baseline only", file=sys.stderr, flush=True)
    else:
        allowed = {"dev": dev_h, "test": test_h, "all": eval_pool}[SPLIT]
    if os.environ.get("OCARANDU_FT_HEARINGS"):
        allowed = [int(x) for x in os.environ["OCARANDU_FT_HEARINGS"].split(",")]
        if any(x in fit_hearings for x in allowed):
            raise SystemExit("a requested hearing is in the direction-fit half; refusing")
    dirs, act_norm = directions(LAYER, fit_hearings)
    hidden = dirs.pop("hidden")
    for k, v in dirs.items():
        if np.isnan(v).any() or not (0.99 < np.linalg.norm(v) < 1.01):
            raise SystemExit(f"direction {k!r} invalid")
    tok = AutoTokenizer.from_pretrained(REPO_ID)
    items = build_items(tok, set(allowed))
    model = AutoModelForCausalLM.from_pretrained(REPO_ID, dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    state = {"dir": None, "amt": 0.0}

    def hook(module, inputs, output):
        if state["dir"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1 and not PROMPT_STEER:
            return output
        h2 = h + state["amt"] * act_norm * state["dir"]
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    model.model.layers[LAYER].register_forward_hook(hook)

    def generate(user):
        ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, return_tensors="pt")
        ids = ids["input_ids"] if not isinstance(ids, torch.Tensor) else ids
        ids = ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f = open(OUT_DIR / "generations.jsonl", "w")
    summary = []
    for cond in CONDITIONS:
        if cond == "baseline":
            dname, alpha = "baseline", 0.0
        else:
            dname, a = cond.split(":")
            alpha = float(a)
            m = re.fullmatch(r"random(\d+)", dname)
            if m:
                dirs[dname] = random_direction(int(m.group(1)), hidden)
            if dname not in dirs:
                print(f"  skip {cond}", file=sys.stderr)
                continue
        state["amt"] = alpha
        state["dir"] = None if alpha == 0.0 else torch.tensor(dirs[dname], dtype=torch.bfloat16, device=DEVICE)
        rows = []
        for i, it in enumerate(items):
            gen = generate(PROMPT.format(transcricao=it["text"]))
            row = {"condition": cond, "direction": dname, "alpha": alpha, "sample_id": it["sample_id"],
                   "n_tokens": it["n_tokens"], "full_tokens": it["full_tokens"], "truncated": it["truncated"],
                   "n_speakers": len(it["speakers"]), "n_gold_participants": len(it["gold"]),
                   "opinion": " ".join(o for p in it["gold"] for o in p["opinioes"]), "true_label": 0,
                   "chunks": [w for d in it["speakers"].values() for d2 in [d] for w in d2["windows"]][:400],
                   "generation": gen, "n_chars": len(gen), **score(gen, it["speakers"], it["gold"])}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{cond}] {i + 1}/{len(items)} hearing {it['sample_id']}: {row['n_opinions']} opinions, "
                  f"grounded {row['share_grounded']}, misattr {row['share_misattributed']}, unsupported "
                  f"{row['share_unsupported']}, unknown {row['share_unknown_speaker']}, gold-op recall "
                  f"{row['gold_opinion_recall']}", file=sys.stderr, flush=True)

        def m(k):
            v = [r[k] for r in rows if r[k] is not None]
            return float(np.mean(v)) if v else float("nan")
        s = dict(condition=cond, n=len(rows), n_opinions=m("n_opinions"), n_participants=m("n_participants"),
                 share_grounded=m("share_grounded"), share_misattributed=m("share_misattributed"),
                 share_unsupported=m("share_unsupported"), share_unknown_speaker=m("share_unknown_speaker"),
                 mean_coverage=m("coverage"), gold_participant_recall=m("gold_participant_recall"),
                 gold_opinion_recall=m("gold_opinion_recall"), gold_opinion_recall_soft=m("gold_opinion_recall_soft"),
                 fabricated_number_rate=m("fabricated_number"))
        summary.append(s)
        print(f"{cond:14s} " + " ".join(f"{k}={v:.3f}" for k, v in s.items() if k not in ("condition", "n")),
              file=sys.stderr, flush=True)
    f.close()
    import pandas as pd
    pd.DataFrame(summary).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"wrote {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
