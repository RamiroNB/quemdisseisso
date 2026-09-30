"""Per-participant opinion extraction from long context, with steering and decoding-side competitors.

Item = a gold participant of an LDS hearing (metadados `envolvidos`, matched to a transcript speaker by
name) with at least MIN_FLOOR_CHARS of speech; per hearing, the OCARANDU_LC_PER_HEARING who spoke most.
The prompt holds that speaker's turns, in order, up to OCARANDU_LC_CTX_TOKENS tokens, and asks for up
to 5 opinions, one short third-person sentence per line. Referee (score_generation), per parsed opinion:
content-word containment in the best OCARANDU_LC_WINDOW_WORDS-word window of the text shown; coverage =
mean containment, low_share = share with containment < 0.3 (ungrounded); fabricated numbers; abstention;
gold_recall = share of gold opinions with >= 50% of their content words in the output (soft: mean share).

Direction, layer and hearing split come from content_hallucination_generation.py: the direction is fit
on the fit half and items come from the eval half (OCARANDU_LC_SPLIT=dev|test|all, default dev);
OCARANDU_LC_FOLD=B swaps the halves for cross-fitting (needs SPLIT=all). Steering acts on generated
positions only unless OCARANDU_LC_PROMPT_STEER=1.
Conditions (OCARANDU_LC_CONDITIONS; default baseline, halluc:1.0 and random0-2:1.0): baseline;
<direction>:<alpha>; copy:<beta> (CopyBias); cad:<lam> (ContextAware); gate:<alpha>@<tau> and
gcad:<lam>@<tau> (halluc direction or CAD only where the token probe in OCARANDU_LC_GATE_PROBE gives
P(novel) > tau); rgate:<alpha>@<rate> and rcad:<lam>@<rate> (the same at a random share of positions).

Usage: uv run python scripts/longcontext_speaker_generation.py   (one GPU; configured by env)
Env: OCARANDU_MODEL_REPO, OCARANDU_DEVICE, OCARANDU_ACT_DIR and OCARANDU_GEN_LAYER as in the sentence-mode
  script; OCARANDU_LC_PER_HEARING (2), OCARANDU_LC_N (100), OCARANDU_LC_MAXNEW (300), OCARANDU_LC_CTX_TOKENS
  (6000), OCARANDU_LC_WINDOW_WORDS (100), OCARANDU_LC_PROMPT=plain|grounded|quote|sourcewords,
  OCARANDU_LC_SAMPLE_K (0 = greedy; K > 0 = K samples per item at OCARANDU_LC_SAMPLE_TEMP, 0.7),
  OCARANDU_LC_HEARINGS (explicit hearing ids), OCARANDU_LC_LOOSE_MATCH=1 (token-subset name match),
  OCARANDU_LC_TAG (output suffix).
Output: data/publichearingbr/longcontext_generation/<model_tag>[_<tag>]/{generations.jsonl,summary.csv}.
Rows use the keys of the sentence-mode runs (generation, chunks = the windows, coverage, abstained,
fabricated_number, opinion = the gold opinions joined). Paired tests: longcontext_stats.py.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attribution_generation import ABSTAIN_RE, containment, content_words  # noqa: E402
from build_tagged_evidence import name_match, norm as norm_name  # noqa: E402
from content_hallucination_generation import (  # noqa: E402
    ACT_DIR, DEVICE, HOLDOUT, LAYER, MODEL_TAG, REPO_ID, ROOT, SEED,
    directions, fabricated_numbers, hearing_split, random_direction,
)
from extract_lds_audit import hearing_turns, load_hearings  # noqa: E402

SPLIT = os.environ.get("OCARANDU_LC_SPLIT", "dev")
# Cross-fitting: fold A fits the direction on the fit half and generates on the eval half; fold B
# swaps the halves, so every hearing gets text from a direction it did not help estimate. Fold B
# needs SPLIT=all (dev/test split only the fold-A eval half); alpha and layer are not re-chosen.
FOLD = os.environ.get("OCARANDU_LC_FOLD", "A")
PER_HEARING = int(os.environ.get("OCARANDU_LC_PER_HEARING", "2"))
N_ITEMS = int(os.environ.get("OCARANDU_LC_N", "100"))
MAX_NEW = int(os.environ.get("OCARANDU_LC_MAXNEW", "300"))
CTX_TOKENS = int(os.environ.get("OCARANDU_LC_CTX_TOKENS", "6000"))
WINDOW_WORDS = int(os.environ.get("OCARANDU_LC_WINDOW_WORDS", "100"))
MIN_FLOOR_CHARS = 1500
CONDITIONS = os.environ.get("OCARANDU_LC_CONDITIONS", "baseline,halluc:1.0,random0:1.0,random1:1.0,random2:1.0").split(",")
PROMPT_STEER = os.environ.get("OCARANDU_LC_PROMPT_STEER", "0") == "1"
PROMPT_KIND = os.environ.get("OCARANDU_LC_PROMPT", "plain")
LOOSE_MATCH = os.environ.get("OCARANDU_LC_LOOSE_MATCH", "0") == "1"  # off by default
OUT_DIR = ROOT / "data" / "publichearingbr" / "longcontext_generation" / (
    MODEL_TAG + (("_" + os.environ["OCARANDU_LC_TAG"]) if os.environ.get("OCARANDU_LC_TAG") else ""))

PROMPTS = {
    "plain": (
        "A seguir estão as falas de {nome} ({cargo}) em uma audiência pública da Câmara dos Deputados.\n\n"
        "{falas}\n\n"
        "Com base nas falas acima, liste até 5 opiniões expressas por {nome}, uma por linha, "
        "cada uma em uma única frase curta na terceira pessoa."
    ),
    "grounded": (
        "A seguir estão as falas de {nome} ({cargo}) em uma audiência pública da Câmara dos Deputados.\n\n"
        "{falas}\n\n"
        "Com base apenas nas falas acima, liste até 5 opiniões expressas por {nome}, uma por linha, "
        "cada uma em uma única frase curta na terceira pessoa. Use somente o que está escrito nas falas: "
        "não acrescente nomes, números, datas, instituições ou conclusões que não apareçam acima."
    ),
    # prompt competitors: ask for literal quotation, or for the source's own words
    "quote": (
        "A seguir estão as falas de {nome} ({cargo}) em uma audiência pública da Câmara dos Deputados.\n\n"
        "{falas}\n\n"
        "Com base nas falas acima, liste até 5 opiniões expressas por {nome}, uma por linha, "
        "cada uma em uma única frase curta na terceira pessoa, citando literalmente as palavras de {nome} sempre que possível."
    ),
    "sourcewords": (
        "A seguir estão as falas de {nome} ({cargo}) em uma audiência pública da Câmara dos Deputados.\n\n"
        "{falas}\n\n"
        "Com base nas falas acima, liste até 5 opiniões expressas por {nome}, uma por linha, "
        "cada uma em uma única frase curta na terceira pessoa. Use apenas palavras e expressões que apareçam nas falas acima, sem parafrasear."
    ),
}
# OCARANDU_LC_SAMPLE_K > 0 draws K samples per item at SAMPLE_TEMP (top-p 0.95) instead of one greedy
# decode; rows then carry `sample`. Default 0 = greedy.
SAMPLE_K = int(os.environ.get("OCARANDU_LC_SAMPLE_K", "0"))
SAMPLE_TEMP = float(os.environ.get("OCARANDU_LC_SAMPLE_TEMP", "0.7"))
FORMAT_TOKENS = "1. 2. 3. 4. 5.\n\n- • "  # list markers the copy-bias decoder must still be able to write


class CopyBias(LogitsProcessor):
    """Decoding-side competitor of the steering direction: +beta on the logit of every token that occurs in
    the prompt (the participant's own words) or in the list format; everything else is left alone."""

    def __init__(self, allowed, beta):
        self.allowed, self.beta = allowed, float(beta)

    def __call__(self, input_ids, scores):
        return scores + self.beta * self.allowed[: scores.shape[-1]].to(scores.dtype)
class ContextAware(LogitsProcessor):
    """Context-aware decoding (Shi et al. 2023) as a competitor of the steering direction.

    The next-token distribution is pushed away from what the model would say about this
    participant without the transcript in front of it:

        logits = (1 + lam) * logits(y | falas, instrução) - lam * logits(y | instrução)

    The second term comes from a parallel decode of the same prompt with `{falas}` empty,
    kept in its own KV cache: one extra forward pass per generated token over a short prompt.
    No vector is added to any representation; only the next-token scores are re-weighted.

    `applies(step)` gates it: always for `cad:<lam>`, on the pre-emptive probe for
    `gcad:<lam>@<tau>`, and on a seeded random mask at a matched rate for `rcad:<lam>@<rate>`.
    """

    def __init__(self, model, amateur_ids, lam, state, applies=None):
        self.model, self.lam, self.ids, self.state = model, float(lam), amateur_ids, state
        self.applies = applies or (lambda step: True)
        self.past, self.step, self.n_applied = None, 0, 0

    def __call__(self, input_ids, scores):
        fire = self.applies(self.step)  # decided before the amateur forward pass runs
        self.state["in_amateur"] = True
        try:
            with torch.no_grad():
                if self.past is None:
                    out = self.model(self.ids, use_cache=True)
                else:
                    out = self.model(input_ids[:, -1:], past_key_values=self.past, use_cache=True)
            self.past = out.past_key_values
            am = out.logits[:, -1, : scores.shape[-1]].to(scores.dtype)
        finally:
            self.state["in_amateur"] = False
        self.step += 1
        if not fire:
            return scores
        self.n_applied += 1
        return (1.0 + self.lam) * scores - self.lam * am


LINE_PREFIX = re.compile(r"^\s*(?:[-•*–]|\d+[.)]|\(\d+\))\s*")
# a preamble line ("Aqui estão 5 opiniões expressas por X:") is not an opinion; if kept,
# it would be scored as a low-coverage one
PREAMBLE = re.compile(r"(?i)^(aqui (estão|está|vão)|seguem|abaixo|a seguir|as (cinco|5|principais) opini|opiniões expressas)")


def windows_of(text, n=WINDOW_WORDS):
    w = text.split()
    return [" ".join(w[i:i + n]) for i in range(0, len(w), n)] or [text]


def parse_opinions(gen):
    ops = []
    for line in gen.splitlines():
        line = LINE_PREFIX.sub("", line).strip().strip('"“”')
        if len(line.split()) >= 4 and not line.endswith(":") and not PREAMBLE.match(line):
            ops.append(line)
    return ops or ([gen.strip()] if gen.strip() and not gen.strip().endswith(":") else [])


def is_abstention(gen, ops):
    """True when the output as a whole declines: nothing parseable, or a single line that
    matches ABSTAIN_RE. Matching the regex anywhere in the output would also fire on
    negations inside an opinion ("não é compatível com o horário")."""
    if not ops:
        return True
    return len(ops) == 1 and bool(ABSTAIN_RE.search(ops[0]))


def score_generation(gen, windows, gold):
    """Everything the referee computes from one generation; shared with
    longcontext_rescore.py so stored generations can be re-scored."""
    ops = parse_opinions(gen)
    abst = is_abstention(gen, ops)
    covs = [containment(o, windows) for o in ops] if not abst else []
    gen_words = content_words(" ".join(ops))
    gold_hits = [len(content_words(g) & gen_words) / max(1, len(content_words(g))) for g in gold]
    return {"opinions": ops, "n_opinions": len(ops), "abstained": abst,
            "coverage": (float(np.mean(covs)) if covs else None),
            "low_share": (float(np.mean([c < 0.3 for c in covs])) if covs else None),
            # numbers checked on the parsed opinions: "1." "2." list markers are not fabrications
            "fabricated_number": fabricated_numbers(" ".join(ops), windows) if not abst else False,
            "gold_recall": float(np.mean([h >= 0.5 for h in gold_hits])) if gold_hits else None,
            "gold_recall_soft": float(np.mean(gold_hits)) if gold_hits else None}


def build_items(tok, hearings_allowed):
    lds = {h["id"]: h for h in load_hearings()}
    rng = random.Random(SEED)
    order = sorted(hearings_allowed)
    rng.shuffle(order)
    items, n_matched, n_unmatched = [], 0, 0
    for hid in order:
        h = lds.get(hid)
        if h is None:
            continue
        by = {}
        for name, chair_role, text in hearing_turns(h["transcricao"]):
            by.setdefault(name, []).append(text)
        cands = []
        for part in h["metadados"].get("envolvidos", []):
            gold = [o for o in part.get("opinioes", []) if o and o.strip()]
            if not gold:
                continue
            match = [s for s in by if name_match(part["nome"], s)]
            if not match and LOOSE_MATCH:  # token-subset match (short name inside a longer full name)
                toks = lambda x: {t for t in norm_name(x).split() if len(t) > 2 and t not in {"dos", "das", "del", "van", "von"}}  # noqa: E731
                match = [s for s in by if toks(part["nome"]) and toks(part["nome"]) <= toks(s)]
            if not match:
                n_unmatched += 1
                continue
            n_matched += 1
            spk = max(match, key=lambda s: sum(len(t) for t in by[s]))
            floor = sum(len(t) for t in by[spk])
            if floor < MIN_FLOOR_CHARS:
                continue
            cands.append((floor, spk, part["nome"], part.get("cargo") or "participante", gold))
        cands.sort(reverse=True)
        for floor, spk, nome, cargo, gold in cands[:PER_HEARING]:
            # keep whole turns, in order, until the token budget is spent
            kept, used = [], 0
            for t in by[spk]:
                n = len(tok(t, add_special_tokens=False).input_ids)
                if used + n > CTX_TOKENS:
                    if not kept:  # one enormous turn: cut it
                        ids = tok(t, add_special_tokens=False).input_ids[:CTX_TOKENS]
                        kept.append(tok.decode(ids))
                        used = len(ids)
                    break
                kept.append(t)
                used += n
            falas = "\n\n".join(kept)
            items.append(dict(sample_id=hid, speaker=spk, nome=nome, cargo=cargo, gold=gold,
                              floor_chars=floor, ctx_tokens=used, falas=falas, windows=windows_of(falas)))
        if len(items) >= N_ITEMS:
            break
    print(f"gold participants matched to a transcript speaker: {n_matched}, unmatched: {n_unmatched}; "
          f"items: {len(items)} over {len({i['sample_id'] for i in items})} hearings; "
          f"median ctx tokens {int(np.median([i['ctx_tokens'] for i in items]))}", file=sys.stderr, flush=True)
    return items[:N_ITEMS]


def main():
    if not HOLDOUT:
        raise SystemExit("this experiment requires the hearing hold-out (OCARANDU_CONTENT_HOLDOUT=1)")
    fit_hearings, eval_pool, dev_h, test_h = hearing_split()
    if FOLD == "B":
        if SPLIT != "all":
            raise SystemExit("fold B has no dev/test split; use OCARANDU_LC_SPLIT=all")
        fit_hearings, eval_pool = set(eval_pool), sorted(fit_hearings)
        print(f"cross-fit fold B: direction from {len(fit_hearings)} hearings, generating on the other "
              f"{len(eval_pool)}", file=sys.stderr, flush=True)
    elif FOLD != "A":
        raise SystemExit("OCARANDU_LC_FOLD must be A or B")
    allowed = {"dev": dev_h, "test": test_h, "all": eval_pool}[SPLIT]
    # explicit hearing list (must lie outside the direction-fit half)
    if os.environ.get("OCARANDU_LC_HEARINGS"):
        allowed = [int(x) for x in os.environ["OCARANDU_LC_HEARINGS"].split(",")]
        bad = [x for x in allowed if x in fit_hearings]
        if bad:
            raise SystemExit(f"hearings {bad} are in the direction-fit half; refusing")
    dirs, act_norm = directions(LAYER, fit_hearings)
    hidden = dirs.pop("hidden")
    for k, v in dirs.items():
        if np.isnan(v).any() or not (0.99 < np.linalg.norm(v) < 1.01):
            raise SystemExit(f"direction {k!r} invalid (nan={np.isnan(v).any()}, norm={np.linalg.norm(v):.3f})")
    tok = AutoTokenizer.from_pretrained(REPO_ID)
    items = build_items(tok, set(allowed))
    if not items:
        raise SystemExit("no items built")
    model = AutoModelForCausalLM.from_pretrained(REPO_ID, dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    state = {"dir": None, "amt": 0.0, "copy": 0.0, "gate": None, "gate_n": 0, "gate_fired": 0,
             "rmask": None, "rpos": 0, "cad": None, "read_probe": False, "in_amateur": False, "last_p": 0.0}
    gate_probe = None
    if os.environ.get("OCARANDU_LC_GATE_PROBE"):
        # pre-emptive token probe saved by lc_stream_stats.py (probe_tokens_novel.npz): fit at this LAYER on
        # the residual before a content word, label = the word is absent from the source. "gate:<alpha>@<tau>"
        # adds the direction only at generated positions where P(next word novel) > tau; the probe reads
        # the tensor the hook modifies, before modification.
        z = np.load(os.environ["OCARANDU_LC_GATE_PROBE"])
        if LAYER not in list(z["layers"]):
            raise SystemExit(f"gate probe has layers {list(z['layers'])}, not {LAYER}")
        gate_probe = {k: torch.tensor(z[f"{k}_{LAYER}"], dtype=torch.float32, device=DEVICE) for k in ("mean", "scale", "coef")}
        gate_probe["intercept"] = float(z[f"intercept_{LAYER}"])

    def hook(module, inputs, output):
        if state.get("in_amateur"):  # the CAD counterfactual pass: never read, never steer
            return output
        if state["dir"] is None and not state.get("read_probe"):
            return output
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] > 1 and not PROMPT_STEER:  # prefill of the long prompt: leave it alone
            return output
        if state.get("read_probe"):
            # gcad: the same pre-emptive probe the token gate uses, read here and consumed by
            # the logits processor of this step (which runs after this forward pass)
            zc = (h[0, -1].float() - gate_probe["mean"]) / gate_probe["scale"]
            state["last_p"] = torch.sigmoid(zc @ gate_probe["coef"] + gate_probe["intercept"]).item()
            state["gate_n"] += 1
        if state["dir"] is None:
            return output
        if state["rmask"] is not None:
            # rgate: steer a seeded random subset of generated positions at a given rate, the
            # control for the gate's choice of positions
            i = state["rpos"]
            state["rpos"] += 1
            state["gate_n"] += 1
            if i >= len(state["rmask"]) or not state["rmask"][i]:
                return output
            state["gate_fired"] += 1
        elif state["gate"] is not None:
            zc = (h[0, -1].float() - gate_probe["mean"]) / gate_probe["scale"]
            p = torch.sigmoid(zc @ gate_probe["coef"] + gate_probe["intercept"]).item()
            state["gate_n"] += 1
            if p <= state["gate"]:
                return output
            state["gate_fired"] += 1
        h2 = h + state["amt"] * act_norm * state["dir"]
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    model.model.layers[LAYER].register_forward_hook(hook)

    def encode(user):
        if tok.chat_template:
            ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, return_tensors="pt")
            ids = ids["input_ids"] if not isinstance(ids, torch.Tensor) else ids
        else:
            ids = tok(f"<s>[INST] {user.strip()} [/INST]", add_special_tokens=False, return_tensors="pt")["input_ids"]
        return ids.to(DEVICE)

    def generate(user, amateur=None):
        if tok.chat_template:
            ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, return_tensors="pt")
            ids = ids["input_ids"] if not isinstance(ids, torch.Tensor) else ids
        else:
            ids = tok(f"<s>[INST] {user.strip()} [/INST]", add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = ids.to(DEVICE)
        kw = {}
        procs = []
        state["cad_proc"] = None
        if state["cad"] is not None:
            lam, mode, thr = state["cad"]
            if mode == "gcad":
                applies = lambda step: state["last_p"] > thr
            elif mode == "rcad":
                applies = lambda step: step < len(state["rmask"]) and state["rmask"][step]
            else:
                applies = lambda step: True
            state["cad_proc"] = ContextAware(model, encode(amateur), lam, state, applies)
            procs.append(state["cad_proc"])
        if state["copy"] > 0:
            allowed = torch.zeros(model.config.vocab_size, dtype=torch.bool, device=DEVICE)
            allowed[ids[0]] = True
            allowed[tok(FORMAT_TOKENS, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(DEVICE)] = True
            for t in list(getattr(tok, "all_special_ids", [])) + ([tok.eos_token_id] if tok.eos_token_id is not None else []):
                allowed[t] = True
            procs.append(CopyBias(allowed, state["copy"]))
        if procs:
            kw["logits_processor"] = LogitsProcessorList(procs)
        outs = []
        with torch.no_grad():
            for k in range(max(1, SAMPLE_K)):
                if SAMPLE_K > 0:
                    torch.manual_seed(SEED * 100003 + k)
                    kw.update(do_sample=True, temperature=SAMPLE_TEMP, top_p=0.95)
                else:
                    kw.update(do_sample=False)
                out = model.generate(ids, max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id, **kw)
                outs.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip())
        return outs

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f = open(OUT_DIR / "generations.jsonl", "w")
    summary = []
    for cond in CONDITIONS:
        if cond == "baseline":
            dname, alpha, tau = "baseline", 0.0, None
        else:
            dname, a = cond.split(":")
            tau = rate = None
            if dname == "gate":  # gate:<alpha>@<tau>  -> halluc direction, added only where the pre-emptive probe fires
                if gate_probe is None:
                    raise SystemExit("condition gate:* needs OCARANDU_LC_GATE_PROBE")
                a, t = a.split("@")
                tau = float(t)
            elif dname == "rgate":  # rgate:<alpha>@<rate> -> same direction at a random `rate` of positions
                a, t = a.split("@")
                rate = float(t)
                if not 0.0 < rate <= 1.0:
                    raise SystemExit("rgate rate must be in (0, 1]")
            elif dname == "gcad":  # gcad:<lam>@<tau> -> context-aware decoding where the probe fires
                if gate_probe is None:
                    raise SystemExit("condition gcad:* needs OCARANDU_LC_GATE_PROBE")
                a, t = a.split("@")
                tau = float(t)
            elif dname == "rcad":  # rcad:<lam>@<rate> -> the rate-matched random-position control of gcad
                a, t = a.split("@")
                rate = float(t)
                if not 0.0 < rate <= 1.0:
                    raise SystemExit("rcad rate must be in (0, 1]")
            alpha = float(a)
            m = re.fullmatch(r"random(\d+)", dname)
            if m:
                dirs[dname] = random_direction(int(m.group(1)), hidden)
            if dname not in ("copy", "gate", "rgate", "cad", "gcad", "rcad") and dname not in dirs:
                print(f"  skip {cond} (no '{dname}' direction)", file=sys.stderr)
                continue
        state["copy"] = alpha if dname == "copy" else 0.0  # "copy:<beta>" = decoding bias, no direction
        state["gate"] = tau if dname == "gate" else None
        # the CAD family never adds a vector: it only re-ranks, so the steering amount stays 0
        state["cad"] = (alpha, dname, tau if dname == "gcad" else None) if dname in ("cad", "gcad", "rcad") else None
        state["read_probe"] = dname == "gcad"
        steer_alpha = 0.0 if dname in ("copy", "cad", "gcad", "rcad") else alpha
        state["amt"] = steer_alpha
        vec_name = "halluc" if dname in ("gate", "rgate") else dname
        state["dir"] = None if steer_alpha == 0.0 else torch.tensor(dirs[vec_name], dtype=torch.bfloat16, device=DEVICE)
        rows = []
        for i, it in enumerate(items):
            state["gate_n"], state["gate_fired"], state["last_p"] = 0, 0, 0.0
            if dname in ("rgate", "rcad"):  # one fixed mask per item, seeded, so the run is reproducible
                rr = random.Random(SEED * 1000003 + i)
                state["rmask"] = [rr.random() < rate for _ in range(MAX_NEW + 8)]
                state["rpos"] = 0
            else:
                state["rmask"] = None
            gens = generate(PROMPTS[PROMPT_KIND].format(nome=it["nome"], cargo=it["cargo"], falas=it["falas"]),
                            amateur=PROMPTS[PROMPT_KIND].format(nome=it["nome"], cargo=it["cargo"], falas="").replace("\n\n\n\n", "\n\n"))
            cad_applied = (state["cad_proc"].n_applied / max(1, state["cad_proc"].step)) if state.get("cad_proc") else None
            gate_info = ({"gate_tau": tau, "gate_rate": state["gate_fired"] / max(1, state["gate_n"])} if dname == "gate"
                         else {"gate_target_rate": rate, "gate_rate": state["gate_fired"] / max(1, state["gate_n"])} if dname == "rgate"
                         else {"cad_lambda": alpha, "cad_tau": tau, "cad_target_rate": rate, "cad_rate": cad_applied} if dname in ("cad", "gcad", "rcad")
                         else {})
            for k, gen in enumerate(gens):
                row = {"condition": cond, "direction": dname, "alpha": alpha, "sample_id": it["sample_id"], **gate_info,
                       "speaker": it["speaker"], "nome": it["nome"], "cargo": it["cargo"],
                       "floor_chars": it["floor_chars"], "ctx_tokens": it["ctx_tokens"],
                       "opinion": " ".join(it["gold"]), "gold": it["gold"], "true_label": 0,
                       "chunks": it["windows"], "generation": gen, "n_chars": len(gen), "prompt_kind": PROMPT_KIND,
                       **({"sample": k, "temperature": SAMPLE_TEMP} if SAMPLE_K else {}),
                       **score_generation(gen, it["windows"], it["gold"])}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                f.flush()
                print(f"  [{cond}] {i + 1}/{len(items)}", file=sys.stderr, flush=True)
        cov = [r["coverage"] if r["coverage"] is not None else 0.0 for r in rows]
        low = [r["low_share"] for r in rows if r["low_share"] is not None]
        rec = [r["gold_recall"] for r in rows if r["gold_recall"] is not None]
        recs = [r["gold_recall_soft"] for r in rows if r["gold_recall_soft"] is not None]
        s = dict(condition=cond, n=len(rows), mean_coverage=float(np.mean(cov)),
                 ungrounded_share=float(np.mean(low)) if low else float("nan"),
                 gold_recall=float(np.mean(rec)) if rec else float("nan"),
                 gold_recall_soft=float(np.mean(recs)) if recs else float("nan"),
                 n_opinions=float(np.mean([r["n_opinions"] for r in rows])),
                 abstain_rate=float(np.mean([r["abstained"] for r in rows])),
                 fabricated_number_rate=float(np.mean([r["fabricated_number"] for r in rows])))
        summary.append(s)
        print(f"{cond:14s} coverage={s['mean_coverage']:.3f} ungrounded={s['ungrounded_share']:.3f} "
              f"gold_recall={s['gold_recall']:.3f} n_op={s['n_opinions']:.1f} abstain={s['abstain_rate']:.2f} "
              f"fab={s['fabricated_number_rate']:.2f}", file=sys.stderr, flush=True)
    f.close()
    import pandas as pd
    pd.DataFrame(summary).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"wrote {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
