"""Identity-swap counterfactuals: the same supported opinion and evidence, credited to someone else.

For a role-stratified sample of human-verified supported opinions (speaker with a derived
role and gender), only the speaker's name and title in the prompt change:
  original        the true speaker
  within_hearing  another participant of the same hearing (a realistic misattribution)
  cross_role      any speaker in the corpus with a different role and the same gender
  cross_gender    any speaker in the corpus with the same role and the opposite gender
One teacher-forced pass per variant reads (a) the in-domain hallucination probe (mean over
the opinion tokens at OCARANDU_PROBE_LAYER), scored by the cross-validation fold model that
held out this hearing, and (b) the verdict P(Sim), renormalised over Sim/Não, for the
question "is this attribution supported by the excerpts?" asked in a second user turn
(causal attention leaves the opinion-token activations unchanged). target_in_chunks marks
swaps whose target's surname occurs in the evidence text.

Output: data/publichearingbr/identity_swap/<model_tag>[_tagged]/{results.jsonl (one row per
opinion x variant), acts_opinion_mean.npy, acts_name_last.npy, meta.json}.
Env: OCARANDU_MODEL_REPO (default NousResearch/Llama-2-7b-chat-hf), OCARANDU_ACT_DIR (probe
activations under data/publichearingbr/, default activations), OCARANDU_PROBE_LAYER (14),
OCARANDU_TAGGED=1 (speaker-tagged evidence from tagged_evidence.jsonl), OCARANDU_SWAP_N (800
opinions), OCARANDU_SAVE_LAYERS, OCARANDU_ANSWER_PREFIX, OCARANDU_DEVICE. Needs one GPU.
Statistics: identity_swap_analysis.py.
"""

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np
import pandas as pd
import torch
from nnsight import LanguageModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ocarandu.data.publichearingbr import load_nli_opinions
from ocarandu.data.speaker_attributes import normalize

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = os.environ.get("OCARANDU_MODEL_REPO", "NousResearch/Llama-2-7b-chat-hf")
DEVICE = os.environ.get("OCARANDU_DEVICE", "cuda:0")
ACT_DIR = ROOT / "data" / "publichearingbr" / os.environ.get("OCARANDU_ACT_DIR", "activations")
ATTR_PATH = ROOT / "data" / "publichearingbr" / "speaker_attributes.jsonl"
MODEL_TAG = REPO_ID.split("/")[-1].lower().replace(".", "").replace("-", "_")
OUT_DIR = ROOT / "data" / "publichearingbr" / "identity_swap" / (MODEL_TAG + ("_tagged" if os.environ.get("OCARANDU_TAGGED", "0") == "1" else ""))
PROBE_LAYER = int(os.environ.get("OCARANDU_PROBE_LAYER", "14"))
# OCARANDU_TAGGED=1 -> evidence chunks carry their transcript turn header ("[SPEAKER] ...")
TAGGED = os.environ.get("OCARANDU_TAGGED", "0") == "1"
TAGGED_PATH = ROOT / "data" / "publichearingbr" / "tagged_evidence.jsonl"
SAVE_LAYERS = [int(x) for x in os.environ.get("OCARANDU_SAVE_LAYERS", "6,8,10,12,14,16,20,24").split(",")]
# The verdict is read at the first assistant token after this prefix, so the
# next token is the answer word itself rather than a space/preamble token.
ANSWER_PREFIX = os.environ.get("OCARANDU_ANSWER_PREFIX", " Resposta:")
N_SAMPLE = int(os.environ.get("OCARANDU_SWAP_N", "800"))
SEED = 0

PROMPT = (
    "Com base nos trechos de uma transcrição de audiência pública a seguir, "
    "avalie a seguinte opinião atribuída a {nome} ({cargo}).\n\n{chunks}"
)
if os.environ.get("OCARANDU_TAGGED", "0") == "1":
    PROMPT = (
        "Com base nos trechos de uma transcrição de audiência pública a seguir (cada trecho começa com o nome "
        "de quem estava com a palavra, entre colchetes), avalie a seguinte opinião atribuída a {nome} ({cargo}).\n\n{chunks}"
    )
QUESTION = "A atribuição acima é sustentada pelos trechos da transcrição? Responda apenas Sim ou Não."


def _ids(x):
    return x["input_ids"] if isinstance(x, dict) or hasattr(x, "keys") else list(x)


def chat_ids(tokenizer, user, assistant, user2):
    """Token ids for a two-turn chat ending in the assistant's answer prefix.

    Built by concatenation like extract_publichearingbr_activations.py (context
    ids + separately tokenised opinion), so the opinion tokens match the probe's
    training inputs. With a chat template, context and tail come from the
    template; without one (the Llama-2 mirror):
    '<s>[INST] user [/INST] assistant </s><s>[INST] user2 [/INST] prefix'.
    Returns (ids, n_context_tokens, n_opinion_tokens)."""
    if tokenizer.chat_template:
        msgs1 = [{"role": "user", "content": user}]
        msgs3 = msgs1 + [{"role": "assistant", "content": assistant}, {"role": "user", "content": user2}]
        text_ctx = tokenizer.apply_chat_template(msgs1, add_generation_prompt=True, tokenize=False)
        text_full = tokenizer.apply_chat_template(msgs3, add_generation_prompt=True, tokenize=False)
        assert text_full.startswith(text_ctx + assistant), "chat template does not render the assistant turn verbatim"
        tail_text = text_full[len(text_ctx) + len(assistant):] + ANSWER_PREFIX.strip()
        ctx = tokenizer(text_ctx, add_special_tokens=False)["input_ids"]
        resp = tokenizer(assistant, add_special_tokens=False)["input_ids"]
        tail = tokenizer(tail_text, add_special_tokens=False)["input_ids"]
        return ctx + resp + tail, len(ctx), len(resp)
    ctx = tokenizer(f"<s>[INST] {user.strip()} [/INST]", add_special_tokens=False)["input_ids"]
    resp = tokenizer(assistant, add_special_tokens=False)["input_ids"]
    tail = tokenizer(f" </s><s>[INST] {user2} [/INST]{ANSWER_PREFIX}", add_special_tokens=False)["input_ids"]
    return ctx + resp + tail, len(ctx), len(resp)


def yes_no_ids(tokenizer):
    def firsts(words):
        out = set()
        for w in words:
            for v in (w, " " + w):
                ids = tokenizer(v, add_special_tokens=False)["input_ids"]
                if ids:
                    out.add(ids[0])
        return sorted(out)
    return firsts(["Sim", "sim", "SIM"]), firsts(["Não", "não", "NÃO", "Nao", "nao"])


def fold_models(index, x, y):
    """Per-fold (scaler, clf) at PROBE_LAYER with GroupKFold by hearing, and the
    hearing -> fold map (the same splits as run_publichearingbr_probe.py)."""
    groups = index.sample_id.values
    models, fold_of = [], {}
    for k, (tr, te) in enumerate(GroupKFold(5).split(x, y, groups)):
        sc = StandardScaler().fit(x[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED).fit(sc.transform(x[tr]), y[tr])
        models.append((sc, clf))
        for h in set(groups[te]):
            fold_of[int(h)] = k
    return models, fold_of


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # --- speakers and attributes ---
    attrs = {}
    for line in open(ATTR_PATH):
        a = json.loads(line)
        attrs[(a["nome"], a["cargo"])] = a
    opinions = [o for o in load_nli_opinions() if len(o.context_chunks) == 4]
    if TAGGED:
        from dataclasses import replace as _replace
        tagged = {}
        for line in open(TAGGED_PATH):
            d = json.loads(line)
            tagged[(d["sample_id"], d["nome"], d["opinion"])] = tuple(d["tagged_chunks"])
        opinions = [_replace(o, context_chunks=tagged[(o.sample_id, o.person_name, o.opinion)]) for o in opinions
                    if (o.sample_id, o.person_name, o.opinion) in tagged]
        print(f"TAGGED evidence: {len(opinions)} opinions with speaker-tagged chunks", file=sys.stderr)
    by_hearing = defaultdict(set)
    for o in opinions:
        by_hearing[o.sample_id].add((o.person_name, o.person_role))
    pool = [(n, c, a["role_rule"], a["gender"]) for (n, c), a in attrs.items()]

    # --- probe fold models from the saved activations ---
    index = pd.read_json(ACT_DIR / "index.jsonl", lines=True)
    x = np.asarray(np.load(ACT_DIR / "opinion_mean.npy", mmap_mode="r")[:, PROBE_LAYER, :], dtype=np.float32)
    y = index.label.values.astype(int)
    models, fold_of = fold_models(index, x, y)
    row_of = {(int(r.sample_id), r.nome, r.opinion): int(r.row) for r in index.itertuples()}
    print(f"fold models ready ({len(models)} folds, layer {PROBE_LAYER})", file=sys.stderr)

    # --- sample supported opinions with known gender ---
    cands = []
    for o in opinions:
        a = attrs.get((o.person_name, o.person_role), {})
        if not o.is_hallucination and a.get("gender") in ("F", "M") and a.get("role_rule") not in (None, "other"):
            cands.append((o, a))
    rng.shuffle(cands)
    # stratify: cap per role so small roles are represented
    per_role = defaultdict(list)
    for o, a in cands:
        per_role[a["role_rule"]].append((o, a))
    cap = max(1, N_SAMPLE // len(per_role))
    sample = []
    for r, items in per_role.items():
        sample.extend(items[:cap])
    leftover = [it for r, items in per_role.items() for it in items[cap:]]
    sample.extend(leftover[: max(0, N_SAMPLE - len(sample))])
    print(f"{len(sample)} supported opinions sampled; per role: "
          + ", ".join(f"{r}={sum(1 for _, a in sample if a['role_rule']==r)}" for r in per_role), file=sys.stderr)

    def pick(pred):
        opts = [p for p in pool if pred(p)]
        return rng.choice(opts) if opts else None

    def variants(o, a):
        out = []
        others = [s for s in by_hearing[o.sample_id] if s[0] != o.person_name]
        if others:
            n, c = rng.choice(sorted(others))
            t = attrs.get((n, c), {})
            out.append(("within_hearing", n, c, t.get("role_rule"), t.get("gender")))
        p = pick(lambda p: p[2] not in (a["role_rule"], "other", None) and p[3] == a["gender"] and p[0] != o.person_name)
        if p:
            out.append(("cross_role", *p))
        p = pick(lambda p: p[2] == a["role_rule"] and p[3] == ("F" if a["gender"] == "M" else "M") and p[0] != o.person_name)
        if p:
            out.append(("cross_gender", *p))
        return out

    lm = LanguageModel(REPO_ID, device_map=DEVICE, dtype=torch.bfloat16)
    tok = lm.tokenizer
    yes_ids, no_ids = yes_no_ids(tok)
    print(f"yes ids {yes_ids} no ids {no_ids}", file=sys.stderr)

    def name_span(ids, nome):
        """Token span of the speaker name inside the first user turn (by decoding prefixes)."""
        text = tok.decode(ids)
        anchor = text.find("atribuída a ")
        start = text.find(nome, anchor + 12 if anchor >= 0 else 0)
        if start < 0:
            return None
        end = start + len(nome)
        # map char span -> token indices via cumulative decode lengths
        pos, first, last = 0, None, None
        for i in range(len(ids)):
            piece = tok.decode(ids[: i + 1])
            if first is None and len(piece) > start:
                first = i
            if len(piece) >= end:
                last = i
                break
            pos = len(piece)
        return (first, last + 1) if first is not None and last is not None else None

    layers_all = sorted(set(SAVE_LAYERS + [PROBE_LAYER]))

    def run(o, nome, cargo):
        user = PROMPT.format(nome=nome, cargo=cargo, chunks="\n\n".join(o.context_chunks))
        ids, ctx_len, resp_len = chat_ids(tok, user, o.opinion, QUESTION)
        ids_t = torch.tensor([ids]).to(DEVICE)
        outs = {}
        with torch.no_grad(), lm.trace(ids_t):
            for l in layers_all:
                outs[l] = lm.model.layers[l].output[0].save()
            outs["logits"] = lm.output.logits.save()
        hs = {l: (outs[l][0] if outs[l].dim() == 3 else outs[l]) for l in layers_all}
        span = name_span(ids, nome)
        opinion_mean = np.stack([hs[l][ctx_len:ctx_len + resp_len, :].float().mean(dim=0).cpu().numpy() for l in SAVE_LAYERS]).astype(np.float16)
        name_last = np.stack([hs[l][span[1] - 1, :].float().cpu().numpy() if span else np.zeros(hs[l].shape[-1], dtype=np.float32) for l in SAVE_LAYERS]).astype(np.float16)
        pooled = hs[PROBE_LAYER][ctx_len:ctx_len + resp_len, :].float().mean(dim=0).cpu().numpy()[None, :]
        sc, clf = models[fold_of[int(o.sample_id)]]
        probe = float(clf.predict_proba(sc.transform(pooled))[0, 1])
        lp = torch.log_softmax(outs["logits"][0, -1].float(), dim=-1)
        p_yes = float(torch.logsumexp(lp[yes_ids], 0).exp()); p_no = float(torch.logsumexp(lp[no_ids], 0).exp())
        return probe, p_yes / (p_yes + p_no + 1e-9), p_yes + p_no, opinion_mean, name_last, span is not None

    out_path = OUT_DIR / "results.jsonl"
    if out_path.exists():
        out_path.unlink()  # always a clean run: activation rows align with result rows
    n_rows_max = len(sample) * 4
    acts_op = np.lib.format.open_memmap(OUT_DIR / "acts_opinion_mean.npy", mode="w+", dtype=np.float16, shape=(n_rows_max, len(SAVE_LAYERS), lm.config.hidden_size))
    acts_nm = np.lib.format.open_memmap(OUT_DIR / "acts_name_last.npy", mode="w+", dtype=np.float16, shape=(n_rows_max, len(SAVE_LAYERS), lm.config.hidden_size))
    json.dump({"save_layers": SAVE_LAYERS, "probe_layer": PROBE_LAYER, "model": REPO_ID, "answer_prefix": ANSWER_PREFIX}, open(OUT_DIR / "meta.json", "w"))
    done = set()
    f = open(out_path, "w")
    n_pass = 0
    for i, (o, a) in enumerate(sample):
        chunks_norm = normalize(" ".join(o.context_chunks))
        rows = [("original", o.person_name, o.person_role, a["role_rule"], a["gender"])] + variants(o, a)
        for variant, nome, cargo, t_role, t_gender in rows:
            key = (o.sample_id, o.person_name, o.opinion, variant)
            if key in done:
                continue
            probe, p_yes, mass, op_vec, nm_vec, name_found = run(o, nome, cargo)
            acts_op[n_pass] = op_vec; acts_nm[n_pass] = nm_vec
            f.write(json.dumps({
                "row": n_pass, "name_found": bool(name_found),
                "sample_id": o.sample_id, "nome": o.person_name, "cargo": o.person_role, "opinion": o.opinion,
                "src_role": a["role_rule"], "src_gender": a["gender"], "variant": variant,
                "target_nome": nome, "target_cargo": cargo, "target_role": t_role, "target_gender": t_gender,
                "target_in_chunks": normalize(nome.split()[-1]) in chunks_norm if variant != "original" else True,
                "probe": probe, "p_yes": p_yes, "yes_no_mass": mass, "model": REPO_ID, "layer": PROBE_LAYER,
            }, ensure_ascii=False) + "\n")
            n_pass += 1
        if (i + 1) % 50 == 0:
            f.flush(); acts_op.flush(); acts_nm.flush()
            print(f"  {i + 1}/{len(sample)} opinions ({n_pass} passes)", file=sys.stderr)
    f.close(); acts_op.flush(); acts_nm.flush()
    json.dump({"save_layers": SAVE_LAYERS, "probe_layer": PROBE_LAYER, "model": REPO_ID, "answer_prefix": ANSWER_PREFIX, "n_rows": n_pass}, open(OUT_DIR / "meta.json", "w"))

    # --- compact summary ---
    df = pd.read_json(out_path, lines=True)
    base = df[df.variant == "original"].set_index(["sample_id", "nome", "opinion"])
    print("\n=== identity swap summary (paired against the original attribution) ===")
    for v in ("within_hearing", "cross_role", "cross_gender"):
        sub = df[df.variant == v].set_index(["sample_id", "nome", "opinion"])
        j = sub.join(base[["probe", "p_yes"]], rsuffix="_orig")
        if j.empty:
            continue
        d_probe = j.probe - j.probe_orig; d_yes = j.p_yes - j.p_yes_orig
        print(f"{v:15s} n={len(j):4d}  Δprobe mean {d_probe.mean():+.3f} (share up {(d_probe>0).mean():.2f})  "
              f"Δp_yes mean {d_yes.mean():+.3f} (share down {(d_yes<0).mean():.2f})")
        if v == "cross_gender":
            for g in ("F", "M"):
                jj = j[j.src_gender == g]
                print(f"   source {g}->{'M' if g=='F' else 'F'}: n={len(jj)} Δprobe {jj.probe.sub(jj.probe_orig).mean():+.3f} Δp_yes {jj.p_yes.sub(jj.p_yes_orig).mean():+.3f}")
        if v == "cross_role":
            for r in sorted(j.src_role.unique()):
                jj = j[j.src_role == r]
                print(f"   source {r:15s}: n={len(jj)} Δprobe {jj.probe.sub(jj.probe_orig).mean():+.3f} Δp_yes {jj.p_yes.sub(jj.p_yes_orig).mean():+.3f}")
        if v == "within_hearing":
            for flag in (True, False):
                jj = j[j.target_in_chunks == flag]
                print(f"   target in chunks={flag}: n={len(jj)} Δprobe {jj.probe.sub(jj.probe_orig).mean():+.3f} Δp_yes {jj.p_yes.sub(jj.p_yes_orig).mean():+.3f}")
    print(f"original: mean probe {base.probe.mean():.3f}, mean p_yes {base.p_yes.mean():.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
