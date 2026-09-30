"""Title swap: does the printed job title change whether the model endorses an attribution?

Only the title printed after the credited name changes; the person, the words, the evidence
and the speaker tags are fixed, so a change in the verdict cannot come from a name/tag
mismatch. The main arm (swap_*) is a false attribution: X's words are credited to Y,
another participant of the same hearing who does not hold the floor in the evidence
(correct answer "Não"); Y's name never changes, only Y's title. A higher P(Sim) under a
higher-ranking title is a credibility premium attached to office. The orig_* arms keep the
true speaker (correct answer "Sim") as a ceiling control. OCARANDU_TITLE_AXIS sets the
conditions:
  role (default)  swap_real (Y's real title), swap_no_cargo (no title), swap_role:<role>
                  (a real title of each role, paired within item; for Y's own role a
                  different title of that role: string changed, rank unchanged),
                  orig_real, orig_cross_role (true speaker, a title of a role other than Y's)
  party           swap_real, swap_party:none ("Deputado/a Federal"), swap_party:<P>
                  ("Deputado/a (P - UF)") for P in OCARANDU_PARTIES, orig_real
  gender          swap_real, swap_gender:M|F (every gendered office word in Y's title
                  in one grammatical gender; epicene titles skipped), orig_real
Items: OCARANDU_TITLE_N (400) supported opinions, spread over five source roles, whose
tagged evidence has the true speaker as its majority speaker (inputs: speaker_attributes.jsonl
and tagged_evidence.jsonl in data/publichearingbr/). Activations averaged over the claim
tokens and over the title tokens are saved at OCARANDU_SAVE_LAYERS.

Output: data/publichearingbr/title_swap/<model_tag>[_tagged][_<axis>]/{results.jsonl,
acts_claim_mean.npy, acts_cargo_mean.npy, meta.json} (no axis suffix for role).
Env: OCARANDU_MODEL_REPO (default meta-llama/Llama-3.1-8B-Instruct), OCARANDU_TAGGED=1
(tagged evidence in the prompt), OCARANDU_DEVICE. Needs one GPU. Analysis:
title_swap_analysis.py (role), title_axis_analysis.py (party, gender), title_swap_decisions.py.
"""

import json
import os
import random
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity_swap import PROMPT, QUESTION, TAGGED, TAGGED_PATH, chat_ids, yes_no_ids  # noqa: E402
from ocarandu.data.publichearingbr import load_nli_opinions  # noqa: E402
from ocarandu.data.speaker_attributes import flip_title_gender  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = os.environ.get("OCARANDU_MODEL_REPO", "meta-llama/Llama-3.1-8B-Instruct")
DEVICE = os.environ.get("OCARANDU_DEVICE", "cuda:0")
MODEL_TAG = REPO_ID.split("/")[-1].lower().replace(".", "").replace("-", "_")
# which identity axis of the title is manipulated: the institutional role, the
# party acronym inside a parliamentary title, or the grammatical gender of the
# office word. party and gender are minimal pairs -- one token / one morpheme.
AXIS = os.environ.get("OCARANDU_TITLE_AXIS", "role")
OUT_DIR = (ROOT / "data" / "publichearingbr" / "title_swap"
           / (MODEL_TAG + ("_tagged" if TAGGED else "") + ("" if AXIS == "role" else f"_{AXIS}")))
# spread across the spectrum, all with real Chamber presence
PARTIES = [x for x in os.environ.get("OCARANDU_PARTIES", "PSOL,PT,MDB,PSDB,PL,NOVO").split(",") if x]
ATTR_PATH = ROOT / "data" / "publichearingbr" / "speaker_attributes.jsonl"
SAVE_LAYERS = [int(x) for x in os.environ.get("OCARANDU_SAVE_LAYERS", "10,12,14,16,18,20").split(",")]
N_SAMPLE = int(os.environ.get("OCARANDU_TITLE_N", "400"))
ROLES = ("parliamentarian", "state", "civil_society", "academia", "private_sector")
SEED = 0


def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def load_attrs():
    """Derived attributes per (nome, cargo), plus a pool of real cargo strings
    per role. Attributes are derived, never annotated by the dataset authors."""
    by_key, cargo_pool = {}, defaultdict(list)
    for line in open(ATTR_PATH):
        d = json.loads(line)
        role = d.get("role_rule")
        by_key[(d["nome"], d["cargo"])] = {
            "role": role,
            "gender": d.get("gender"),
            "uf": d.get("uf_cargo") or d.get("camara_uf"),
        }
        if role in ROLES and d["cargo"] and len(d["cargo"]) < 160:
            cargo_pool[role].append(d["cargo"])
    return by_key, {r: sorted(set(v)) for r, v in cargo_pool.items()}


def party_title(gender, party, uf):
    """A parliamentary title differing only in the party acronym."""
    head = "Deputada" if gender == "F" else "Deputado"
    if party is None:
        return f"{head} Federal"
    return f"{head} ({party} - {uf or 'SP'})"


def axis_variants(axis, o, y_name, y_attr, src_role, cargo_pool, rng):
    """Conditions for one item. swap_* credit the words to y_name (who never
    held the floor -> the attribution is false); orig_* keep the true speaker."""
    y_role = y_attr["role"]
    if axis == "role":
        tgt = rng.choice([r for r in ROLES if r != y_role and cargo_pool[r]])
        variants = [
            ("orig_real", o.person_name, o.person_role, src_role),
            ("orig_cross_role", o.person_name, rng.choice(cargo_pool[tgt]), tgt),
            ("swap_real", y_name, y_attr["cargo"], y_role),
            ("swap_no_cargo", y_name, None, None),
        ]
        # every rank is shown to the same false claimant on the same opinion, so
        # ranks are compared paired within item. The claimant's own role is
        # included with a different real title of that role: that cell is the
        # same-role control (string changed, rank unchanged).
        for shown in ROLES:
            pool = [c for c in cargo_pool[shown] if c != y_attr["cargo"]]
            if pool:
                variants.append((f"swap_role:{shown}", y_name, rng.choice(pool), shown))
        return variants
    if axis == "party":
        # identical string except the party acronym; UF held fixed per item
        g, uf = y_attr["gender"], y_attr["uf"]
        variants = [
            ("orig_real", o.person_name, o.person_role, src_role),
            ("swap_real", y_name, y_attr["cargo"], y_role),
            ("swap_party:none", y_name, party_title(g, None, uf), None),
        ]
        for party in PARTIES:
            variants.append((f"swap_party:{party}", y_name, party_title(g, party, uf), party))
        return variants
    if axis == "gender":
        # one morpheme: the office word's grammatical gender, nothing else
        masc = flip_title_gender(y_attr["cargo"], "M", head_only=False)
        fem = flip_title_gender(y_attr["cargo"], "F", head_only=False)
        if not masc or not fem or masc == fem:
            return None  # epicene head ("Representante"): nothing to flip
        return [
            ("orig_real", o.person_name, o.person_role, src_role),
            ("swap_real", y_name, y_attr["cargo"], y_role),
            ("swap_gender:M", y_name, masc, "M"),
            ("swap_gender:F", y_name, fem, "F"),
        ]
    raise SystemExit(f"unknown OCARANDU_TITLE_AXIS={axis!r}")


def render(nome, cargo, chunks):
    if cargo:
        return PROMPT.format(nome=nome, cargo=cargo, chunks=chunks)
    return PROMPT.format(nome=nome, cargo="", chunks=chunks).replace(f"{nome} ()", nome)


def cargo_span(tok, user, nome, cargo):
    """Token range covering the cargo string inside the context segment.

    chat_ids builds the context as tok(text_ctx), so re-tokenising the same
    rendered text with offsets gives positions that line up exactly. Falls back
    to the name's own span in the no_cargo condition, where there is no title.
    """
    if tok.chat_template:
        text_ctx = tok.apply_chat_template([{"role": "user", "content": user}],
                                           add_generation_prompt=True, tokenize=False)
    else:
        text_ctx = f"<s>[INST] {user.strip()} [/INST]"
    enc = tok(text_ctx, add_special_tokens=False, return_offsets_mapping=True)
    n = len(enc["input_ids"])
    needle = cargo if cargo else nome
    anchor = text_ctx.find(f"{nome} ({cargo})" if cargo else nome)
    i = text_ctx.find(needle, max(anchor, 0))
    if i < 0:
        return n - 1, n
    off = enc["offset_mapping"]
    lo = min((j for j, (a, b) in enumerate(off) if a <= i < b), default=n - 1)
    hi = max((j for j, (a, b) in enumerate(off) if a < i + len(needle) <= b), default=lo)
    return lo, hi + 1


def build_items(rng):
    by_key, cargo_pool = load_attrs()
    tagged = {}
    for line in open(TAGGED_PATH):
        d = json.loads(line)
        tagged[(d["sample_id"], d["nome"], d["opinion"])] = d

    cand, by_hearing = [], defaultdict(set)
    for o in load_nli_opinions():
        if len(o.context_chunks) != 4:
            continue
        by_hearing[o.sample_id].add((o.person_name, o.person_role))
        d = tagged.get((o.sample_id, o.person_name, o.opinion))
        if o.is_hallucination or d is None or not d.get("majority_is_attributed"):
            continue
        attr = by_key.get((o.person_name, o.person_role))
        if attr and attr["role"] in ROLES:
            cand.append((o, d, attr["role"]))

    by_role = defaultdict(list)
    for row in cand:
        by_role[row[2]].append(row)
    per_role = max(1, N_SAMPLE // len(ROLES))

    items = []
    for role in ROLES:
        rows = by_role[role]
        rng.shuffle(rows)
        taken = 0
        for o, d, src_role in rows:
            if taken >= per_role:
                break
            # a false claimant from the same hearing who does not hold the floor here
            spoke = {norm(s) for s in d["chunk_speakers"] if s}
            others = sorted(s for s in by_hearing[o.sample_id]
                            if s[0] != o.person_name
                            and not any(norm(s[0].split()[-1]) in x for x in spoke))
            if not others:
                continue
            y_name, y_cargo = rng.choice(others)
            y_attr = by_key.get((y_name, y_cargo))
            if not y_attr or y_attr["role"] not in ROLES:
                continue
            y_attr = {**y_attr, "cargo": y_cargo}
            if AXIS == "party" and not y_attr["gender"]:
                continue  # the party title needs the right gendered head word
            variants = axis_variants(AXIS, o, y_name, y_attr, src_role, cargo_pool, rng)
            if not variants:
                continue
            items.append({"o": o, "d": d, "src_role": src_role, "y_name": y_name,
                          "y_role": y_attr["role"], "y_gender": y_attr["gender"],
                          "y_uf": y_attr["uf"], "variants": variants})
            taken += 1
    return items, cargo_pool


def main():
    import torch
    from nnsight import LanguageModel

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    items, cargo_pool = build_items(rng)
    n_max = sum(len(it["variants"]) for it in items)
    print(f"{len(items)} opinions x {n_max / max(len(items), 1):.1f} variants = {n_max} forward passes "
          f"(title pool: {', '.join(f'{r}:{len(cargo_pool[r])}' for r in ROLES)})", file=sys.stderr, flush=True)

    lm = LanguageModel(REPO_ID, device_map=DEVICE, dispatch=True, torch_dtype=torch.bfloat16)
    tok = lm.tokenizer
    yes_ids, no_ids = yes_no_ids(tok)

    acts_claim = np.lib.format.open_memmap(OUT_DIR / "acts_claim_mean.npy", mode="w+", dtype=np.float16,
                                           shape=(n_max, len(SAVE_LAYERS), lm.config.hidden_size))
    acts_cargo = np.lib.format.open_memmap(OUT_DIR / "acts_cargo_mean.npy", mode="w+", dtype=np.float16,
                                           shape=(n_max, len(SAVE_LAYERS), lm.config.hidden_size))
    f = open(OUT_DIR / "results.jsonl", "w")
    row_i = 0
    for n_done, it in enumerate(items):
        o, d = it["o"], it["d"]
        chunks = "\n\n".join(d["tagged_chunks"] if TAGGED else o.context_chunks)
        for variant, nome, cargo, role_shown in it["variants"]:
            user = render(nome, cargo, chunks)
            ids, n_ctx, n_resp = chat_ids(tok, user, o.opinion, QUESTION)
            c_lo, c_hi = cargo_span(tok, user, nome, cargo)
            outs = {}
            with torch.no_grad(), lm.trace(torch.tensor([ids], device=lm.device)):
                for li, layer in enumerate(SAVE_LAYERS):
                    outs[li] = lm.model.layers[layer].output[0].save()
                outs["logits"] = lm.output.logits.save()
            p = torch.softmax(outs["logits"][0, -1].detach().float(), -1)
            p_yes, p_no = float(p[yes_ids].sum()), float(p[no_ids].sum())
            for li in range(len(SAVE_LAYERS)):
                ot = outs[li]
                h = (ot[0] if ot.dim() == 3 else ot).detach().float().cpu().numpy()
                acts_claim[row_i, li] = h[n_ctx:n_ctx + n_resp].mean(0)
                acts_cargo[row_i, li] = h[c_lo:c_hi].mean(0)
            f.write(json.dumps({"row": row_i, "sample_id": o.sample_id,
                                "true_speaker": o.person_name, "credited": nome,
                                "variant": variant, "cargo": cargo, "role_shown": role_shown,
                                "axis": AXIS, "src_role": it["src_role"],
                                "y_role": it["y_role"], "y_gender": it["y_gender"],
                                "y_uf": it["y_uf"], "opinion": o.opinion,
                                "p_yes": p_yes, "p_no": p_no,
                                "mass": p_yes + p_no}, ensure_ascii=False) + "\n")
            row_i += 1
        if (n_done + 1) % 50 == 0:
            f.flush(); acts_claim.flush(); acts_cargo.flush()
            print(f"  {n_done + 1}/{len(items)} opinions", file=sys.stderr, flush=True)
    f.close(); acts_claim.flush(); acts_cargo.flush()
    json.dump({"save_layers": SAVE_LAYERS, "model": REPO_ID, "tagged": TAGGED,
               "axis": AXIS, "parties": PARTIES, "n_rows": row_i},
              open(OUT_DIR / "meta.json", "w"))
    print(f"wrote {OUT_DIR} ({row_i} rows)")


if __name__ == "__main__":
    main()
