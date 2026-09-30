"""Stage 2 of the published-article audit: attributions and omissions in the LDS articles. CPU only.

Input: the stage-1 embeddings in data/publichearingbr/lds_audit/<model_tag>/ (written by
scripts/extract_lds_audit.py), the LDS transcripts and articles, and
data/publichearingbr/speaker_attributes.jsonl. Floor-holders are speakers with more than MIN_FLOOR
characters of floor text.
An attribution is an article sentence that credits a statement to a floor-holder through a pattern
(e.g. "Segundo / Para / De acordo com <Name>", "<Name> afirmou / destacou ..."); a bare name is not
one. It is compared with the competing floor-holder of highest coverage on two independent signals:
content-word coverage of the sentence in the speaker's full turns (not the truncated embedding
input), and max cosine to the speaker's turn embeddings.
    supported      no competitor is clearly better on both signals
    other_better   a competitor is better on both by COV_MARGIN / COS_MARGIN: a candidate
                   misattribution, to be checked by hand, not a confirmed error
    unverifiable   no floor-holder reaches UNVERIF_COV coverage (e.g. background the journalist
                   added)
Omission rows: one per floor-holder; named_in_article is a surname-substring test and role/gender
are joined by surname (scripts/lds_omission_relabel.py re-derives all three with stricter rules).
--validate prints the share of attributions that would be flagged if swapped to the competitor,
next to the share of real attributions flagged (false-positive rate).

Usage: uv run python scripts/lds_audit_stage2.py [model_tag] [--max-hearings N] [--validate]
       (model_tag defaults to llama_31_8b_instruct)
Output: data/publichearingbr/lds_audit/<model_tag>/stage2/{attributions.jsonl,omissions.jsonl}
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tagged_evidence import HEADER, parse_header  # noqa: E402

from ocarandu.data.publichearingbr import DEFAULT_LDS_PATH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LDS = DEFAULT_LDS_PATH
ATTRS = ROOT / "data" / "publichearingbr" / "speaker_attributes.jsonl"

VERBS = (r"afirm|disse|declar|destac|defend|explic|ressalt|apont|critic|cobr|sugeri|prop[oô]s|"
         r"lembr|acrescent|avali|argument|alert|reclam|question|pediu|solicit|relat|coment|observ|"
         r"denunci|elogi|anunci|inform|confirm|negou|admit|conclu|advert|enfatiz|reiter")
# A person's name, allowing lowercase particles ("Marcel van Hattem", "Maria da Silva").
NAME = (r"[A-ZÀ-Ú][\wÀ-ú'.-]+(?:\s+(?:d[aeo]s?\s+|van\s+|von\s+|del\s+|della\s+)?"
        r"[A-ZÀ-Ú][\wÀ-ú'.-]+){0,3}")
LEAD = re.compile(r"\b(?:segundo|para|de acordo com|conforme|na avalia[çc][ãa]o de|na opini[ãa]o de)\s+"
                  r"(?:o |a )?(?:deputad[oa]|senador[a]?|ministr[oa]|president[ea]|professor[a]?|"
                  r"advogad[oa]|diretor[a]?|secret[áa]ri[oa]|relator[a]?|jornalista|pesquisador[a]?)?\s*"
                  r"(" + NAME + r")", re.I)
# The article style is "O deputado Marcel van Hattem (Novo-RS), que pediu a
# audiencia, destacou que ...": a party parenthetical and an appositive clause
# may sit between the name and the verb.
TRAIL = re.compile(r"(" + NAME + r")"
                   r"(?:\s*\([^)]{0,50}\))?"
                   r"(?:\s*,[^,.]{0,90},)?"
                   r"\s+(?:tamb[ée]m\s+|ainda\s+|j[áa]\s+|n[ãa]o\s+)?(?:" + VERBS + r")\w*")
STOP = set("""de da do das dos a o as os e em para com que por no na nos nas um uma ao aos se nao sim
mais muito ja sobre entre como ou mas tambem sua seu suas seus isso este esta esse essa aqui ha sao
foi ser estao tem temos vamos eu voce nossa nosso pela pelo ate quando onde porque entao ainda apenas
todos todas cada outro outra ter fazer pode deve audiencia publica comissao deputado deputada""".split())
MIN_FLOOR = 1500      # chars of speaking time to count as a substantial participant
COV_MARGIN = 0.10     # how much better a competitor must be on coverage
COS_MARGIN = 0.02     # ...and on cosine, before we call it a candidate misattribution
UNVERIF_COV = 0.15    # below this coverage for every speaker: nothing in the record matches


def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    return re.sub(r"\s+", " ", "".join(c for c in s if not unicodedata.combining(c))).lower().strip()


def content_words(t):
    return {w for w in re.findall(r"[a-z0-9]{4,}", norm(t)) if w not in STOP}


def surname(name):
    parts = [p for p in norm(name).split() if len(p) > 2 and p not in ("dos", "das", "dei")]
    return parts[-1] if parts else norm(name)


def parse_turns(transcript):
    """speaker -> full concatenated floor text (not truncated, unlike the embedding inputs)."""
    tn = re.sub(r"\s+", " ", transcript)
    heads = list(HEADER.finditer(tn))
    turns = defaultdict(list)
    for k, m in enumerate(heads):
        name, _ = parse_header(m)
        stop = heads[k + 1].start() if k + 1 < len(heads) else len(tn)
        turns[name].append(tn[m.end():stop].strip())
    return {k: " ".join(v) for k, v in turns.items()}


def resolve(cand, speakers):
    """Map a name captured from the article to a hearing speaker, or None.

    Surname alone is only trusted for a single-token candidate. A multi-token
    name must also agree on the first token, otherwise "Alexandre de Moraes"
    (the judge being discussed) resolves to the speaker "Marcelo Moraes" and
    the audit invents a misattribution that was never made.
    """
    c = norm(cand)
    if len(c) < 4:
        return None
    c_parts = [p for p in c.split() if len(p) > 2 and p not in ("dos", "das", "van", "von", "del")]
    cs = surname(cand)
    for sp in speakers:
        s_norm = norm(sp)
        if c == s_norm or c in s_norm or s_norm in c:
            return sp
    for sp in speakers:
        s_parts = [p for p in norm(sp).split() if len(p) > 2 and p not in ("dos", "das", "van", "von", "del")]
        if not s_parts or len(cs) <= 3 or cs != surname(sp):
            continue
        if len(c_parts) < 2:          # single token: surname match is all we have
            return sp
        if c_parts[0] == s_parts[0]:  # multi-token: first name must agree too
            return sp
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_tag", nargs="?", default="llama_31_8b_instruct")
    ap.add_argument("--max-hearings", type=int, default=0)
    ap.add_argument("--validate", action="store_true",
                    help="re-score every attributed sentence with the attribution deliberately "
                         "swapped to another floor-holder; a detector that does not flag those "
                         "cannot be trusted to flag real ones")
    args = ap.parse_args()
    D = ROOT / "data" / "publichearingbr" / "lds_audit" / args.model_tag
    OUT = D / "stage2"
    OUT.mkdir(parents=True, exist_ok=True)

    turn_idx = [json.loads(l) for l in open(D / "turn_index.jsonl")]
    sent_idx = [json.loads(l) for l in open(D / "sent_index.jsonl")]
    TV = np.load(D / "turn_mean.npy", mmap_mode="r")
    SV = np.load(D / "sent_mean.npy", mmap_mode="r")
    LAYER_I = TV.shape[1] // 2  # middle of the saved layers

    attrs = {}
    for line in open(ATTRS):
        d = json.loads(line)
        attrs[surname(d["nome"])] = d

    turn_rows = defaultdict(list)
    for t in turn_idx:
        turn_rows[t["hearing_id"]].append(t)
    sent_rows = defaultdict(list)
    for s in sent_idx:
        sent_rows[s["hearing_id"]].append(s)

    hearings = {}
    for line in open(LDS, encoding="utf-8"):
        r = json.loads(line)
        hearings[r["id"]] = r

    att_out, omit_out = [], []
    ids = sorted(sent_rows)
    if args.max_hearings:
        ids = ids[:args.max_hearings]

    for hid in ids:
        rec = hearings.get(hid)
        if rec is None:
            continue
        full = parse_turns(rec["transcricao"])
        floor = {k: v for k, v in full.items() if len(v) > MIN_FLOOR}
        if not floor:
            continue
        cw = {k: content_words(v) for k, v in floor.items()}
        # embedding rows per speaker (truncated inputs, fine for similarity)
        rows_by_speaker = defaultdict(list)
        for t in turn_rows[hid]:
            if t["speaker"] in floor:
                rows_by_speaker[t["speaker"]].append(t["row"])
        vecs = {k: np.asarray(TV[v, LAYER_I, :], dtype=np.float32) for k, v in rows_by_speaker.items() if v}
        for k in vecs:
            vecs[k] /= np.linalg.norm(vecs[k], axis=1, keepdims=True).clip(1e-6)

        article_text = norm(rec["materia"])
        named = {sp for sp in floor if surname(sp) and len(surname(sp)) > 3 and surname(sp) in article_text}

        for s in sent_rows[hid]:
            text = s["text"]
            cands = [m.group(1) for m in LEAD.finditer(text)] + [m.group(1) for m in TRAIL.finditer(text)]
            sp = next((r for r in (resolve(c, floor) for c in cands) if r), None)
            if sp is None:
                continue
            w = content_words(text)
            if len(w) < 4:
                continue
            cov = {k: len(w & cw[k]) / len(w) for k in floor}
            sv = np.asarray(SV[s["row"], LAYER_I, :], dtype=np.float32)
            sv /= np.linalg.norm(sv).clip(1e-6)
            # best-matching turn of each speaker, not the average turn
            cos = {k: (float(np.max(vecs[k] @ sv)) if k in vecs else -1.0) for k in floor}
            others = [k for k in floor if k != sp]
            best_o = max(others, key=lambda k: cov[k]) if others else None
            if max(cov.values()) < UNVERIF_COV:
                verdict = "unverifiable"
            elif (best_o and cov[best_o] > cov[sp] + COV_MARGIN
                  and cos.get(best_o, -1) > cos.get(sp, -1) + COS_MARGIN):
                verdict = "other_better"
            else:
                verdict = "supported"
            att_out.append({"hearing_id": hid, "sent_i": s["sent_i"], "text": text,
                            "attributed_to": sp, "verdict": verdict,
                            "cov_attributed": round(cov[sp], 4),
                            "cov_best_other": round(cov[best_o], 4) if best_o else None,
                            "best_other": best_o,
                            "cos_attributed": round(cos.get(sp, -1), 4),
                            "cos_best_other": round(cos.get(best_o, -1), 4) if best_o else None})

        for sp, txt in floor.items():
            a = attrs.get(surname(sp), {})
            omit_out.append({"hearing_id": hid, "speaker": sp, "floor_chars": len(txt),
                             "named_in_article": sp in named,
                             "role": a.get("role_rule"), "gender": a.get("gender")})

    if args.validate:
        import random as _rnd
        rng = _rnd.Random(0)
        flagged_true = sum(1 for r in att_out if r["verdict"] == "other_better")
        swapped = 0
        for r in att_out:
            pool = [k for k in (r["best_other"],) if k] or []
            if not pool:
                continue
            # the swap: pretend the article credited the best competing speaker.
            # Detection means the true speaker now looks clearly better.
            cov_s, cov_t = r["cov_best_other"], r["cov_attributed"]
            cos_s, cos_t = r["cos_best_other"], r["cos_attributed"]
            if cov_t > cov_s + COV_MARGIN and cos_t > cos_s + COS_MARGIN:
                swapped += 1
        n = len(att_out)
        print("\n=== DETECTOR VALIDATION ===")
        print(f"true attributions flagged as other_better (false-positive rate): {flagged_true}/{n} = {flagged_true/n:.1%}")
        print(f"swapped attributions that WOULD be flagged (detection rate):     {swapped}/{n} = {swapped/n:.1%}")
        print("A detection rate near the false-positive rate means the signals cannot separate\n"
              "speakers within a hearing.")

    with (OUT / "attributions.jsonl").open("w") as f:
        for r in att_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "omissions.jsonl").open("w") as f:
        for r in omit_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(att_out)} attributed sentences over {len(ids)} hearings; {len(omit_out)} speaker-hearing rows")
    from collections import Counter
    print(Counter(r["verdict"] for r in att_out))


if __name__ == "__main__":
    main()
