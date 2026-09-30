"""Rebuild PublicHearingBR's NLI evidence chunks with the speaker of each chunk.

The NLI file's `chunks_proximos` are transcript passages with the stenographic
turn headers ("O SR. ... - ") stripped, so the evidence does not say who was
speaking. Each chunk is located in the full transcript (LDS file) and prefixed
with the speaker of the closest preceding header, "[SPEAKER] chunk"
("[ORADOR NÃO IDENTIFICADO]" if the chunk is not found or no header precedes it).
Each row also records whether the attributed speaker holds the floor in the
evidence (fuzzy name match). A header names whoever holds the floor and a chunk
may quote someone else, so a mismatch suggests misattribution but does not prove it.

Usage: uv run python scripts/build_tagged_evidence.py
Output: data/publichearingbr/tagged_evidence.jsonl, one row per opinion with four
chunks, keyed like the NLI loader (sample_id, nome, opinion).
"""

import json
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from ocarandu.data.publichearingbr import DEFAULT_NLI_PATH, load_nli_opinions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "publichearingbr" / "tagged_evidence.jsonl"
LDS_PATH = Path(str(DEFAULT_NLI_PATH).replace("_NLI", "_LDS"))

# "O SR. LEONARDO MONTEIRO(Bloco/PT - MG) - ", "A SRA. PRESIDENTA(Erika Kokay. PT - DF) - ",
# "O SR. CRISTIANO NABUCO - ", "O SR. PRESIDENTE(Aureo Ribeiro. Bloco/SOLIDARIEDADE - RJ) - "
HEADER = re.compile(r"(O SR\.|A SRA\.)\s+([A-ZÀ-Ú][A-ZÀ-Ú' .\-]{1,80}?)\s*(?:\(([^)]{0,160})\))?\s*-\s")


def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_header(m):
    """Return (speaker name, {honorific, detail, chair_role}); for a chair header
    (PRESIDENTE, RELATOR, ...) the person's name is taken from the parenthesis."""
    honorific, name, paren = m.group(1), m.group(2).strip(" ."), (m.group(3) or "").strip()
    role = None
    if name.upper().startswith(("PRESIDENT", "RELATOR", "VICE-PRESIDENT")):
        role = name.title()
        # "Aureo Ribeiro. Bloco/SOLIDARIEDADE - RJ" -> name before the first '.'
        person = paren.split(".")[0].strip() if paren else ""
        name, paren = (person or name), paren[len(person):].strip(". ") if person else paren
    return name.title() if name.isupper() else name, {"honorific": honorific, "detail": paren, "chair_role": role}


def name_match(a, b):
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    pa = [p for p in a.split() if len(p) > 2 and p not in {"dos", "das", "del", "van", "von"}]
    pb = [p for p in b.split() if len(p) > 2 and p not in {"dos", "das", "del", "van", "von"}]
    if len(pa) >= 2 and len(pb) >= 2 and pa[0] == pb[0] and pa[-1] == pb[-1]:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.85 or (len(pa) >= 1 and len(pb) >= 1 and pa[-1] == pb[-1] and pa[0][0] == pb[0][0])


def main():
    lds = {r["id"]: r["transcricao"] for r in (json.loads(l) for l in open(LDS_PATH))}
    # per transcript: whitespace-collapsed text and the header positions in that text
    prepared = {}
    for sid, t in lds.items():
        tn = re.sub(r"\s+", " ", t)
        heads = [(m.start(), *parse_header(m)) for m in HEADER.finditer(tn)]
        prepared[sid] = (tn, heads)
    print(f"{len(prepared)} transcripts; turn headers per transcript: median "
          f"{sorted(len(h) for _, h in prepared.values())[len(prepared)//2]}", file=sys.stderr)

    ops = [o for o in load_nli_opinions() if len(o.context_chunks) == 4]
    stats = Counter()
    with open(OUT, "w") as f:
        for o in ops:
            tn, heads = prepared[o.sample_id]
            tagged, speakers, located = [], [], 0
            for c in o.context_chunks:
                cn = re.sub(r"\s+", " ", c).strip()
                probe_txt = cn[:160]
                i = tn.find(probe_txt)
                if i < 0:  # try a mid-chunk anchor (leading fragments are sometimes cut)
                    mid = cn[len(cn) // 3: len(cn) // 3 + 120]
                    i = tn.find(mid) if len(mid) > 40 else -1
                if i < 0:
                    tagged.append(f"[ORADOR NÃO IDENTIFICADO] {c}"); speakers.append(None); stats["chunk_not_located"] += 1
                    continue
                located += 1
                prev = None
                for pos, name, meta in heads:
                    if pos <= i:
                        prev = (name, meta)
                    else:
                        break
                if prev is None:
                    tagged.append(f"[ORADOR NÃO IDENTIFICADO] {c}"); speakers.append(None); stats["no_header"] += 1
                    continue
                name, meta = prev
                label = name + (f" ({meta['chair_role']})" if meta["chair_role"] else "")
                tagged.append(f"[{label}] {c}"); speakers.append(name); stats["tagged"] += 1
            known = [s for s in speakers if s]
            matches = [name_match(o.person_name, s) for s in known]
            row = {
                "sample_id": o.sample_id, "nome": o.person_name, "cargo": o.person_role, "opinion": o.opinion,
                "label": int(o.is_hallucination), "tagged_chunks": tagged, "chunk_speakers": speakers,
                "n_located": located, "n_chunks_by_attributed_speaker": int(sum(matches)),
                "attributed_speaker_in_evidence": bool(any(matches)),
                "majority_speaker": Counter(known).most_common(1)[0][0] if known else None,
                "majority_is_attributed": bool(known) and name_match(o.person_name, Counter(known).most_common(1)[0][0]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["opinions"] += 1
            stats["attributed_in_evidence"] += row["attributed_speaker_in_evidence"]
            stats["majority_is_attributed"] += row["majority_is_attributed"]
            stats[f"attributed_in_evidence|label={row['label']}"] += row["attributed_speaker_in_evidence"]
            stats[f"n|label={row['label']}"] += 1
    n = stats["opinions"]
    print(f"\nopinions {n}; chunks tagged {stats['tagged']}/{4*n} ({stats['tagged']/(4*n):.1%}), not located {stats['chunk_not_located']}, no header {stats['no_header']}")
    print(f"attributed speaker holds the floor in >=1 evidence chunk: {stats['attributed_in_evidence']/n:.1%}; is the majority speaker: {stats['majority_is_attributed']/n:.1%}")
    for lab in (0, 1):
        print(f"  label={lab} ({'supported' if lab==0 else 'hallucinated'}): attributed speaker in evidence {stats[f'attributed_in_evidence|label={lab}']/stats[f'n|label={lab}']:.1%} (n={stats[f'n|label={lab}']})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
