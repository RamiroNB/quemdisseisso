"""Line-level audit of a whole-transcript run: repetition, parser artefacts, unique-line referee.

A line repeated n times counts n times in the per-line shares, and conditions can differ in
how much the model repeats itself, so the referee classes are also reported on unique
lines (distinct hearing, name and normalised text). Also counts lines whose parsed
"name" is a format artefact ("Opinião", a raw "O SR./A SRA." header, a preamble).

Usage: uv run python scripts/fulltranscript_lines.py <tag>
Reads data/publichearingbr/fulltranscript_generation/<tag>/generations.jsonl and writes
lines.md in the same directory.
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = sys.argv[1]
DIR = ROOT / "data" / "publichearingbr" / "fulltranscript_generation" / TAG
ARTEFACT = re.compile(r"(?i)^(opini[aã]o|o sr\.?|a sra\.?|aqui est|seguem|abaixo)")


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def main():
    rows = [json.loads(l) for l in open(DIR / "generations.jsonl")]
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    out = [f"# Line-level audit: {TAG}", "",
           "Unique = distinct (hearing, name, normalised text). Artefact = parsed name is a format token, not a person.", "",
           "| condition | hearings | lines | unique | dup share | hearings ≥30% dup | words/line | cov_own all | cov_own unique | artefact names | grounded (unique) | misattributed (unique) | unsupported (unique) | empty outputs |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        lines = [(r["sample_id"], o) for r in rs for o in r["opinions"]]
        seen, uniq = set(), []
        for h, o in lines:
            k = (h, o["name"].lower(), norm(o["text"]))
            if k not in seen:
                seen.add(k)
                uniq.append(o)
        per_h = []
        for r in rs:
            L = [norm(o["text"]) for o in r["opinions"]]
            per_h.append(1 - len(set(L)) / len(L) if L else 0.0)
        named_all = [o["cov_own"] for _, o in lines if o["cls"] != "unknown_speaker"]
        named_u = [o["cov_own"] for o in uniq if o["cls"] != "unknown_speaker"]
        art = sum(bool(ARTEFACT.match(o["name"].strip())) for o in uniq)
        words = [len(o["text"].split()) for _, o in lines]
        share = lambda cls: (sum(o["cls"] == cls for o in uniq) / len(uniq)) if uniq else float("nan")
        empty = sum(r["n_opinions"] == 0 for r in rs)
        out.append(f"| {c} | {len(rs)} | {len(lines)} | {len(uniq)} | {1 - len(uniq) / len(lines):.2f} | "
                   f"{sum(p >= 0.3 for p in per_h)} | {st.mean(words):.1f} | {st.mean(named_all):.3f} | {st.mean(named_u):.3f} | "
                   f"{art} | {share('grounded'):.3f} | {share('misattributed'):.3f} | {share('unsupported'):.3f} | {empty} |")
    (DIR / "lines.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
