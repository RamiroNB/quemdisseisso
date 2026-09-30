"""Register and copying diagnostics per condition, beside coverage.

Coverage is lexical: an output that quotes the transcript instead of reporting on it scores high
without doing the task (the prompt asks for a third-person report). Per condition:
  first_person   share of generations with first-person markers (nós, eu, temos, precisamos,
                 defendemos, nossa ...): the transcript's register, not a summary's
  third_person   share that opens like a report (a, o, os, foi, segundo, há ...) or contains a
                 reporting verb (afirmou, destacou, defendeu ...)
  ngram_copy     share of the generation's word 4-grams that occur verbatim in the shown chunks
                 (0 = fully abstractive, 1 = pasted)
  odd_chars      share with characters outside Portuguese orthography (fluency damage such as 'quiśeja')

Usage: uv run python scripts/content_hallucination_register.py <run_dir_tag> [...]
Reads data/publichearingbr/content_hallucination_generation/<tag>/generations.jsonl; prints one table per run.
"""
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIRST = re.compile(r"\b(nós|eu|nossa|nosso|nossas|nossos|temos|precisamos|defendemos|manifestamos|queremos|estamos|vamos|fizemos|conseguimos|acreditamos|entendemos|propomos|pedimos|sabemos)\b", re.I)
THIRD = re.compile(r"^\s*(a |o |os |as |foi |foram |durante |na |no |em |segundo |conforme |há |houve )|\b(afirm|destac|defend|propôs|propos|ressalt|relat|discutid|mencion|apont|argument|critic|question)\w*", re.I)
ODD = re.compile(r"[^\w\s.,;:!?()\-—–\"'“”‘’«»/%ºª§áéíóúâêôãõàçüÁÉÍÓÚÂÊÔÃÕÀÇÜ]")


def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    return re.sub(r"\s+", " ", "".join(c for c in s if not unicodedata.combining(c))).lower().strip()


def ngrams(text, n=4):
    w = norm(text).split()
    return {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def main():
    for tag in sys.argv[1:]:
        rows = [json.loads(l) for l in open(ROOT / "data/publichearingbr/content_hallucination_generation" / tag / "generations.jsonl")]
        by = defaultdict(list)
        for r in rows:
            by[r["condition"]].append(r)
        print(f"\n### {tag}")
        print(f"{'condition':14s} {'cov':>6s} {'1st-p':>6s} {'3rd-p':>6s} {'4gram':>6s} {'odd':>5s} {'len':>4s}")
        for cond in ["baseline"] + sorted(c for c in by if c != "baseline"):
            g = by[cond]
            cov = np.mean([r["coverage"] or 0.0 for r in g])
            fp = np.mean([bool(FIRST.search(r["generation"])) for r in g])
            tp = np.mean([bool(THIRD.search(r["generation"])) for r in g])
            ng = []
            for r in g:
                gn = ngrams(r["generation"])
                cn = set().union(*(ngrams(c) for c in r["chunks"]))
                ng.append(len(gn & cn) / len(gn) if gn else 0.0)
            odd = np.mean([bool(ODD.search(r["generation"])) for r in g])
            print(f"{cond:14s} {cov:6.3f} {fp:6.2f} {tp:6.2f} {np.mean(ng):6.3f} {odd:5.2f} {np.mean([r['n_chars'] for r in g]):4.0f}")


if __name__ == "__main__":
    main()
