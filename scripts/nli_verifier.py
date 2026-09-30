"""An off-the-shelf multilingual NLI cross-encoder as a competing detector.

Runs on the same evidence bases the probe is evaluated on:
  corruptions      generated opinions and their rule-corrupted variants
                   (probe_fresh_corruptions.py), against the same four windows the
                   probe saw; paired win rate and pooled ROC-AUC per corruption type,
                   beside the lexical referee (content-word containment) and the probe.
  benchmark        the dataset's NLI half with its human label, against each opinion's
                   four `chunks_proximos`.
A claim is scored against each window separately and the best window is kept
(SummaC-ZS: a claim is supported if some passage supports it). Two scores per item:
max P(entailment) and max of P(entailment) - P(contradiction). Higher means more
supported, so the detector score is the negation.

Usage: uv run python scripts/nli_verifier.py [corruptions|benchmark ...]
(default: both). Env: OCARANDU_DEVICE, OCARANDU_NLI_MODEL (default
MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7), OCARANDU_NLI_BATCH, OCARANDU_NLI_MAXLEN,
OCARANDU_NLI_PF_TAG (probe_fresh tag, default llama_31_8b_instruct).
Writes data/publichearingbr/nli_verifier/{corruptions_<tag>,benchmark}.csv and
appends the tables to stats.md there.
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from attribution_generation import containment  # noqa: E402

from ocarandu.data.publichearingbr import load_nli_opinions  # noqa: E402

DATA = ROOT / "data" / "publichearingbr"
OUT = DATA / "nli_verifier"
GEN = DATA / "longcontext_generation"
PF = DATA / "probe_fresh"
DEVICE = os.environ.get("OCARANDU_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
REPO = os.environ.get("OCARANDU_NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
BATCH = int(os.environ.get("OCARANDU_NLI_BATCH", "64"))
MAXLEN = int(os.environ.get("OCARANDU_NLI_MAXLEN", "384"))
PF_TAG = os.environ.get("OCARANDU_NLI_PF_TAG", "llama_31_8b_instruct")
# probe columns to print beside the NLI ones, by probe_fresh tag
PROBE_COL = {"llama_31_8b_instruct": "p12", "qwen3_4b_instruct_2507": "p21"}


class NLI:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(REPO)
        self.m = AutoModelForSequenceClassification.from_pretrained(REPO, dtype=torch.float16).to(DEVICE).eval()
        lab = {v.lower(): k for k, v in self.m.config.id2label.items()}
        self.i_ent = next(i for k, i in lab.items() if k.startswith("entail"))
        self.i_con = next(i for k, i in lab.items() if k.startswith("contra"))

    def probs(self, pairs):
        """pairs = [(premise, hypothesis)]; returns (n, 3) probabilities, length-sorted batching."""
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(order), BATCH):
                idx = order[s:s + BATCH]
                # longest_first: the premise (a 100-word window) is what normally gets cut, but a very
                # long generated claim can exceed MAXLEN on its own, and "only_first" then raises
                enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation="longest_first",
                               max_length=MAXLEN, padding=True, return_tensors="pt").to(DEVICE)
                p = torch.softmax(self.m(**enc).logits.float(), -1).cpu().numpy()
                for j, i in enumerate(idx):
                    out[i] = p[j]
                if (s // BATCH) % 50 == 0:
                    print(f"  {s}/{len(order)}", file=sys.stderr, flush=True)
        return out

    def score_items(self, items):
        """items = [(windows, claim)] -> (entail_max, entail_minus_contra_max) per item."""
        pairs, owner = [], []
        for i, (wins, claim) in enumerate(items):
            for w in (wins or [""]):
                pairs.append((w, claim))
                owner.append(i)
        p = self.probs(pairs)
        ent = np.full(len(items), -1.0)
        ec = np.full(len(items), -2.0)
        for k, i in enumerate(owner):
            ent[i] = max(ent[i], float(p[k, self.i_ent]))
            ec[i] = max(ec[i], float(p[k, self.i_ent] - p[k, self.i_con]))
        return ent, ec


def gen_windows(tags):
    """(run tag, sample_id, speaker) -> the participant's windows."""
    out = {}
    for t in tags:
        f = GEN / t / "generations.jsonl"
        if not f.exists():
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            out[(t, r["sample_id"], r["speaker"])] = r["chunks"]
    return out


def auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        t = s == v
        if t.sum() > 1:
            ranks[t] = ranks[t].mean()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# --------------------------------------------------------------------------- corruptions
def task_corruptions(nli, L):
    rows = [json.loads(l) for l in open(PF / PF_TAG / "rows.jsonl", encoding="utf-8")]
    sc = {}
    p = PF / PF_TAG / "scores.csv"
    if p.exists():
        col = PROBE_COL.get(PF_TAG, "p14")
        for r in csv.DictReader(open(p, encoding="utf-8")):
            sc[int(r["idx"])] = float(r[col]) if col in r else np.nan
    wins = gen_windows(sorted({r["run"] for r in rows}))
    items = []
    for r in rows:
        w = wins.get((r["run"], r["sample_id"], r["speaker"]), [])
        shown = [w[i] for i in r["chunk_idx"] if i < len(w)] if w else []
        items.append((shown, r["text"]))
    ent, ec = nli.score_items(items)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"corruptions_{PF_TAG}.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["idx", "pair_id", "variant", "run", "condition", "sample_id", "speaker", "opinion_idx",
                      "nli_entail_max", "nli_ec_max", "containment_shown", "probe"])
        for r, e, c in zip(rows, ent, ec):
            wtr.writerow([r["idx"], r["pair_id"], r["variant"], r["run"], r["condition"], r["sample_id"],
                          r["speaker"], r["opinion_idx"], f"{e:.5f}", f"{c:.5f}",
                          f"{r['containment_shown']:.5f}", f"{sc.get(r['idx'], float('nan')):.5f}"])
    # paired: within a pair_id, is the corrupted variant ranked as less supported than its original?
    by_pair = defaultdict(dict)
    for i, r in enumerate(rows):
        by_pair[r["pair_id"]][r["variant"]] = (ent[i], ec[i], r["containment_shown"], sc.get(r["idx"], np.nan))
    variants = sorted({r["variant"] for r in rows} - {"original"})
    L += ["", f"## Corruptions ({PF_TAG}): paired win rate against the claim's own original", "",
          "A win = the detector ranks the corrupted claim as less supported than the uncorrupted one. "
          "Ties count as half a win (the referee's ties are the point: the corruption keeps the content words). "
          "All three detectors see the SAME four windows.", "",
          "| corruption | pairs | NLI entail | NLI entail−contra | lexical containment | probe | NLI ties |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for v in variants:
        pairs = [(d["original"], d[v]) for d in by_pair.values() if "original" in d and v in d]
        if not pairs:
            continue
        def win(j, lower_is_supported=False):
            w = t = 0
            for o, c in pairs:
                a, b = o[j], c[j]
                if np.isnan(a) or np.isnan(b):
                    continue
                if a == b:
                    t += 1
                elif (b < a) != lower_is_supported:
                    w += 1
            n = sum(1 for o, c in pairs if not (np.isnan(o[j]) or np.isnan(c[j])))
            return (w + t / 2) / n if n else float("nan"), t / n if n else float("nan")
        we, tie = win(0)
        wc, _ = win(1)
        wl, _ = win(2)
        wp, _ = win(3, lower_is_supported=True)  # probe: higher score = more hallucinated
        L.append(f"| {v} | {len(pairs)} | {we:.3f} | {wc:.3f} | {wl:.3f} | {wp:.3f} | {tie:.2f} |")
    L += ["", "Pooled ROC-AUC, corrupted (=1) vs original (=0), per corruption:", "",
          "| corruption | NLI entail | NLI entail−contra | lexical | probe |", "|---|---:|---:|---:|---:|"]
    for v in variants:
        idx = [i for i, r in enumerate(rows) if r["variant"] in ("original", v)]
        y = [int(rows[i]["variant"] == v) for i in idx]
        L.append(f"| {v} | {auc([-ent[i] for i in idx], y):.3f} | {auc([-ec[i] for i in idx], y):.3f} | "
                 f"{auc([-rows[i]['containment_shown'] for i in idx], y):.3f} | "
                 f"{auc([sc.get(rows[i]['idx'], np.nan) for i in idx], y):.3f} |")


# --------------------------------------------------------------------------- benchmark
def task_benchmark(nli, L):
    ops = [o for o in load_nli_opinions() if o.opinion.strip() and o.context_chunks]
    ent, ec = nli.score_items([(list(o.context_chunks), o.opinion) for o in ops])
    y = [int(o.is_hallucination) for o in ops]
    lex = [containment(o.opinion, list(o.context_chunks)) for o in ops]
    with open(OUT / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["sample_id", "opinion", "label", "nli_entail_max", "nli_ec_max", "containment"])
        for o, e, c, lx, yy in zip(ops, ent, ec, lex, y):
            wtr.writerow([o.sample_id, o.opinion, yy, f"{e:.5f}", f"{c:.5f}", f"{lx:.5f}"])
    L += ["", "## Benchmark half (4,237 GPT-4 opinions, the dataset's own human label)", "",
          "| detector | ROC-AUC |", "|---|---:|",
          f"| NLI entail (max over the 4 chunks) | {auc([-e for e in ent], y):.3f} |",
          f"| NLI entail − contradiction | {auc([-c for c in ec], y):.3f} |",
          f"| lexical content-word containment | {auc([-x for x in lex], y):.3f} |",
          "", "Compare with the in-domain residual probe's out-of-fold ROC-AUC (scripts/run_publichearingbr_probe.py)."]


def main():
    tasks = sys.argv[1:] or ["corruptions", "benchmark"]
    OUT.mkdir(parents=True, exist_ok=True)
    nli = NLI()
    L = [f"# An off-the-shelf NLI cross-encoder as a competing detector", "",
         f"Model `{REPO}`, max length {MAXLEN}, scored on {DEVICE}. Generated by `scripts/nli_verifier.py`.", ""]
    for t in tasks:
        print(f"=== {t}", file=sys.stderr, flush=True)
        if t == "corruptions":
            task_corruptions(nli, L)
        elif t == "benchmark":
            task_benchmark(nli, L)
        else:
            raise SystemExit(f"unknown task {t}")
    p = OUT / "stats.md"
    old = p.read_text(encoding="utf-8") if p.exists() else ""
    p.write_text(("\n".join(L) + "\n") if not old else old + "\n\n---\n\n" + "\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
