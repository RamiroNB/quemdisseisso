"""Analyse a blind annotation round built by build_synth3_sets.py.

Merges the rater files data/publichearingbr/annotation/<name>/<raw>/set-*.jsonl with
the key <name>_key/key.csv and reports, in this order:

  0. rater quality   accuracy on the calibration items (dataset GPT-4 opinions shown with
                     their own four chunks and the benchmark's human label), pooled and per
                     rater, with precision/recall against that label; agreement between the
                     two raters of the replicated participants (Cohen's kappa).
  1. rates by arm    unsupported share at the opinion level, share of participants with at
                     least one unsupported opinion, form-issue share.
  2. paired tests    for every pair of arms, exact McNemar on the participant-level outcome
                     "this participant's summary has >= 1 unsupported opinion"; the arms
                     share the participant and the evidence shown to the rater.
  3. vs the referee  ROC-AUC of 1 - containment for the rater's "nao_sustentada", and
                     precision/recall of the referee's flag (containment < 0.3), per arm.
  4. vs the probe    ROC-AUC of the residual probe for the same label, for the arms that
                     probe_fresh/<tag>/scores.csv covers (PROBE_SRC below).

Replicated blocks are excluded from the rate and paired tables (they would count one
participant twice) and used only for the agreement figures.

Usage: [OCARANDU_SYNTH3_NAME=synth3] [OCARANDU_SYNTH3_RAW=raw] [OCARANDU_SYNTH3_PROBE_COL=p14]
       uv run python scripts/synth3_results.py
Writes <name>/results.md, or results_<raw>.md for another rater directory. CPU only.
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
# OCARANDU_SYNTH3_NAME selects the round (the builder's --name): "synth3" is the three-arm
# Llama-3.1 round; any other round takes its arms from its key.
NAME = os.environ.get("OCARANDU_SYNTH3_NAME", "synth3")
# OCARANDU_SYNTH3_RAW names the rater directory inside the round (default "raw"), so a second set of
# raters can annotate the same sets separately; the report then goes to results_<raw>.md.
RAW = os.environ.get("OCARANDU_SYNTH3_RAW", "raw")
# probe layer chosen per model on its own held-out data; the default column is p14 for the Llama-3.1
# round "synth3" and p21 (Qwen3-4B) for any other round. Set it for a round of another model; a wrong
# layer is not detected.
PROBE_COL = os.environ.get("OCARANDU_SYNTH3_PROBE_COL", "p14" if NAME == "synth3" else "p21")
D = ROOT / "data" / "publichearingbr" / "annotation" / NAME
KEY = ROOT / "data" / "publichearingbr" / "annotation" / f"{NAME}_key" / "key.csv"
PF = ROOT / "data" / "publichearingbr" / "probe_fresh"
POS, NEG, UNK = "sustentada", "nao_sustentada", "nao_da_para_dizer"
def _arms_from_key():
    """Arms found in the round's key, baseline first and the rest in alphabetical order."""
    import csv as _csv
    seen = []
    for r in _csv.DictReader(open(KEY, encoding="utf-8-sig")):
        if r["kind"] == "pipeline" and r["arm"] not in seen:
            seen.append(r["arm"])
    return sorted(seen, key=lambda a: (a != "baseline", a))
ARMS = ["baseline", "steered", "gated"] if NAME == "synth3" else _arms_from_key()
# (arm label -> (probe_fresh tag, run tag, condition)) for the optional probe column;
# Llama-3.1 runs unless the round name contains "qwen"
PROBE_SRC = {
    "baseline": ("llama_31_8b_instruct_samples", "llama_31_8b_instruct_dial", "baseline"),
    "steered": ("llama_31_8b_instruct_samples", "llama_31_8b_instruct_dial", "halluc:1.0"),
} if "qwen" not in NAME else {
    "baseline": ("qwen3_4b_instruct_2507", "qwen3_4b_instruct_2507_lc_xfitA", "baseline"),
    "steered": ("qwen3_4b_instruct_2507", "qwen3_4b_instruct_2507_lc_xfitA", "halluc:1.0"),
}


def kappa(pairs):
    if not pairs:
        return float("nan")
    labs = sorted({x for p in pairs for x in p})
    n = len(pairs)
    obs = sum(a == b for a, b in pairs) / n
    exp = sum((sum(a == l for a, _ in pairs) / n) * (sum(b == l for _, b in pairs) / n) for l in labs)
    return (obs - exp) / (1 - exp) if exp < 1 else float("nan")


def mcnemar(b, c):
    return binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


def auc(scores, labels):
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    if len(set(y.tolist())) < 2:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    for v in np.unique(s):
        t = s == v
        if t.sum() > 1:
            ranks[t] = ranks[t].mean()
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def probe_scores():
    """(arm, sample_id, speaker, opinion_idx) -> P(halluc) from the hearing-disjoint probes."""
    out = {}
    for arm, (pf_tag, run_tag, cond) in PROBE_SRC.items():
        p = PF / pf_tag / "scores.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("variant") != "original" or r.get("run") != run_tag or r.get("condition") != cond:
                continue
            col = PROBE_COL if PROBE_COL in r else next((k for k in r if k.startswith("p") and k[1:].isdigit()), None)
            if col:
                out[(arm, int(r["sample_id"]), r["speaker"], int(r["opinion_idx"]))] = float(r[col])
    return out


def main():
    key = {r["id"]: r for r in csv.DictReader(open(KEY, encoding="utf-8-sig"))}
    labels, files = {}, sorted((D / RAW).glob("set-*.jsonl"))
    if not files:
        raise SystemExit(f"no rater files in {D / RAW}")
    bad = 0
    for p in files:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if d.get("id") in key and d.get("support") in (POS, NEG, UNK):
                labels[(p.stem, d["id"])] = d
            else:
                bad += 1
    L = [f"# Blind annotation ({NAME}): " + " vs ".join(ARMS), "",
         f"{len(files)} rater files, {len(labels)} labelled items ({bad} unparseable or unknown-id lines skipped). "
         f"Raters are independent agents, one per set; blind to the arm; the arms of a participant were shown "
         f"the SAME evidence (all of that person's 100-word windows) and never appeared in the same set.", ""]

    rows = [(setname, key[iid], d) for (setname, iid), d in labels.items()]

    # ---------- 0. rater quality ----------
    calib = [(s, k, d) for s, k, d in rows if k["kind"] == "calib"]
    per_rater = defaultdict(lambda: [0, 0])
    hit = tot = 0
    for s, k, d in calib:
        if d["support"] == UNK:
            continue
        gold = int(k["human_label_dataset"])  # 1 = hallucinated per the benchmark's annotator
        pred = int(d["support"] == NEG)
        per_rater[s][0] += int(pred == gold)
        per_rater[s][1] += 1
        hit += int(pred == gold)
        tot += 1
    accs = sorted(h / n for h, n in per_rater.values() if n)
    L += ["## 0. Rater quality", "",
          f"Calibration items (dataset opinions with the benchmark's own human label, half hallucinated by design): "
          f"**{hit}/{tot} correct = {hit / max(1, tot):.3f}**. Per-rater accuracy: median {np.median(accs):.2f}, "
          f"range {min(accs):.2f}–{max(accs):.2f} over {len(accs)} raters."]
    if tot:
        gold = [int(k["human_label_dataset"]) for s, k, d in calib if d["support"] != UNK]
        pred = [int(d["support"] == NEG) for s, k, d in calib if d["support"] != UNK]
        tp = sum(g and p for g, p in zip(gold, pred)); fp = sum((not g) and p for g, p in zip(gold, pred))
        fn = sum(g and (not p) for g, p in zip(gold, pred))
        L.append(f"On those items the raters call {sum(pred)}/{len(pred)} unsupported: precision "
                 f"{tp / max(1, tp + fp):.2f}, recall {tp / max(1, tp + fn):.2f} against the human label.")
    # double-annotated blocks: the replica of a block carries a different item id,
    # so pair on (participant, arm, opinion index)
    byitem = defaultdict(list)
    for s, k, d in rows:
        if k["kind"] == "pipeline":
            byitem[(k["sample_id"], k["speaker"], k["arm"], k["opinion_idx"])].append((k["replicate"], d["support"]))
    pairs = [(dict(v)["0"], dict(v)["1"]) for v in byitem.values()
             if len(v) == 2 and {r for r, _ in v} == {"0", "1"}]
    agree = np.mean([a == b for a, b in pairs]) if pairs else float("nan")
    nao = [(a, b) for a, b in pairs if NEG in (a, b)]
    L += ["", f"Double-annotated items (the 20 replicated participants, both raters blind and in different sets): "
             f"{len(pairs)} items, raw agreement {agree:.3f}, Cohen's kappa **{kappa(pairs):.3f}**; of the "
             f"{len(nao)} items either rater called unsupported, both did in {sum(a == b for a, b in nao)}.", ""]

    # ---------- 1. rates by arm (replicates excluded) ----------
    first = {}
    for s, k, d in rows:
        if k["kind"] != "pipeline" or k["replicate"] != "0":
            continue
        first[k["id"]] = (k, d)
    by_arm_op = defaultdict(list)
    by_part = defaultdict(dict)  # (sample_id, speaker) -> arm -> [flags]
    form = defaultdict(list)
    for k, d in first.values():
        arm = k["arm"]
        if d["support"] != UNK:
            by_arm_op[arm].append(int(d["support"] == NEG))
        form[arm].append(int(str(d.get("form_issue", 0)) in ("1", "True", "true")))
        by_part[(k["sample_id"], k["speaker"])].setdefault(arm, []).append(d["support"])
    L += ["## 1. Rates by arm", "",
          "| arm | opinions judged | unsupported (opinion level) | cannot tell | participants | with >=1 unsupported | form issues |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    part_flag = {}
    for arm in ARMS:
        ops = by_arm_op[arm]
        unk = sum(1 for k, d in first.values() if k["arm"] == arm and d["support"] == UNK)
        parts = {p: v[arm] for p, v in by_part.items() if arm in v}
        flags = {p: int(any(x == NEG for x in v)) for p, v in parts.items()}
        part_flag[arm] = flags
        L.append(f"| {arm} | {len(ops)} | {np.mean(ops):.3f} ({sum(ops)}) | {unk} | {len(flags)} | "
                 f"{np.mean(list(flags.values())):.3f} ({sum(flags.values())}) | {np.mean(form[arm]):.3f} |")
    L.append("")

    # ---------- 2. paired tests ----------
    L += ["## 2. Paired comparison, participant level (exact McNemar)", "",
          "Outcome: this participant's summary contains at least one opinion the rater called unsupported. "
          "The arms share the participant and the evidence shown, so the pairing is exact.", "",
          "| comparison | pairs | rate A | rate B | fixed (A yes, B no) | broken (A no, B yes) | p |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for a, b in [(x, y) for i, x in enumerate(ARMS) for y in ARMS[i + 1:]]:
        if a not in part_flag or b not in part_flag:
            continue
        common = sorted(set(part_flag[a]) & set(part_flag[b]))
        fa = [part_flag[a][p] for p in common]
        fb = [part_flag[b][p] for p in common]
        fixed = sum(x and not y for x, y in zip(fa, fb))
        broken = sum((not x) and y for x, y in zip(fa, fb))
        L.append(f"| {a} → {b} | {len(common)} | {np.mean(fa):.3f} | {np.mean(fb):.3f} | {fixed} | {broken} | "
                 f"{mcnemar(fixed, broken):.3g} |")
    L.append("")
    L.append("A significant p with `fixed` > `broken` is the only reading under which the intervention reduces "
             "reader-visible unsupported content on this task.")

    # ---------- 3. vs the lexical referee, 4. vs the probe ----------
    pr = probe_scores()
    L += ["", "## 3. Detectors against the rater's label, per arm", "",
          f"| arm | items | unsupported | referee AUC (1-containment) | referee P/R at <0.3 | "
          f"probe AUC (L{PROBE_COL[1:]}) |",
          "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        rs = [(k, d) for k, d in first.values() if k["arm"] == arm and d["support"] != UNK]
        y = [int(d["support"] == NEG) for k, d in rs]
        ref = [1 - float(k["referee_containment"]) for k, d in rs]
        flag = [int(k["referee_flag"]) for k, d in rs]
        tp = sum(a and b for a, b in zip(y, flag)); fp = sum((not a) and b for a, b in zip(y, flag))
        fn = sum(a and (not b) for a, b in zip(y, flag))
        ps = [pr.get((arm, int(k["sample_id"]), k["speaker"], int(k["opinion_idx"])), float("nan")) for k, d in rs]
        pa = auc(ps, y) if not all(np.isnan(ps)) else float("nan")
        L.append(f"| {arm} | {len(rs)} | {sum(y)} | {auc(ref, y):.3f} | "
                 f"{tp / max(1, tp + fp):.2f} / {tp / max(1, tp + fn):.2f} | "
                 f"{'—' if np.isnan(pa) else f'{pa:.3f}'} |")
    L += ["", "Referee = best-window content-word containment, the runs' own scorer. Probe = the in-domain "
              "residual-stream probe, hearing-disjoint, on the same opinion re-read in the probe's training "
              "format; blank where that arm was not extracted.", ""]

    # counts for the record
    L.append(f"Label distribution overall: {dict(Counter(d['support'] for s, k, d in rows if k['kind'] == 'pipeline'))}.")
    text = "\n".join(L) + "\n"
    (D / ("results.md" if RAW == "raw" else f"results_{RAW}.md")).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
