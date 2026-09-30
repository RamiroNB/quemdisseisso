"""Paired tests for a longcontext_speaker_generation.py run.

Reads data/publichearingbr/longcontext_generation/<run_dir_tag>/generations.jsonl
and writes paired_tests.md next to it. Every condition is paired positionally
with the baseline (the script exits if the (sample_id, speaker) order differs).
Tests: sign-flip permutation by hearing, Holm across the conditions of each
metric, exact McNemar on "the item has an ungrounded opinion or abstains", and
specificity = steered minus the mean of the random directions at the same alpha.
Item-level outcomes:
  coverage          mean best-window containment of the generated opinions
  ungrounded        share of generated opinions with containment < 0.3
  gold_recall       share of the participant's gold opinions with >= 50% of their
                    content words in the generation (gold_recall_soft: mean fraction)
  n_opinions        opinions produced; read beside coverage, as saying less is not a win
Missing values (e.g. abstentions) count as coverage 0, ungrounded 1, recall 0.

Usage: uv run python scripts/longcontext_stats.py <run_dir_tag> [n_perm, default 20000]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content_hallucination_stats import block_perm_p, holm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAG = sys.argv[1]
N_PERM = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
DIR = ROOT / "data" / "publichearingbr" / "longcontext_generation" / TAG


def fields(rows):
    cov = np.array([r["coverage"] if r["coverage"] is not None else 0.0 for r in rows])
    ung = np.array([r["low_share"] if r["low_share"] is not None else 1.0 for r in rows])
    rec = np.array([r["gold_recall"] if r["gold_recall"] is not None else 0.0 for r in rows])
    recs = np.array([r.get("gold_recall_soft") if r.get("gold_recall_soft") is not None else 0.0 for r in rows])
    nop = np.array([r["n_opinions"] for r in rows], dtype=float)
    abst = np.array([bool(r["abstained"]) for r in rows])
    return dict(coverage=cov, ungrounded=ung, gold_recall=rec, gold_recall_soft=recs, n_opinions=nop, abstained=abst,
                any_ungrounded=(ung > 0) | abst)


def main():
    rows = [json.loads(l) for l in open(DIR / "generations.jsonl")]
    groups = defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r)
    base = groups["baseline"]
    sig = lambda rs: [(r["sample_id"], r["speaker"]) for r in rs]  # noqa: E731
    ref = sig(base)
    sids = [r["sample_id"] for r in base]
    fb = fields(base)
    conds = [c for c in groups if c != "baseline"]
    for c in conds:
        if sig(groups[c]) != ref:
            raise SystemExit(f"row order differs for {c}; cannot pair positionally")
    n = len(base)
    lines = [f"# Long-context speaker task: paired tests ({TAG})", "",
             f"{n} participant-hearing items over {len(set(sids))} held-out hearings, "
             f"median context {int(np.median([r['ctx_tokens'] for r in base]))} tokens. Baseline: coverage "
             f"{fb['coverage'].mean():.3f}, ungrounded share {fb['ungrounded'].mean():.3f}, gold recall "
             f"{fb['gold_recall'].mean():.3f}, opinions/item {fb['n_opinions'].mean():.2f}, abstain {fb['abstained'].mean():.3f}.",
             "Permutation null flips signs per hearing; Holm across each metric's column.", ""]
    metrics = ("coverage", "ungrounded", "gold_recall", "gold_recall_soft", "n_opinions")
    stats = {c: {} for c in conds}
    for m in metrics:
        ps = []
        for c in conds:
            d = fields(groups[c])[m] - fb[m]
            p = block_perm_p(d, sids, n_perm=N_PERM)
            stats[c][m] = (fields(groups[c])[m].mean(), d.mean(), p)
            ps.append(p)
        for c, a in zip(conds, holm(ps)):
            stats[c][m] = stats[c][m] + (a,)
    lines += ["| condition | " + " | ".join(f"{m} (Δ, Holm p)" for m in metrics) + " | items with any ungrounded opinion (McNemar) |",
              "|---|" + "---:|" * (len(metrics) + 1)]
    for c in conds:
        f = fields(groups[c])
        bb = int((fb["any_ungrounded"] & ~f["any_ungrounded"]).sum())
        cc = int((~fb["any_ungrounded"] & f["any_ungrounded"]).sum())
        mc = binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0
        cells = [f"{v:.3f} ({d:+.3f}, {a:.3g})" for (v, d, p, a) in (stats[c][m] for m in metrics)]
        lines.append(f"| {c} | " + " | ".join(cells) +
                     f" | {int(fb['any_ungrounded'].sum())}→{int(f['any_ungrounded'].sum())} ({bb}→{cc}, p={mc:.3g}) |")
    # specificity: steered minus mean of randoms at the same alpha
    lines += ["", "## Specificity: steered direction minus the mean of the random directions (same alpha)", "",
              "| steered | metric | Δ steered | mean Δ random (k) | difference | perm p |", "|---|---|---:|---:|---:|---:|"]
    for c in conds:
        if not c.startswith(("halluc", "misattr", "probe")):
            continue
        alpha = c.split(":")[1]
        rands = [r for r in conds if r.startswith("random") and r.endswith(":" + alpha)]
        if not rands:
            continue
        for m in metrics:
            ds = fields(groups[c])[m] - fb[m]
            dr = np.mean([fields(groups[r])[m] - fb[m] for r in rands], axis=0)
            lines.append(f"| {c} | {m} | {ds.mean():+.3f} | {dr.mean():+.3f} ({len(rands)}) | "
                         f"{(ds - dr).mean():+.3f} | {block_perm_p(ds - dr, sids, n_perm=N_PERM):.4g} |")
    text = "\n".join(lines) + "\n"
    (DIR / "paired_tests.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
