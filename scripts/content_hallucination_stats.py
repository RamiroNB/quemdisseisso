"""Paired significance tests for a content_hallucination_generation.py run.

Every condition generates for the same ordered item list, so rows pair positionally with baseline.
Several items can come from one hearing, so the permutation null flips signs in blocks of one hearing
(`sample_id`); p-values are Holm-corrected within each table. Abstentions score coverage 0.
  vs baseline   Δcoverage (block permutation) and the low-coverage flag, coverage < 0.3 (exact McNemar)
  specificity   halluc:α (or halluc_en:α) minus the mean of the random directions at the same α and mode,
                on the same items: separates the direction's effect from that of any perturbation of that size

Usage: uv run python scripts/content_hallucination_stats.py <model_tag[_gentag]>
Writes data/publichearingbr/content_hallucination_generation/<tag>/paired_tests.md.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen25_7b_instruct"
DIR = ROOT / "data" / "publichearingbr" / "content_hallucination_generation" / TAG
N_PERM = 20000
SEED = 0


def block_perm_p(delta, groups, n_perm=N_PERM):
    """Two-sided p for mean(delta) = 0, flipping the sign of each group's (hearing's) block at once."""
    rng = np.random.default_rng(SEED)
    delta = np.asarray(delta, dtype=float)
    obs = abs(delta.mean())
    uniq = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    gi = np.array([uniq[g] for g in groups])
    hits = 0
    for _ in range(n_perm):
        hits += abs((delta * rng.choice([-1.0, 1.0], size=len(uniq))[gi]).mean()) >= obs
    return (hits + 1) / (n_perm + 1)


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def cov_of(rows):
    return np.array([r["coverage"] if r["coverage"] is not None else 0.0 for r in rows])


def main():
    rows = [json.loads(l) for l in open(DIR / "generations.jsonl")]
    groups = defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r)
    key = lambda r: (r["sample_id"], r.get("opinion", ""), r["generation"] is not None)  # noqa: E731
    base = groups["baseline"]
    n = len(base)
    sids = [r["sample_id"] for r in base]
    b_cov = cov_of(base)
    b_low = b_cov < 0.3
    b_fab = np.array([bool(r["fabricated_number"]) for r in base])
    b_abst = np.array([bool(r["abstained"]) for r in base])

    lines = [f"# Content-hallucination generation: paired tests ({TAG})", "",
             f"{n} items over {len(set(sids))} held-out hearings. Baseline: coverage {b_cov.mean():.3f}, "
             f"low-coverage {b_low.mean():.3f}, fabricated-number {b_fab.mean():.3f}, abstain {b_abst.mean():.3f}.",
             "Permutation null flips signs per hearing; p-values Holm-corrected across this table.", "",
             "| condition | coverage | Δcov | perm p | Holm p | low-cov | McNemar p | fab-num | abstain |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    conds = [c for c in groups if c != "baseline" and len(groups[c]) == n]
    stats = {}
    for c in conds:
        g = groups[c]
        cov = cov_of(g)
        d = cov - b_cov
        low = cov < 0.3
        bb = int((b_low & ~low).sum()); cc = int((~b_low & low).sum())
        stats[c] = dict(cov=cov, d=d, p=block_perm_p(d, sids),
                        low=low.mean(), mc=binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0,
                        fab=np.mean([bool(r["fabricated_number"]) for r in g]),
                        abst=np.mean([bool(r["abstained"]) for r in g]))
    adj = holm([stats[c]["p"] for c in conds]) if conds else []
    for c, a in zip(conds, adj):
        s = stats[c]
        lines.append(f"| {c} | {s['cov'].mean():.3f} | {s['d'].mean():+.3f} | {s['p']:.4g} | {a:.4g} | "
                     f"{s['low']:.3f} | {s['mc']:.4g} | {s['fab']:.3f} | {s['abst']:.3f} |")

    # specificity: halluc:α vs mean of random*:α on the same items
    lines += ["", "## Specificity: steering direction vs random directions at the same strength", "",
              "| α | halluc Δcov | mean random Δcov (k dirs) | halluc − random | perm p | Holm p |",
              "|---|---:|---:|---:|---:|---:|"]
    spec_rows, spec_p = [], []
    for c in conds:
        m = re.fullmatch(r"halluc(_en)?(@\w+)?:([-\d.]+)", c)
        if not m:
            continue
        mode, alpha = (m.group(2) or ""), m.group(3)
        rands = [k for k in conds if re.fullmatch(rf"random\d*{re.escape(mode)}:{re.escape(alpha)}", k)]
        if not rands:
            continue
        rand_mean = np.mean([stats[k]["cov"] for k in rands], axis=0)
        d = stats[c]["cov"] - rand_mean
        p = block_perm_p(d, sids)
        spec_rows.append((c.split(":")[0] + ":" + alpha, stats[c]["d"].mean(), float((rand_mean - b_cov).mean()), len(rands), d.mean(), p))
        spec_p.append(p)
    for (alpha, hd, rd, k, dd, p), a in zip(spec_rows, holm(spec_p) if spec_p else []):
        lines.append(f"| {alpha} | {hd:+.3f} | {rd:+.3f} ({k}) | {dd:+.3f} | {p:.4g} | {a:.4g} |")
    if not spec_rows:
        lines.append("| — | no random condition at a matching α in this run | | | | |")

    text = "\n".join(lines) + "\n"
    (DIR / "paired_tests.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
