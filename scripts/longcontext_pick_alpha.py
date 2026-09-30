"""Pick the steering alpha for the long-context speaker task on the dev half, by a
rule fixed before the test half is generated.

Reads data/publichearingbr/longcontext_generation/<run_dir_tag>/generations.jsonl
(conditions baseline, halluc:<alpha> and random<k>:<alpha>). A candidate alpha passes if:
  1. its coverage gain over baseline is Holm-significant across the tested alphas
     (sign-flip permutation by hearing);
  2. it beats the mean of the random directions at the nearest tested alpha
     on coverage (perm p < 0.05);
  3. the soft gold-recall change is not a significant loss
     (perm p >= 0.05 or delta >= 0);
  4. the output stays a summary: first-person share <= baseline + 0.10 and
     odd-character share <= baseline + 0.10 (a third-person check does not
     apply to list output, so these two artefact markers are used instead).
The largest passing alpha is chosen; rules 3-4 are what stop it before outputs
degrade. If nothing passes, it falls back to the smallest tested alpha with a
warning. Prints CHOSEN_ALPHA=<value> for a calling shell script.

Usage: uv run python scripts/longcontext_pick_alpha.py <run_dir_tag>
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content_hallucination_register import FIRST, ODD  # noqa: E402
from content_hallucination_stats import block_perm_p, holm  # noqa: E402
from longcontext_stats import fields  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAG = sys.argv[1]
DIR = ROOT / "data" / "publichearingbr" / "longcontext_generation" / TAG
N_PERM = 5000


def odd_share(rows):
    return float(np.mean([len(ODD.findall(r["generation"])) / max(1, len(r["generation"])) > 0.01 for r in rows]))


def first_share(rows):
    return float(np.mean([bool(FIRST.search(r["generation"])) for r in rows]))


def main():
    rows = [json.loads(l) for l in open(DIR / "generations.jsonl")]
    groups = defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r)
    base = groups["baseline"]
    sids = [r["sample_id"] for r in base]
    fb = fields(base)
    alphas = sorted({float(c.split(":")[1]) for c in groups if c.startswith("halluc:")})
    rand_alphas = sorted({float(c.split(":")[1]) for c in groups if c.startswith("random")})
    cov_p = {a: block_perm_p(fields(groups[f"halluc:{a}"])["coverage"] - fb["coverage"], sids, n_perm=N_PERM) for a in alphas}
    adj = dict(zip(alphas, holm([cov_p[a] for a in alphas])))
    b_first, b_odd = first_share(base), odd_share(base)
    passing = []
    print(f"baseline: coverage {fb['coverage'].mean():.3f}, first-person {b_first:.2f}, odd {b_odd:.2f}")
    for a in alphas:
        g = groups[f"halluc:{a}"]
        f = fields(g)
        d_cov = f["coverage"] - fb["coverage"]
        ra = min(rand_alphas, key=lambda x: abs(x - a))
        rands = [c for c in groups if c.startswith("random") and c.endswith(f":{ra}")]
        d_rand = np.mean([fields(groups[c])["coverage"] - fb["coverage"] for c in rands], axis=0)
        p_spec = block_perm_p(d_cov - d_rand, sids, n_perm=N_PERM)
        d_rec = f["gold_recall_soft"] - fb["gold_recall_soft"]
        p_rec = block_perm_p(d_rec, sids, n_perm=N_PERM)
        fp, od = first_share(g), odd_share(g)
        ok = (adj[a] < 0.05 and d_cov.mean() > 0 and p_spec < 0.05 and d_cov.mean() > d_rand.mean()
              and (d_rec.mean() >= 0 or p_rec >= 0.05) and fp <= b_first + 0.10 and od <= b_odd + 0.10)
        print(f"alpha {a}: dcov {d_cov.mean():+.3f} (Holm {adj[a]:.3g}) | vs randoms@{ra} {d_cov.mean()-d_rand.mean():+.3f} (p {p_spec:.3g}) | "
              f"drecall_soft {d_rec.mean():+.3f} (p {p_rec:.3g}) | first-person {fp:.2f} | odd {od:.2f} | {'PASS' if ok else 'fail'}")
        if ok:
            passing.append(a)
    if passing:
        chosen = max(passing)
        print(f"rule: largest passing alpha -> CHOSEN_ALPHA={chosen}")
    else:
        chosen = min(alphas)
        print(f"!! NO ALPHA PASSED THE RULE; falling back to the smallest tested -> CHOSEN_ALPHA={chosen}")


if __name__ == "__main__":
    main()
