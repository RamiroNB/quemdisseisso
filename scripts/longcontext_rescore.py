"""Re-score stored long-context generations without regenerating them.

For each data/publichearingbr/longcontext_generation/<run_dir_tag>/, recomputes the
derived fields of every row (parsed opinions, coverage, ungrounded share, abstention,
fabricated numbers, gold recall) from the stored raw text with
longcontext_speaker_generation.score_generation, and rewrites generations.jsonl and
summary.csv in place. Run longcontext_stats.py afterwards.

Usage: uv run python scripts/longcontext_rescore.py <run_dir_tag> [<run_dir_tag> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from longcontext_speaker_generation import ROOT, score_generation  # noqa: E402

BASE = ROOT / "data" / "publichearingbr" / "longcontext_generation"
DERIVED = ("opinions", "n_opinions", "abstained", "coverage", "low_share", "fabricated_number",
           "gold_recall", "gold_recall_soft")


def main():
    for tag in sys.argv[1:]:
        d = BASE / tag
        rows = [json.loads(l) for l in open(d / "generations.jsonl")]
        for r in rows:
            for k in DERIVED:
                r.pop(k, None)
            r.update(score_generation(r["generation"], r["chunks"], r["gold"]))
        with open(d / "generations.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = []
        for cond in dict.fromkeys(r["condition"] for r in rows):
            rs = [r for r in rows if r["condition"] == cond]
            cov = [r["coverage"] if r["coverage"] is not None else 0.0 for r in rs]
            low = [r["low_share"] for r in rs if r["low_share"] is not None]
            rec = [r["gold_recall"] for r in rs if r["gold_recall"] is not None]
            recs = [r["gold_recall_soft"] for r in rs if r["gold_recall_soft"] is not None]
            summary.append(dict(condition=cond, n=len(rs), mean_coverage=float(np.mean(cov)),
                                ungrounded_share=float(np.mean(low)) if low else float("nan"),
                                gold_recall=float(np.mean(rec)) if rec else float("nan"),
                                gold_recall_soft=float(np.mean(recs)) if recs else float("nan"),
                                n_opinions=float(np.mean([r["n_opinions"] for r in rs])),
                                abstain_rate=float(np.mean([r["abstained"] for r in rs])),
                                fabricated_number_rate=float(np.mean([r["fabricated_number"] for r in rs]))))
            s = summary[-1]
            print(f"{tag} {cond:14s} coverage={s['mean_coverage']:.3f} ungrounded={s['ungrounded_share']:.3f} "
                  f"recall={s['gold_recall']:.3f}/{s['gold_recall_soft']:.3f} n_op={s['n_opinions']:.2f} "
                  f"abstain={s['abstain_rate']:.2f} fab={s['fabricated_number_rate']:.2f}")
        pd.DataFrame(summary).to_csv(d / "summary.csv", index=False)


if __name__ == "__main__":
    main()
