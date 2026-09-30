"""Statistics for the identity-swap counterfactuals (scripts/identity_swap.py).

Two questions, answered for the probe (internal) and for P(Sim) (behavioural):
  1. Does the verdict on the same supported opinion and evidence change when only the
     speaker's identity changes? Paired deltas by swap type and by source/target group,
     with a sign-flip permutation test that flips whole hearings (respects clustering),
     and flip rates at the operating point (probe threshold, P(Sim) = 0.5).
  2. Misattribution detection: can the signal tell a real attribution from the same words
     credited to another participant of the hearing? ROC-AUC original vs within-hearing
     swap, overall and split by target_in_chunks (the target's surname occurs in the
     evidence); plus detection by the true speaker's role and gender.

Usage: uv run python scripts/identity_swap_analysis.py [model_tag] [probe_threshold]
       (defaults llama_2_7b_chat_hf, 0.5). Reads and writes
       data/publichearingbr/identity_swap/<model_tag>/{results.jsonl, summary.md}.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_TAG = sys.argv[1] if len(sys.argv) > 1 else "llama_2_7b_chat_hf"
IN_PATH = ROOT / "data" / "publichearingbr" / "identity_swap" / MODEL_TAG / "results.jsonl"
OUT_PATH = ROOT / "data" / "publichearingbr" / "identity_swap" / MODEL_TAG / "summary.md"
PROBE_THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
N_PERM = 5000
SEED = 0


def paired_perm(delta, hearing, n_perm=N_PERM, seed=SEED):
    """Two-sided p for mean(delta)=0, flipping signs per hearing cluster."""
    rng = np.random.default_rng(seed)
    delta = np.asarray(delta); hearing = np.asarray(hearing)
    clusters = defaultdict(list)
    for i, h in enumerate(hearing):
        clusters[h].append(i)
    idx = [np.array(v) for v in clusters.values()]
    obs = abs(delta.mean())
    count = 0
    for _ in range(n_perm):
        signs = np.ones(len(delta))
        for ix in idx:
            if rng.random() < 0.5:
                signs[ix] = -1
        if abs((delta * signs).mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def main():
    df = pd.read_json(IN_PATH, lines=True)
    key = ["sample_id", "nome", "opinion"]
    base = df[df.variant == "original"].set_index(key)
    md = [f"# Identity-swap results ({MODEL_TAG}; n originals = {len(base)})\n"]
    md.append(f"Probe threshold for flip rates: {PROBE_THR}; behavioural threshold: P(Sim)=0.5. "
              f"Baseline on originals: mean probe {base.probe.mean():.3f}, mean P(Sim) {base.p_yes.mean():.3f}, "
              f"share P(Sim)>=0.5 {(base.p_yes>=0.5).mean():.2f}.\n")

    md.append("\n## 1. Does the verdict move when only the speaker changes?\n")
    md.append("| swap | direction | n | Δprobe | perm p | probe flips→hallucinated | ΔP(Sim) | perm p | verdict flips→Não |\n|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for v in ("within_hearing", "cross_role", "cross_gender"):
        sub = df[df.variant == v].set_index(key).join(base[["probe", "p_yes"]], rsuffix="_orig")
        if sub.empty:
            continue
        sub = sub.reset_index()
        groups = [("all", sub)]
        if v == "cross_gender":
            groups += [(f"{g}→{'M' if g=='F' else 'F'}", sub[sub.src_gender == g]) for g in ("F", "M")]
        if v == "cross_role":
            groups += [(f"{r}→other role", sub[sub.src_role == r]) for r in sorted(sub.src_role.dropna().unique())]
            groups += [(f"→{r}", sub[sub.target_role == r]) for r in sorted(sub.target_role.dropna().unique())]
        if v == "within_hearing":
            groups += [("target's words in chunks", sub[sub.target_in_chunks == True]), ("target absent from chunks", sub[sub.target_in_chunks == False])]
        for name, g in groups:
            if len(g) < 20:
                continue
            dp = (g.probe - g.probe_orig).values; dy = (g.p_yes - g.p_yes_orig).values
            p1 = paired_perm(dp, g.sample_id.values); p2 = paired_perm(dy, g.sample_id.values)
            flip_probe = ((g.probe_orig < PROBE_THR) & (g.probe >= PROBE_THR)).mean()
            flip_yes = ((g.p_yes_orig >= 0.5) & (g.p_yes < 0.5)).mean()
            md.append(f"| {v} | {name} | {len(g)} | {dp.mean():+.3f} | {p1:.3g} | {flip_probe:.2f} | {dy.mean():+.3f} | {p2:.3g} | {flip_yes:.2f} |")

    md.append("\n## 2. Misattribution detection: original vs the same words credited to another participant\n")
    w = df[df.variant == "within_hearing"].set_index(key).join(base[["probe", "p_yes"]], rsuffix="_orig").reset_index()
    md.append("| subset | n pairs | probe ROC-AUC | P(Sim) ROC-AUC (as 1-P) | probe pairwise win | P(Sim) pairwise win |\n|---|---:|---:|---:|---:|---:|")
    for name, g in (("all", w), ("target's words in chunks", w[w.target_in_chunks == True]), ("target absent from chunks", w[w.target_in_chunks == False])):
        if len(g) < 20:
            continue
        y = np.r_[np.zeros(len(g)), np.ones(len(g))]  # 1 = misattributed (swap)
        probe_auc = roc_auc_score(y, np.r_[g.probe_orig.values, g.probe.values])
        yes_auc = roc_auc_score(y, np.r_[1 - g.p_yes_orig.values, 1 - g.p_yes.values])
        md.append(f"| {name} | {len(g)} | {probe_auc:.3f} | {yes_auc:.3f} | {(g.probe > g.probe_orig).mean():.2f} | {(g.p_yes < g.p_yes_orig).mean():.2f} |")

    md.append("\n## 3. Fairness of misattribution detection: is the swap caught equally for every source group?\n")
    md.append("| grouping | group | n | probe pairwise win | ΔP(Sim) |\n|---|---|---:|---:|---:|")
    for col in ("src_role", "src_gender"):
        for g_name, g in w.groupby(col):
            if len(g) < 20:
                continue
            md.append(f"| {col} | {g_name} | {len(g)} | {(g.probe > g.probe_orig).mean():.2f} | {(g.p_yes - g.p_yes_orig).mean():+.3f} |")

    report = "\n".join(md)
    OUT_PATH.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
