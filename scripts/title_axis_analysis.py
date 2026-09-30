"""Analysis of the minimal-pair identity axes of the title swap (scripts/title_swap.py).

Handles runs with OCARANDU_TITLE_AXIS=party or gender, where every condition differs from
the reference (the claimant's real title, on the false attribution) by a single token (a
party acronym) or a single morpheme (the grammatical gender of the office word). The words,
the evidence, the speaker tags and the credited person are identical across conditions, so
a movement in P(Sim) is the model pricing that one identity cue.

Every test is paired within item, with a sign-flip permutation null flipped per hearing so
shared evidence and repeated speakers cannot manufacture significance; CIs are
hearing-level bootstrap; Holm correction is applied across the levels of the axis. Also
reported: the widest paired contrast between two levels; for gender, the masculine-minus-
feminine effect by the claimant's derived gender; for party, an exploratory bloc contrast.

Usage: uv run python scripts/title_axis_analysis.py <model_tag>_<axis> [n_perm]
       (e.g. qwen25_7b_instruct_tagged_party, the default; n_perm 5000). Writes
       data/publichearingbr/title_swap/<model_tag>_<axis>/summary.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen25_7b_instruct_tagged_party"
N_PERM = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
DIR = ROOT / "data" / "publichearingbr" / "title_swap" / TAG
KEY = ["sample_id", "opinion"]
SEED = 0


def perm_p(delta, groups, n_perm=N_PERM):
    rng = np.random.default_rng(SEED)
    delta = np.asarray(delta, dtype=float)
    delta = delta[~np.isnan(delta)]
    if not len(delta):
        return 1.0
    obs = abs(delta.mean())
    uniq = {g: i for i, g in enumerate(pd.unique(groups))}
    gi = np.array([uniq[g] for g in groups])
    hits = 0
    for _ in range(n_perm):
        hits += abs((delta * rng.choice([-1.0, 1.0], size=len(uniq))[gi]).mean()) >= obs
    return (hits + 1) / (n_perm + 1)


def boot_ci(delta, groups, n_boot=2000):
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({"d": np.asarray(delta, dtype=float), "g": groups}).dropna()
    by = [g.d.values for _, g in df.groupby("g")]
    if not by:
        return float("nan"), float("nan")
    means = [np.concatenate([by[i] for i in rng.integers(0, len(by), len(by))]).mean()
             for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    df = pd.read_json(DIR / "results.jsonl", lines=True)
    axis = df.axis.iloc[0]
    prefix = f"swap_{axis}:"
    levels = [v.split(":", 1)[1] for v in df.variant.unique() if v.startswith(prefix)]
    ref = df[df.variant == "swap_real"][KEY + ["p_yes", "y_gender", "y_role", "src_role"]].rename(
        columns={"p_yes": "p_ref"})
    n_items = len(ref)

    out = [f"# Minimal-pair identity axis: {axis} ({TAG})", "",
           f"{n_items} items. The words, the evidence, the speaker tags and the credited person are "
           f"identical in every condition; only the {'party acronym' if axis == 'party' else 'grammatical gender of the office word'} "
           f"differs. The words belong to another participant, so the correct answer is \"Não\" throughout "
           f"and any movement is the model pricing that one cue. Mean Sim/Não mass {df.mass.mean():.3f}.",
           "",
           f"Belief on the correct attribution: P(Sim) = {df[df.variant == 'orig_real'].p_yes.mean():.3f}. "
           f"On the FALSE attribution with the claimant's real title: {ref.p_ref.mean():.3f}.", ""]

    wide = df[df.variant.str.startswith(prefix)].copy()
    wide["level"] = wide.variant.str.split(":", n=1).str[1]
    piv = wide.pivot_table(index=KEY, columns="level", values="p_yes").reset_index()
    piv = piv.merge(ref, on=KEY)
    gid = piv.sample_id.values

    out += [f"## Effect of each {axis} level (paired Δ vs the claimant's real title)", "",
            "| level | n | P(Sim) | Δ | 95% CI | perm p |", "|---|---:|---:|---:|---:|---:|"]
    rows, pvals = [], []
    for lv in sorted(levels):
        if lv not in piv:
            continue
        d = (piv[lv] - piv.p_ref).values
        p = perm_p(d, gid)
        lo, hi = boot_ci(d, gid)
        rows.append((lv, int(np.sum(~np.isnan(d))), float(np.nanmean(piv[lv])), float(np.nanmean(d)), lo, hi, p))
        pvals.append(p)
    adj = holm(np.array(pvals)) if pvals else []
    order = np.argsort([-r[3] for r in rows])
    for i in order:
        lv, n, py, d, lo, hi, p = rows[i]
        out.append(f"| {lv} | {n} | {py:.3f} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.4g} (Holm {adj[i]:.4g}) |")

    # the widest contrast on the axis, paired
    if len(rows) >= 2:
        hi_lv = rows[int(np.argmax([r[3] for r in rows]))][0]
        lo_lv = rows[int(np.argmin([r[3] for r in rows]))][0]
        d = (piv[hi_lv] - piv[lo_lv]).values
        p = perm_p(d, gid)
        ci = boot_ci(d, gid)
        out += ["", f"## Widest contrast: {hi_lv} vs {lo_lv}", "",
                f"Paired on the same items, a claimant labelled **{hi_lv}** is endorsed "
                f"{float(np.nanmean(d)):+.3f} more than the same claimant labelled **{lo_lv}** "
                f"(95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], perm p {p:.4g})."]

    if axis == "gender" and {"F", "M"} <= set(piv.columns):
        out += ["", "## Congruence: does it matter whether the person is a woman?", "",
                "| claimant's derived gender | n | Δ(masculine title − feminine title) | perm p |",
                "|---|---:|---:|---:|"]
        for g, sub in piv.groupby("y_gender"):
            d = (sub.M - sub.F).values
            out.append(f"| {g} | {int(np.sum(~np.isnan(d)))} | {float(np.nanmean(d)):+.3f} | "
                       f"{perm_p(d, sub.sample_id.values):.4g} |")

    if axis == "party":
        blocs = {"left (PSOL, PT)": ["PSOL", "PT"], "centre (MDB, PSDB)": ["MDB", "PSDB"],
                 "right (PL, NOVO)": ["PL", "NOVO"]}
        avail = {k: [c for c in v if c in piv.columns] for k, v in blocs.items()}
        avail = {k: v for k, v in avail.items() if v}
        if len(avail) >= 2:
            out += ["", "## By bloc (exploratory, not pre-specified)", "",
                    "| bloc | Δ vs real title |", "|---|---:|"]
            for k, cols in avail.items():
                out.append(f"| {k} | {float(np.nanmean(piv[cols].mean(axis=1) - piv.p_ref)):+.3f} |")
            if "left (PSOL, PT)" in avail and "right (PL, NOVO)" in avail:
                d = (piv[avail["right (PL, NOVO)"]].mean(axis=1)
                     - piv[avail["left (PSOL, PT)"]].mean(axis=1)).values
                out += ["", f"Right-bloc minus left-bloc: {float(np.nanmean(d)):+.3f} "
                            f"(perm p {perm_p(d, gid):.4g})."]

    text = "\n".join(out) + "\n"
    (DIR / "summary.md").write_text(text)
    print(text)
    print(f"wrote {DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
