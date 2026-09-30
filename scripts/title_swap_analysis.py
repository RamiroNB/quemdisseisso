"""Analysis of the role axis of the title swap (scripts/title_swap.py).

Every condition is paired within item: the same opinion, evidence and credited person,
only the printed title differs. P-values are two-sided, from a sign-flip permutation null
with signs flipped in blocks of one hearing, so shared evidence and repeated speakers
cannot manufacture significance; CIs are hearing-level bootstrap.

Pre-specified tests, Holm-corrected across the four:
  H1  (primary) a borrowed parliamentarian title is believed more than a borrowed
      civil-society title, on the same false attribution (the credibility premium)
  H2  a borrowed parliamentarian title is believed more than another title of the
      claimant's own role, i.e. rank, not merely a changed string
  H3  having no title at all differs from having the claimant's real title
  H4  the correct-attribution arm moves when the true speaker's title is replaced
      (ceiling control)
Also reported: the rank ladder against the claimant's real title, a title-length
robustness check, and the parliamentarian-title effect by the true speaker's role.

Usage: uv run python scripts/title_swap_analysis.py <model_tag> [n_perm]
       (defaults llama_31_8b_instruct_tagged, 5000); writes
       data/publichearingbr/title_swap/<model_tag>/summary.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_TAG = sys.argv[1] if len(sys.argv) > 1 else "llama_31_8b_instruct_tagged"
N_PERM = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
DIR = ROOT / "data" / "publichearingbr" / "title_swap" / MODEL_TAG
ROLES = ("parliamentarian", "state", "academia", "private_sector", "civil_society")
KEY = ["sample_id", "opinion"]
SEED = 0


def perm_p(delta, groups, n_perm=N_PERM):
    """Two-sided p for mean(delta) = 0 under within-hearing sign flips."""
    rng = np.random.default_rng(SEED)
    delta = np.asarray(delta, dtype=float)
    obs = abs(delta.mean())
    uniq = {g: i for i, g in enumerate(pd.unique(groups))}
    gi = np.array([uniq[g] for g in groups])
    hits = 0
    for _ in range(n_perm):
        hits += abs((delta * rng.choice([-1.0, 1.0], size=len(uniq))[gi]).mean()) >= obs
    return (hits + 1) / (n_perm + 1)


def boot_ci(delta, groups, n_boot=2000):
    """Hearing-level bootstrap CI for the mean paired delta."""
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({"d": np.asarray(delta, dtype=float), "g": groups})
    by = [g.d.values for _, g in df.groupby("g")]
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(by), len(by))
        means.append(np.concatenate([by[i] for i in pick]).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def paired(df, a, b):
    """Rows of variant a joined to variant b on the same item."""
    x = df[df.variant == a][KEY + ["p_yes", "src_role", "y_role"]]
    y = df[df.variant == b][KEY + ["p_yes"]]
    m = x.merge(y, on=KEY, suffixes=("_a", "_b"))
    m["delta"] = m.p_yes_a - m.p_yes_b
    return m


def same_role_control(df):
    """The rank cell whose title comes from the claimant's own role, paired with
    swap_real: the string changed but the rank did not."""
    a = df[df.variant.str.startswith("swap_role:") & (df.role_shown == df.y_role)]
    b = df[df.variant == "swap_real"][KEY + ["p_yes"]]
    m = a[KEY + ["p_yes", "src_role", "y_role"]].merge(b, on=KEY, suffixes=("_a", "_b"))
    m["delta"] = m.p_yes_a - m.p_yes_b
    return m


def main():
    df = pd.read_json(DIR / "results.jsonl", lines=True)
    n_items = df[df.variant == "swap_real"].shape[0]
    out = [f"# Does the model price the job title? ({MODEL_TAG})", ""]
    out.append(
        f"{n_items} items. In every condition the words, the evidence, the speaker tags and the credited "
        f"person are identical -- only the printed title changes. The *false-attribution* arm credits the "
        f"words to another participant of the same hearing who does not hold the floor in the evidence, so "
        f"the correct answer is \"Não\"; the correct-attribution arm keeps the true speaker (answer \"Sim\") "
        f"and exists to show the ceiling. Mean Sim/Não mass {df.mass.mean():.3f}.")
    ref = df[df.variant == "swap_real"].p_yes.mean()
    orig = df[df.variant == "orig_real"].p_yes.mean()
    out += ["", f"Belief on a correct attribution: P(Sim) = {orig:.3f}. "
                f"Belief on a FALSE attribution with the claimant's real title: P(Sim) = {ref:.3f}.", ""]

    out += ["## The rank ladder (paired Δ vs the claimant's own real title)", "",
            "| title given to the false claimant | n | P(Sim) | Δ | 95% CI | endorses (P≥0.5) |",
            "|---|---:|---:|---:|---:|---:|"]
    ladder = {}
    for role in ROLES:
        m = paired(df, f"swap_role:{role}", "swap_real")
        if m.empty:
            continue
        lo, hi = boot_ci(m.delta.values, m.sample_id.values)
        ladder[role] = m
        out.append(f"| {role} | {len(m)} | {m.p_yes_a.mean():.3f} | {m.delta.mean():+.3f} | "
                   f"[{lo:+.3f}, {hi:+.3f}] | {float((m.p_yes_a >= 0.5).mean()):.2f} |")
    same = same_role_control(df)
    for label, m in (("another title of the claimant's own role (control)", same),
                     ("no title at all", paired(df, "swap_no_cargo", "swap_real"))):
        if m.empty:
            continue
        lo, hi = boot_ci(m.delta.values, m.sample_id.values)
        out.append(f"| _{label}_ | {len(m)} | {m.p_yes_a.mean():.3f} | {m.delta.mean():+.3f} | "
                   f"[{lo:+.3f}, {hi:+.3f}] | {float((m.p_yes_a >= 0.5).mean()):.2f} |")

    tests = {}
    # H1 parliamentarian vs civil_society, paired on the same item
    if "parliamentarian" in ladder and "civil_society" in ladder:
        a = ladder["parliamentarian"][KEY + ["p_yes_a"]].rename(columns={"p_yes_a": "parl"})
        b = ladder["civil_society"][KEY + ["p_yes_a"]].rename(columns={"p_yes_a": "civ"})
        m = a.merge(b, on=KEY)
        d = (m.parl - m.civ).values
        tests["H1 ->parliamentarian > ->civil_society"] = perm_p(d, m.sample_id.values)
        lo, hi = boot_ci(d, m.sample_id.values)
        out += ["", "## H1: the credibility premium", "",
                f"On the same {len(m)} false attributions, giving the claimant a **parliamentarian's** title "
                f"yields P(Sim) {m.parl.mean():.3f} against {m.civ.mean():.3f} for a **civil-society** title: "
                f"**{d.mean():+.3f}** (95% CI [{lo:+.3f}, {hi:+.3f}], perm p "
                f"{tests['H1 ->parliamentarian > ->civil_society']:.4g})."]

    # H2 parliamentarian vs a different title of the claimant's own role
    if "parliamentarian" in ladder:
        a = ladder["parliamentarian"][KEY + ["p_yes_a"]].rename(columns={"p_yes_a": "parl"})
        b = same_role_control(df)[KEY + ["p_yes_a"]].rename(columns={"p_yes_a": "same"})
        m = a.merge(b, on=KEY)
        if not m.empty:
            d = (m.parl - m.same).values
            tests["H2 ->parliamentarian > same-role title"] = perm_p(d, m.sample_id.values)
            out += ["", "## H2: rank, or just a different string?", "",
                    f"Both conditions replace the title with another real one; only one changes the rank. "
                    f"Parliamentarian {m.parl.mean():.3f} vs same-role {m.same.mean():.3f}: **{d.mean():+.3f}** "
                    f"(perm p {tests['H2 ->parliamentarian > same-role title']:.4g})."]

    m = paired(df, "swap_no_cargo", "swap_real")
    if not m.empty:
        tests["H3 no title != real title"] = perm_p(m.delta.values, m.sample_id.values)
    m = paired(df, "orig_cross_role", "orig_real")
    if not m.empty:
        tests["H4 correct arm moves"] = perm_p(m.delta.values, m.sample_id.values)
        out += ["", "## H4: the ceiling control", "",
                f"On a CORRECT attribution, replacing the true speaker's title moves belief "
                f"{m.delta.mean():+.3f} (from {m.p_yes_b.mean():.3f} to {m.p_yes_a.mean():.3f}, perm p "
                f"{tests['H4 correct arm moves']:.4g}). "
                + ("The correct arm is saturated (P(Sim) > 0.99), so this movement is at the ceiling and the "
                   "title effect is concentrated where the model is in doubt."
                   if m.p_yes_b.mean() > 0.99 else
                   "The correct arm is NOT saturated for this model, and the title also moves belief in a TRUE "
                   "attribution: the effect is not confined to the doubt regime here.")]

    # Robustness: parliamentary titles are short and formulaic ("Deputado (PT - SP)")
    # while civil-society ones are long and specific, so the ladder could be a
    # length artefact. academia vs private_sector compares roles whose titles have
    # similar median length.
    sw = df[df.variant.str.startswith("swap_role:")].copy()
    sw["len"] = sw.cargo.str.len()
    piv = sw.pivot_table(index=KEY, columns="role_shown", values="p_yes")
    lens = sw.pivot_table(index=KEY, columns="role_shown", values="len")
    gid = piv.reset_index().sample_id.values
    out += ["", "## Robustness: is the ladder just the title's length?", "",
            "| title | median length (chars) |", "|---|---:|"]
    for role in ROLES:
        if role in lens:
            out.append(f"| {role} | {lens[role].median():.0f} |")
    m = sw.merge(df[df.variant == "swap_real"][KEY + ["p_yes"]].rename(columns={"p_yes": "p0"}), on=KEY)
    out += ["", f"Correlation between Δ and title length: {(m.p_yes - m.p0).corr(m['len']):+.3f}.", "",
            "Length-matched contrasts (paired on the same item):", "",
            "| contrast | median lengths | Δ | perm p |", "|---|---|---:|---:|"]
    for a, b in (("academia", "private_sector"), ("academia", "civil_society")):
        if a in piv and b in piv:
            d = (piv[a] - piv[b]).values
            out.append(f"| {a} − {b} | {lens[a].median():.0f} vs {lens[b].median():.0f} | "
                       f"{d.mean():+.3f} | {perm_p(d, gid):.4g} |")
    if "parliamentarian" in piv and "civil_society" in piv:
        close = (lens.parliamentarian - lens.civil_society).abs() <= 15
        d = (piv.parliamentarian - piv.civil_society)[close].values
        if len(d) > 10:
            out.append(f"| H1 on length-matched items only | within 15 chars, n={int(close.sum())} | "
                       f"{d.mean():+.3f} | {perm_p(d, gid[close.values]):.4g} |")

    out += ["", "## Whose words are being reassigned? (Δ for a parliamentarian title, by the true speaker's role)", "",
            "| true speaker's role | n | Δ |", "|---|---:|---:|"]
    if "parliamentarian" in ladder:
        for role, g in ladder["parliamentarian"].groupby("src_role"):
            out.append(f"| {role} | {len(g)} | {g.delta.mean():+.3f} |")

    if tests:
        names = list(tests)
        adj = holm(np.array([tests[n] for n in names]))
        out += ["", "## Pre-specified tests (Holm-corrected)", "",
                "| hypothesis | perm p | Holm p |", "|---|---:|---:|"]
        for n, a in zip(names, adj):
            out.append(f"| {n} | {tests[n]:.4g} | {a:.4g} |")

    text = "\n".join(out) + "\n"
    (DIR / "summary.md").write_text(text)
    print(text)
    print(f"wrote {DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
