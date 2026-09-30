"""Decision level of the title swap: does the printed title flip the verdict, or only move P(Sim)?

title_swap_analysis.py reports paired mean differences in P(Sim). This script pairs the same
items and, for each pre-specified contrast (H1-H4) and each rung of the rank ladder, counts
the verdicts that flip in each direction, with an exact McNemar test (two-sided binomial on
the discordant pairs). "Endorses" = renormalised P(Sim) >= 0.5, the rule of the `endorses`
column of summary.md.

It also reports the Sim/Não answer mass. P(Sim) is renormalised over the two answer tokens;
when a model puts most of its next-token mass elsewhere (a formatting token, a different
phrasing), the renormalised number is a weak instrument and must be read with the mass.

Usage: uv run python scripts/title_swap_decisions.py <model_tag> [<model_tag> ...]
       (default llama_31_8b_instruct_tagged). Writes
       data/publichearingbr/title_swap/<model_tag>/decisions.md beside summary.md.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "publichearingbr" / "title_swap"
ROLES = ("parliamentarian", "state", "academia", "private_sector", "civil_society")
KEY = ["sample_id", "opinion"]
THR = 0.5


def pair(df, a_mask, b_mask, label):
    a = df[a_mask][KEY + ["p_yes"]].rename(columns={"p_yes": "pa"})
    b = df[b_mask][KEY + ["p_yes"]].rename(columns={"p_yes": "pb"})
    if a.duplicated(KEY).any() or b.duplicated(KEY).any():
        raise SystemExit(f"{label}: items are not unique on {KEY}; cannot pair")
    m = a.merge(b, on=KEY)
    if m.empty:
        raise SystemExit(f"{label}: no pairs")
    return m


def decision_row(label, m):
    ea, eb = m.pa.values >= THR, m.pb.values >= THR
    only_a = int((ea & ~eb).sum())  # A endorses, B does not
    only_b = int((~ea & eb).sum())  # B endorses, A does not
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return (f"| {label} | {len(m)} | {ea.mean():.3f} | {eb.mean():.3f} | {only_a} | {only_b} | "
            f"{(only_a - only_b) / len(m):+.3f} | {p:.4g} |"), only_a, only_b, p


def run(tag):
    d = BASE / tag
    df = pd.read_json(d / "results.jsonl", lines=True)
    v = df.variant
    same_role = v.str.startswith("swap_role:") & (df.role_shown == df.y_role)
    contrasts = [
        ("H1 parliamentarian title vs civil-society title (false attribution)",
         v == "swap_role:parliamentarian", v == "swap_role:civil_society"),
        ("H2 parliamentarian title vs another title of the claimant's own role",
         v == "swap_role:parliamentarian", same_role),
        ("H3 no title vs the claimant's real title", v == "swap_no_cargo", v == "swap_real"),
        ("H4 correct arm: cross-role title vs the true speaker's real title",
         v == "orig_cross_role", v == "orig_real"),
    ] + [(f"ladder: {r} title vs the claimant's real title", v == f"swap_role:{r}", v == "swap_real")
         for r in ROLES]

    mass = df.mass
    endorse_false = float((df[v == "swap_real"].p_yes >= THR).mean())
    endorse_true = float((df[v == "orig_real"].p_yes >= THR).mean())
    out = [f"# Decision level: does the title flip the verdict? ({tag})", "",
           f"Verdict = renormalised P(Sim) >= {THR}. Sim/Não answer mass: mean {mass.mean():.3f}, "
           f"median {mass.median():.3f}, share of rows with mass < 0.5: {(mass < 0.5).mean():.3f}"
           + (" -- **the two answer tokens carry a minority of the next-token mass for most items; "
              "P(Sim) is a weak instrument for this model.**" if mass.median() < 0.5 else "."),
           "",
           f"Endorses the false attribution (real title): {endorse_false:.3f}; "
           f"endorses the correct attribution: {endorse_true:.3f}.", "",
           "Paired on the same item; `only A` = A endorses and B does not, `only B` the reverse; "
           "net = (only A − only B) / n; exact McNemar = two-sided binomial on the discordant pairs.", "",
           "| contrast (A vs B) | n | endorses A | endorses B | only A | only B | net flips | McNemar p |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    h1 = None
    for label, am, bm in contrasts:
        row, a, b, p = decision_row(label, pair(df, am, bm, label))
        out.append(row)
        if label.startswith("H1"):
            h1 = (a, b, p)
    a, b, p = h1
    verdict = ("replicates at the decision level" if p < 0.05 and a > b else
               "does NOT replicate at the decision level" if p >= 0.05 else
               "moves the WRONG way at the decision level")
    out += ["", f"**H1 at the decision level: {a} items flip toward endorsing under the parliamentary title, "
                f"{b} the other way (exact McNemar p = {p:.3g}) -- {verdict}.**"]
    text = "\n".join(out) + "\n"
    (d / "decisions.md").write_text(text)
    print(text)
    print(f"wrote {d / 'decisions.md'}\n")


if __name__ == "__main__":
    for t in sys.argv[1:] or ["llama_31_8b_instruct_tagged"]:
        run(t)
