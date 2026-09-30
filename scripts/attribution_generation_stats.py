"""Paired significance tests for the generation-level attribution test.

Reads data/publichearingbr/attribution_generation/<tag>/generations.jsonl (from
scripts/attribution_generation.py) and compares every condition with the `baseline`
condition within each prompt variant, by exact McNemar (binomial on the discordant pairs).
Every condition generates for the same ordered pair list, so rows are paired by position
(the same opinion and the same swap target); one opinion can appear with several swap
targets, which is why position, not (sample_id, kind), is the pairing key.

Outcomes:
  misattribution  on swap rows whose target never held the floor: the model
                  produced a statement instead of abstaining.
  grounded_own    on original rows: not abstained and >=50% of the generation
                  is contained in the true speaker's own chunks.
  over_refusal    on original rows: abstained although the speaker did speak.

Usage: uv run python scripts/attribution_generation_stats.py <model_tag>
Writes paired_tests.md next to the generations and prints it.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]


def load(tag):
    path = ROOT / "data/publichearingbr/attribution_generation" / tag / "generations.jsonl"
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def outcomes(row):
    """Yield (outcome_name, 1/0) for whichever outcomes this row is eligible for."""
    if row["kind"] != "original":
        if not row["target_has_turn"]:
            yield "misattribution", int(not row["abstained"])
    else:
        yield "over_refusal", int(row["abstained"])
        grounded = (not row["abstained"]) and (row.get("contain_own") or 0.0) >= 0.5
        yield "grounded_own", int(grounded)


def mcnemar(pairs):
    """pairs: list of (baseline, condition) 0/1. Returns n, rates, b, c, p."""
    b = sum(1 for x, y in pairs if x == 1 and y == 0)  # baseline only
    c = sum(1 for x, y in pairs if x == 0 and y == 1)  # condition only
    n = len(pairs)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return n, sum(x for x, _ in pairs) / n, sum(y for _, y in pairs) / n, b, c, p


def main():
    tag = sys.argv[1]
    rows = load(tag)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["prompt"], row["condition"])].append(row)

    # Every condition must have generated for the identical ordered pair list,
    # otherwise positional pairing would compare different opinions.
    signature = lambda rs: [(r["kind"], r["sample_id"], r["target"]) for r in rs]
    reference = signature(next(iter(groups.values())))
    for key, rs in groups.items():
        if signature(rs) != reference:
            raise SystemExit(f"row order differs for {key}; cannot pair positionally")

    # (prompt, condition, outcome) -> {position: value}
    table = defaultdict(dict)
    for (prompt, condition), rs in groups.items():
        for position, row in enumerate(rs):
            for name, value in outcomes(row):
                table[(prompt, condition, name)][position] = value

    prompts = sorted({r["prompt"] for r in rows})
    conditions = [c for c in dict.fromkeys(r["condition"] for r in rows) if c != "baseline"]

    lines = [f"# Generation-level attribution: paired tests ({tag})", ""]
    lines.append("Exact McNemar vs the unsteered baseline, paired on (opinion, swap). "
                 "`b` = baseline-only events, `c` = condition-only events.")
    for prompt in prompts:
        lines += ["", f"## prompt = {prompt}", "",
                  "| outcome | condition | n pairs | baseline | steered | b | c | p |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for name in ("misattribution", "grounded_own", "over_refusal"):
            base = table.get((prompt, "baseline", name), {})
            for cond in conditions:
                got = table.get((prompt, cond, name), {})
                keys = sorted(set(base) & set(got))
                if not keys:
                    continue
                pairs = [(base[k], got[k]) for k in keys]
                n, r0, r1, b, c, p = mcnemar(pairs)
                lines.append(f"| {name} | {cond} | {n} | {r0:.3f} | {r1:.3f} | {b} | {c} | {p:.4g} |")

    out = ROOT / "data/publichearingbr/attribution_generation" / tag / "paired_tests.md"
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
