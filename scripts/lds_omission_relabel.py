"""Re-derive the omission audit's labels, measure how far the original labels were off, and re-run
the omission tests (speaking time, gender, parliamentarian, session chair, finer roles). CPU only.

"Published" labels in the report are the original ones in stage2/omissions.jsonl, written by
scripts/lds_audit_stage2.py: named_in_article = the last token of the header name as a substring of
the normalised article (misses long civil names; hits homonyms, institutions and the reporter's
byline), and role/gender = speaker_attributes.jsonl joined by surname only, the last record winning
(so a speaker can inherit another person's attributes). The population is kept identical (asserted
row by row); the new labels are:
  named     word-bounded matching of the header name against the article, with the reporter's
            closing byline removed: the first name followed within three words by another of the
            person's name tokens, a single-token name ("Sanderson"), an office word immediately
            before the surname ("os deputados Bacelar"), or a strict name match to one of the
            hearing's `metadados` participants; plus the hand-checked MISSES. `named_how` records
            the rule; None (unresolvable) when the header parses to a bare title.
  chair     the header resolves a chair role (PRESIDENTE / VICE-PRESIDENTE / RELATOR) to this person.
  gender    the transcript honorific, "O SR." / "A SRA." (None if a speaker's headers carry both).
  parl      the header's party-UF parenthetical ("(Bloco/PT - SP)"), present for parliamentarians.
  role      the derived role only where exactly one attribute record is the same person (name match)
            in the same hearing (`sample_ids`). This population is selected: attribute records exist
            for the NLI half's speakers, who are disproportionately the ones the article names.

Usage: uv run python scripts/lds_omission_relabel.py [model_tag] [n_perm]
       (defaults: llama_31_8b_instruct, 2000 within-hearing permutations)
Output: data/publichearingbr/lds_audit/<model_tag>/stage2/omissions_v2.jsonl and
        omission_candidates_checked.csv (the hand-check list); the report is printed and written to
        stage2/lds_omission_relabel.md.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from difflib import SequenceMatcher  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tagged_evidence import HEADER, name_match, parse_header  # noqa: E402
from lds_audit_stage2 import ATTRS, LDS, MIN_FLOOR, norm, surname  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAG = sys.argv[1] if len(sys.argv) > 1 else "llama_31_8b_instruct"
N_PERM = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
STAGE2 = ROOT / "data" / "publichearingbr" / "lds_audit" / TAG / "stage2"
OUT_ROWS = STAGE2 / "omissions_v2.jsonl"
OUT_CAND = STAGE2 / "omission_candidates_checked.csv"
OUT_DOC = STAGE2 / "lds_omission_relabel.md"
SEED = 0

PART = {"de", "da", "do", "das", "dos", "e", "van", "von", "del", "della", "di", "du"}
LEAD_TITLE = {"ministro", "ministra", "dr", "dra", "prof", "professor", "professora", "deputado", "deputada",
              "senador", "senadora", "sr", "sra", "presidente", "vereador", "vereadora"}
UNRESOLVABLE = {("dr",), ("prof",), ("interprete",)}   # header parsed to a bare title: several people merged
OFFICE = (r"(?:deputad[oa]s?|senador(?:a|es)?|ministr[oa]|secretari[oa]|president[ea]|relator[a]?|"
          r"vereador[a]?|governador[a]?|prefeit[oa])")
PARTICLES = r"(?:(?:de|da|do|das|dos|e|van|von|del|di)\s+)*"
PARTY_UF = re.compile(r"\b[A-Z][A-Za-zÀ-ú/ .]{1,40}\s*-\s*[A-Z]{2}\b")

# Floor-holders the rules call omitted although the article names them, found by reading every
# omitted row that has a >=4-letter token of its name in its article (all candidates, with the
# context of each hit, go to omission_candidates_checked.csv); a candidate not listed here is someone
# else (a homonym, "Sao Paulo" for a Paulo, the reporter). Third field: the evidence (the article's
# wording and, where the name differs, the speaker's own turn or header).
MISSES = [
    (12, "Patrícia Rodrigues Da Silva", "article 'Pagu Rodrigues'; her turn: 'Eu me chamo Pagu Rodrigues'"),
    (55, "Maria Walneide Ribeiro De Oliveira Romano", "article 'Walneide Romano'"),
    (83, "Marco Antonio Da Silva Souza", "article 'Markinhus Souza'; his header: 'O SR. MARCO ANTONIO DA SILVA SOUZA (MARKINHUS SOUZA)'"),
    (90, "Elisabete Laurindo De Souza", "article 'Elizabeth de Souza, representante do Conselho Federal de Educacao Fisica'; her turn speaks for that council"),
    (82, "Alcides De Lima Tserewaptu", "article 'mestre Alcides, presidente do forum para as culturas populares'"),
    (91, "Márcio Caires", "article 'Mario Caires, presidente da Neoenergia Cosern'"),
    (161, "Ricardo Catanant", "article 'Eduardo Catanant, diretor da Anac'; his turn is ANAC's"),
    (165, "Maria Raquel Mesquita Melo", "article 'Raquel Melo, do Ministerio do Planejamento'"),
    (170, "Rosijane Fernandes Moura", "article 'Rosijane Tukano'; her turn: 'Eu sou Rosijane. Sou do povo tukano'"),
    (176, "Renata Flores Tibyriçá", "article 'Renata Tibirica', defensora publica"),
    (182, "Daniele Rodrigues Campos", "article 'Daniela Rodrigues, representante da ANS'; her turn is the ANS's"),
    (184, "Guilherme Coutinho Calheiros", "article 'Guilherme Calheiro'"),
    (206, "Rogean Vinícius Santos Soares", "article 'Vinicius Soares, presidente da ANPG'"),
]


def floor_holders(transcript):
    """Same speakers and texts as lds_audit_stage2.parse_turns, plus what the header says about them."""
    tn = re.sub(r"\s+", " ", transcript)
    heads = list(HEADER.finditer(tn))
    text, hon, det, chair = defaultdict(list), defaultdict(set), defaultdict(set), defaultdict(bool)
    for k, m in enumerate(heads):
        name, info = parse_header(m)
        stop = heads[k + 1].start() if k + 1 < len(heads) else len(tn)
        text[name].append(tn[m.end():stop].strip())
        hon[name].add(info["honorific"])
        det[name].add(info["detail"])
        chair[name] |= bool(info["chair_role"])
    out = {}
    for sp, parts in text.items():
        t = " ".join(parts)
        if len(t) > MIN_FLOOR:
            g = {{"O SR.": "M", "A SRA.": "F"}[h] for h in hon[sp]}
            out[sp] = dict(floor_chars=len(t), gender_header=g.pop() if len(g) == 1 else None,
                           parl_header=any(PARTY_UF.search(d or "") for d in det[sp]), chair=chair[sp])
    return out


def tokens(name, strip_titles=True):
    t = [x for x in re.findall(r"[a-z0-9'-]+", norm(name)) if x not in PART]
    while strip_titles and len(t) > 1 and t[0] in LEAD_TITLE:
        t = t[1:]
    return t


def article_words(materia):
    """Normalised article text without the closing byline ("Reportagem - Jose Carlos Oliveira
    Edicao - ..."), which otherwise 'names' every speaker who shares the reporter's name."""
    a = re.sub(r"[^a-z0-9' -]+", " ", norm(materia))
    cut = a.rfind(" reportagem ")
    if cut >= 0 and len(a) - cut < 250:
        a = a[:cut]
    return " " + a + " "


def strict_match(a, b):
    """Same person: equal, contained, same first and last token, or near-identical spelling.
    Deliberately stricter than build_tagged_evidence.name_match, whose last-token + first-initial
    fallback pairs "Celso Giannazi" with "Carlos Giannazi" and "Daniel Neto" with "Damiana Neto"."""
    a, b = norm(a), norm(b)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    pa = [p for p in a.split() if p not in PART]
    pb = [p for p in b.split() if p not in PART]
    if len(pa) >= 2 and len(pb) >= 2 and pa[0] == pb[0] and pa[-1] == pb[-1]:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.85


def named_rule(speaker, art, participants):
    t = tokens(speaker)
    if not t or tuple(t) in UNRESOLVABLE:
        return None, "unresolvable"
    for tt in (t, tokens(speaker, strip_titles=False)):
        for later in tt[1:]:
            if re.search(r"\b" + re.escape(tt[0]) + r"\b(?:\s+\S+){0,3}?\s+" + PARTICLES + re.escape(later) + r"\b", art):
                return True, "first name + another name token"
    if len(tokens(speaker, strip_titles=False)) == 1 and re.search(r"\b" + re.escape(t[0]) + r"\b", art):
        return True, "single-token name"
    # office word immediately before the surname ("os deputados Bacelar"); with a word in between it
    # is another person ("a deputada Daiana Santos" is not Soraya Santos)
    if len(t) > 1 and len(t[-1]) > 2 and re.search(r"\b" + OFFICE + r"\s+" + re.escape(t[-1]) + r"\b", art):
        return True, "office word + surname"
    if any(strict_match(speaker, p) for p in participants):
        return True, "metadados participant"
    return False, "not found"


def logit(formula, d):
    return smf.logit(formula, data=d).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d.hearing_id})


def perm_coef(d, y, x, covars="lf", n_perm=N_PERM):
    """Two-sided p for the coefficient of x in logit(y ~ covars + x), shuffling x among the
    floor-holders of the same hearing, so hearing composition cannot create the effect."""
    rng = np.random.default_rng(SEED)
    obs = smf.logit(f"{y} ~ {covars} + {x}", data=d).fit(disp=0).params[x]
    d = d.sort_values("hearing_id").reset_index(drop=True)
    h = pd.factorize(d.hearing_id)[0]
    hits = 0
    for _ in range(n_perm):
        dd = d.copy()
        dd[x] = d[x].values[np.lexsort((rng.random(len(d)), h))]
        hits += abs(smf.logit(f"{y} ~ {covars} + {x}", data=dd).fit(disp=0).params[x]) >= abs(obs)
    return (hits + 1) / (n_perm + 1)


def effect_row(label, d, y, x, covars="lf", perm=True):
    m = logit(f"{y} ~ {covars} + {x}", d)
    b, (lo, hi) = m.params[x], m.conf_int().loc[x]
    p_perm = f"{perm_coef(d, y, x, covars):.4f}" if perm else "—"
    return (f"| {label} | {len(d)} | {d[d[x] == 1][y].mean():.1%} | {d[d[x] == 0][y].mean():.1%} | "
            f"{np.exp(b):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}] | {m.pvalues[x]:.4f} | {p_perm} |")


def main():
    hearings = {json.loads(l)["id"]: json.loads(l) for l in open(LDS, encoding="utf-8")}
    published = [json.loads(l) for l in open(STAGE2 / "omissions.jsonl")]
    attr_rows = [json.loads(l) for l in open(ATTRS)]
    by_surname_last = {surname(a["nome"]): a for a in attr_rows}          # the original surname join, reproduced
    misses = {(h, s): e for h, s, e in MISSES}

    rows = []
    for hid in sorted({r["hearing_id"] for r in published}):
        rec = hearings[hid]
        art = article_words(rec["materia"])
        participants = [e["nome"] for e in rec["metadados"].get("envolvidos", [])]
        for sp, info in floor_holders(rec["transcricao"]).items():
            named, why = named_rule(sp, art, participants)
            if named is False and (hid, sp) in misses:
                named, why = True, "hand-checked miss"
            old_attr = by_surname_last.get(surname(sp))
            same = [a for a in attr_rows if hid in a["sample_ids"] and name_match(sp, a["nome"])]
            rows.append(dict(hearing_id=hid, speaker=sp, floor_chars=info["floor_chars"],
                             named_in_article=named, named_how=why,
                             gender=info["gender_header"], parliamentarian=info["parl_header"], chair=info["chair"],
                             role=same[0]["role_rule"] if len(same) == 1 else None,
                             in_metadados=any(strict_match(sp, p) for p in participants),
                             old_attr_same_person=(old_attr is not None and name_match(sp, old_attr["nome"])),
                             old_attr_this_hearing=(old_attr is not None and hid in old_attr["sample_ids"])))
    assert len(rows) == len(published) and all(
        (a["hearing_id"], a["speaker"], a["floor_chars"]) == (b["hearing_id"], b["speaker"], b["floor_chars"])
        for a, b in zip(rows, published)), "population differs from the published omissions.jsonl"
    for a, b in zip(rows, published):
        a.update(old_named=b["named_in_article"], old_role=b["role"], old_gender=b["gender"])
    # every omitted row with a >=4-letter token of its name in the article, for the hand check:
    # the ones in MISSES are the same person, the others are someone else
    cand = []
    for r in rows:
        if r["named_in_article"] is not True and r["named_how"] != "unresolvable" or r["named_how"] == "hand-checked miss":
            art = article_words(hearings[r["hearing_id"]]["materia"])
            hits = []
            for t in dict.fromkeys(tokens(r["speaker"], strip_titles=False)):
                m = re.search(r"\b" + re.escape(t) + r"\b", art) if len(t) >= 4 else None
                if m:
                    hits.append(f"[{t}] " + art[max(0, m.start() - 70): m.end() + 50].strip())
            if hits or r["named_how"] == "hand-checked miss":
                cand.append(dict(hearing_id=r["hearing_id"], speaker=r["speaker"], parliamentarian=r["parliamentarian"],
                                 same_person=r["named_how"] == "hand-checked miss",
                                 evidence=misses.get((r["hearing_id"], r["speaker"]), " || ".join(hits))))
    pd.DataFrame(cand).to_csv(OUT_CAND, index=False)
    with OUT_ROWS.open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("hearing_id", "speaker", "floor_chars", "named_in_article",
                                                  "named_how", "role", "gender", "parliamentarian", "chair")},
                               ensure_ascii=False) + "\n")

    df = pd.DataFrame(rows)
    cdf = pd.DataFrame(cand)
    out = [f"# The omission audit, relabelled ({TAG}, {df.hearing_id.nunique()} hearings, {len(df)} floor-holders)", "",
           "Written by `scripts/lds_omission_relabel.py`. Same population as `stage2/omissions.jsonl` "
           "(asserted row by row); only the labels change. See the script's docstring for the two defects.", ""]

    # ---------------------------------------------------------------- 1. how wrong the published labels are
    has = df[df.old_role.notna()]
    wrong_person = int((~has.old_attr_same_person).sum())
    other_hearing = int((has.old_attr_same_person & ~has.old_attr_this_hearing).sum())
    both_g = df[df.old_gender.isin(["M", "F"]) & df.gender.notna()]
    out += ["## 1. How wrong the published labels are", "",
            "**Role and gender (surname-keyed join).**", "",
            f"- Floor-holders given an attribute record: {len(has)} of {len(df)}.",
            f"- The record is a **different person** (name does not match): **{wrong_person}** ({wrong_person/len(has):.0%}).",
            f"- Same name, but the record comes from another hearing (usually the same person elsewhere): {other_hearing}.",
            f"- Published gender disagrees with the transcript honorific (O SR. / A SRA.) on "
            f"**{int((both_g.old_gender != both_g.gender).sum())} of {len(both_g)}** rows that have both "
            f"({(both_g.old_gender != both_g.gender).mean():.0%}); {int(df.old_gender.isna().sum())} rows had no gender at all, "
            f"the honorific leaves {int(df.gender.isna().sum())}.", ""]
    res = df[df.named_in_article.notna()].copy()
    res["named_in_article"] = res.named_in_article.astype(bool)
    ct = Counter(zip(res.old_named, res.named_in_article))
    meta = res[res.in_metadados]
    out += ["**Named in the article (surname substring).**", "",
            f"- Old and new labels disagree on {ct[(True, False)] + ct[(False, True)]} of {len(res)} resolvable rows: "
            f"{ct[(True, False)]} old 'named' now 'omitted' (substring or homonym hits), "
            f"{ct[(False, True)]} old 'omitted' now 'named' (long names, spellings, nicknames). "
            f"{int(df.named_in_article.isna().sum())} rows are unresolvable (the header parsed to a bare title such as 'Dr').",
            f"- Independent check: of the {len(meta)} floor-holders who are article-derived `metadados` participants "
            f"(so certainly named), the published label marks **{meta.old_named.mean():.1%}** as named.",
            f"- New labels by rule: " + ", ".join(f"{k} {v}" for k, v in Counter(res.named_how).most_common()) + ".",
            f"- Residual error of the rules, read by hand: {len(cdf)} omitted rows have a >=4-letter name token in "
            f"their article ({int(cdf.parliamentarian.sum())} parliamentarians); {int(cdf.same_person.sum())} are the same "
            f"person ({int((cdf.same_person & cdf.parliamentarian).sum())} of them parliamentarians), listed in `MISSES` "
            f"and corrected above; the rest are someone else (`{OUT_CAND.name}`).", ""]

    # ---------------------------------------------------------------- 2. omission and speaking time, before and after
    res["om_new"] = 1 - res.named_in_article.astype(int)
    res["om_old"] = 1 - res.old_named.astype(int)
    res["lf"] = np.log(res.floor_chars)
    res["decile"] = pd.qcut(res.floor_chars, 10, labels=False, duplicates="drop")
    res["female"] = (res.gender == "F").astype(int)
    res["parl"] = res.parliamentarian.astype(int)
    m_old, m_new = logit("om_old ~ lf", res), logit("om_new ~ lf", res)
    out += ["## 2. Omission and speaking time", "",
            f"Omission overall: published **{df.old_named.eq(False).mean():.1%}** (all {len(df)} rows), "
            f"relabelled **{res.om_new.mean():.1%}** ({len(res)} resolvable rows; published labels on the same rows "
            f"{res.om_old.mean():.1%}).", "",
            "| floor-time decile | n | omitted, published labels | omitted, relabelled |", "|---|---:|---:|---:|"]
    for dcl, g in res.groupby("decile"):
        out.append(f"| {int(dcl)} | {len(g)} | {g.om_old.mean():.1%} | {g.om_new.mean():.1%} |")
    out += ["", f"log(floor chars), cluster-robust logit: published labels {m_old.params.lf:.3f} (z = {m_old.tvalues.lf:.1f}); "
            f"relabelled {m_new.params.lf:.3f} (z = {m_new.tvalues.lf:.1f}).", ""]

    # ---------------------------------------------------------------- 3. identity after speaking time
    g = res[res.gender.notna()]
    hdr = ("| model | n | omitted, group=1 | omitted, group=0 | odds ratio of omission, group=1 [95% CI] | "
           "cluster-robust p | within-hearing perm. p |")
    out += ["## 3. Identity, after speaking time", "",
            "`omitted ~ log(floor chars) + group`, cluster-robust by hearing; the permutation shuffles the group "
            f"label among the floor-holders of the same hearing ({N_PERM} draws) and refits.", "",
            "**Gender** (group = woman, from the transcript honorific; every floor-holder):", "", hdr, "|---|---:|---:|---:|---:|---:|---:|",
            effect_row("relabelled", g, "om_new", "female"),
            effect_row("published 'named', honorific gender", g, "om_old", "female", perm=False)]
    go = res[res.old_gender.isin(["M", "F"])].copy()
    go["female_old"] = (go.old_gender == "F").astype(int)
    out += [effect_row("as published (surname-joined gender)", go, "om_old", "female_old", perm=False), ""]
    res["chair_i"] = res.chair.astype(int)
    out += ["**Parliamentarian** (group = party-UF in the transcript header; every floor-holder):", "", hdr,
            "|---|---:|---:|---:|---:|---:|---:|",
            effect_row("relabelled", res, "om_new", "parl"),
            effect_row("relabelled, chairs excluded", res[~res.chair], "om_new", "parl"),
            effect_row("relabelled, chair as a covariate", res, "om_new", "parl", covars="lf + chair_i"),
            effect_row("group = session chair (vs everyone else), relabelled", res, "om_new", "chair_i"),
            effect_row("relabelled, without the hand-checked misses",
                       res.assign(om_nm=np.where(res.named_how == "hand-checked miss", 1, res.om_new)), "om_nm", "parl", perm=False),
            effect_row("published 'named', header role", res, "om_old", "parl", perm=False)]
    pm = res[res.parl == 1].floor_chars.median()
    om_ = res[res.parl == 0].floor_chars.median()
    fm, mm = g[g.female == 1].floor_chars.median(), g[g.female == 0].floor_chars.median()
    out += ["", f"Floor time itself (median chars): parliamentarians {pm:.0f}, everyone else {om_:.0f} "
            f"(Mann-Whitney p = {stats.mannwhitneyu(res[res.parl == 1].floor_chars, res[res.parl == 0].floor_chars).pvalue:.2g}); "
            f"women {fm:.0f}, men {mm:.0f} (p = {stats.mannwhitneyu(g[g.female == 1].floor_chars, g[g.female == 0].floor_chars).pvalue:.2g}).", ""]

    # summary table: every floor-holder, groups read off the transcript header
    res["parl_nc"] = ((res.parl == 1) & ~res.chair).astype(int)
    m3 = logit("om_new ~ lf + chair_i + parl_nc", res)
    mg = logit("om_new ~ lf + female", g)

    def matched(d):
        per = [s.om_new.mean() for _, s in d.groupby("decile") if len(s) >= 5]
        return float(np.mean(per)) if per else float("nan")

    def ci(m, x):
        lo, hi = m.conf_int().loc[x]
        return f"{np.exp(m.params[x]):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}], p = {m.pvalues[x]:.3g}"
    out += ["**Summary** (relabelled, every resolvable floor-holder; floor-matched = mean of the "
            "within-decile omission rates, deciles with >= 5 rows; odds ratio of omission after log floor time):", "",
            "| group | n | omitted, raw | omitted, floor-matched | odds ratio [95% CI] |", "|---|---:|---:|---:|---:|"]
    for label, d, oddsr in [
            ("session chair", res[res.chair], ci(m3, "chair_i") + " (vs everyone else)"),
            ("other parliamentarians", res[res.parl_nc == 1], ci(m3, "parl_nc") + " (vs everyone else)"),
            ("everyone else", res[(res.parl == 0) & ~res.chair], "reference"),
            ("women", g[g.female == 1], ci(mg, "female") + " (vs men)"),
            ("men", g[g.female == 0], "reference")]:
        out.append(f"| {label} | {len(d)} | {d.om_new.mean():.1%} | {matched(d):.1%} | {oddsr} |")
    out += ["", f"Permutation p (within hearing, {N_PERM} draws): chair {perm_coef(res, 'om_new', 'chair_i', 'lf + parl_nc'):.4f}, "
            f"other parliamentarians {perm_coef(res, 'om_new', 'parl_nc', 'lf + chair_i'):.4f}.", ""]

    # finer roles: published join vs same-person join (selected population)
    def role_block(d, y, col):
        d = d[d[col].notna()]
        base = smf.logit(f"{y} ~ lf", data=d).fit(disp=0)
        full = smf.logit(f"{y} ~ lf + C({col}, Treatment('parliamentarian'))", data=d).fit(disp=0)
        lr = 2 * (full.llf - base.llf)
        k = int(full.df_model - base.df_model)
        return len(d), lr, k, stats.chi2.sf(lr, k), full
    n1, lr1, k1, p1, _ = role_block(res.assign(r=res.old_role), "om_old", "r")
    n2, lr2, k2, p2, full2 = role_block(res, "om_new", "role")
    n3, lr3, k3, p3, _ = role_block(res[~res.chair], "om_new", "role")
    joined = res[res.role.notna()]
    out += ["**Finer roles** (joint likelihood-ratio test of the role block after log floor time):", "",
            "| labels | n | LR chi2 | df | p |", "|---|---:|---:|---:|---:|",
            f"| as published (surname join, published 'named') | {n1} | {lr1:.2f} | {k1} | {p1:.3g} |",
            f"| same-person join, relabelled | {n2} | {lr2:.2f} | {k2} | {p2:.3g} |",
            f"| same-person join, relabelled, chairs excluded | {n3} | {lr3:.2f} | {k3} | {p3:.3g} |", "",
            f"The same-person join covers {len(joined)} floor-holders and is selected on the outcome: "
            f"{joined.named_in_article.mean():.0%} of them are named, against "
            f"{res[res.role.isna()].named_in_article.mean():.0%} of the rest. Read the finer roles as supporting only.", "",
            "| role (vs parliamentarian) | odds ratio of omission | p |", "|---|---:|---:|"]
    for k in full2.params.index:
        if k.startswith("C(role"):
            out.append(f"| {k.split('[T.')[1].rstrip(']')} | {np.exp(full2.params[k]):.2f} | {full2.pvalues[k]:.4f} |")
    OUT_DOC.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwrote {OUT_ROWS} and {OUT_DOC}")


if __name__ == "__main__":
    main()
