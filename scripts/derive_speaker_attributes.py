"""Derive speaker-identity attributes for every distinct (nome, cargo) pair in
PublicHearingBR's NLI file.

Sources, in order of trust:
  1. the cargo string (rules in ocarandu.data.speaker_attributes): party, UF,
     grammatical gender of the title, coarse role taxonomy.
  2. Camara dos Deputados open-data API (deputies only): sexo, party and UF
     at lookup time, birth UF. Legislatures 56 and 57 cover Nov 2021 - May
     2024. The API gives the *current* party, so a party parsed from cargo
     wins when both exist.
  3. pt.wikipedia (MediaWiki query API): does an article exist for the name?
     An entity-popularity proxy (Mallen et al. 2023).

Nothing here is human-annotated; the coverage printed per attribute (share of
opinions with a usable value) should be reported with any table built on it.

Usage: uv run python scripts/derive_speaker_attributes.py   (needs network access)
Output: data/publichearingbr/speaker_attributes.jsonl
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests

from ocarandu.data.publichearingbr import load_nli_opinions
from ocarandu.data.speaker_attributes import (
    gender_from_title,
    is_parliamentarian,
    normalize,
    parse_party,
    parse_uf,
    role_from_title,
)

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "publichearingbr" / "speaker_attributes.jsonl"
CAMARA = "https://dadosabertos.camara.leg.br/api/v2"
WIKI_API = "https://pt.wikipedia.org/w/api.php"
UA = {"User-Agent": "ocarandu-research/0.1 (PUCRS MALTA Lab; academic use; ramiro.barros@edu.pucrs.br)", "accept": "application/json"}
SLEEP = 0.15


def _get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=20)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(2 + attempt * 2)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            time.sleep(1 + attempt)
    return None


_camara_cache: dict[str, dict | None] = {}


def camara_lookup(nome: str) -> dict | None:
    """Best-effort match of a parliamentary name across legislatures 56/57."""
    key = normalize(nome)
    if key in _camara_cache:
        return _camara_cache[key]
    hit = None
    for legislatura in (57, 56):
        data = _get(f"{CAMARA}/deputados", {"nome": nome, "idLegislatura": legislatura, "itens": 10})
        time.sleep(SLEEP)
        rows = (data or {}).get("dados", [])
        exact = [d for d in rows if normalize(d["nome"]) == key]
        if exact:
            hit = exact[0]
            break
        if len(rows) == 1:
            hit = rows[0]
            break
    if hit is not None:
        detail = _get(f"{CAMARA}/deputados/{hit['id']}")
        time.sleep(SLEEP)
        d = (detail or {}).get("dados", {})
        hit = {
            "camara_id": hit["id"],
            "camara_nome": hit["nome"],
            "camara_party_current": hit.get("siglaPartido"),
            "camara_uf": hit.get("siglaUf"),
            "camara_sexo": d.get("sexo"),
            "camara_uf_nascimento": d.get("ufNascimento"),
            "camara_escolaridade": d.get("escolaridade"),
        }
    _camara_cache[key] = hit
    return hit


def wikipedia_batch(names: list[str], batch_size: int = 50) -> dict[str, dict]:
    """pt.wikipedia existence via the batched MediaWiki query API (the
    per-page REST summary endpoint is rate-limited). Follows redirects and
    title normalisation; flags disambiguation pages; keeps article length
    (bytes) as a crude popularity proxy."""
    out = {}
    for i in range(0, len(names), batch_size):
        chunk = names[i:i + batch_size]
        data = _get(WIKI_API, {"action": "query", "titles": "|".join(chunk), "redirects": 1,
                               "prop": "pageprops|info", "ppprop": "disambiguation|wikibase_item",
                               "format": "json", "formatversion": 2})
        time.sleep(SLEEP)
        q = (data or {}).get("query", {})
        # map requested name -> final title through normalisation + redirects
        final = {n: n for n in chunk}
        for step in ("normalized", "redirects"):
            for m in q.get(step, []) or []:
                for k, v in final.items():
                    if v == m["from"]:
                        final[k] = m["to"]
        pages = {p["title"]: p for p in q.get("pages", [])}
        for n in chunk:
            p = pages.get(final[n], {})
            exists = bool(p) and not p.get("missing")
            out[n] = {"wiki_exists": exists and "disambiguation" not in p.get("pageprops", {}),
                      "wiki_title": p.get("title") if exists else None,
                      "wiki_is_disambiguation": exists and "disambiguation" in p.get("pageprops", {}),
                      "wiki_length": p.get("length") if exists else None,
                      "wikidata_id": p.get("pageprops", {}).get("wikibase_item") if exists else None}
    return out


def main():
    opinions = load_nli_opinions()
    speakers = {}
    for o in opinions:
        key = (o.person_name, o.person_role)
        speakers.setdefault(key, {"n_opinions": 0, "n_hallucinated": 0, "sample_ids": set()})
        speakers[key]["n_opinions"] += 1
        speakers[key]["n_hallucinated"] += int(o.is_hallucination)
        speakers[key]["sample_ids"].add(o.sample_id)
    print(f"{len(speakers)} distinct (nome, cargo) speakers over {len(opinions)} opinions", file=sys.stderr)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wiki = wikipedia_batch(sorted({nome for nome, _ in speakers}))
    print(f"wikipedia batch done: {sum(v['wiki_exists'] for v in wiki.values())}/{len(wiki)} names have an article", file=sys.stderr)
    rows = []
    for i, ((nome, cargo), agg) in enumerate(sorted(speakers.items())):
        row = {
            "nome": nome,
            "cargo": cargo,
            "n_opinions": agg["n_opinions"],
            "n_hallucinated": agg["n_hallucinated"],
            "sample_ids": sorted(agg["sample_ids"]),
            "is_parliamentarian": is_parliamentarian(cargo),
            "party_cargo": parse_party(cargo),
            "uf_cargo": parse_uf(cargo),
            "gender_title": gender_from_title(cargo),
            "role_rule": role_from_title(cargo, nome),
        }
        if row["is_parliamentarian"]:
            row.update(camara_lookup(nome) or {"camara_id": None})
        row.update(wiki[nome])
        # consolidated fields, with provenance
        row["party"] = row["party_cargo"] or row.get("camara_party_current")
        row["party_source"] = "cargo" if row["party_cargo"] else ("camara_api" if row.get("camara_party_current") else None)
        row["gender"] = row["gender_title"] or row.get("camara_sexo")
        row["gender_source"] = "title" if row["gender_title"] else ("camara_api" if row.get("camara_sexo") else None)
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(speakers)} speakers processed", file=sys.stderr)

    with open(OUT_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # coverage report (weighted by opinions, which is the unit of analysis)
    def cov(field):
        n = sum(r["n_opinions"] for r in rows if r.get(field) not in (None, "other", False))
        return n / len(opinions)

    print("\n=== coverage (share of opinions with a usable value) ===")
    print(f"role (not 'other'):    {cov('role_rule'):.1%}")
    print(f"gender:                {cov('gender'):.1%}  (title {sum(r['n_opinions'] for r in rows if r['gender_source']=='title')/len(opinions):.1%}, camara {sum(r['n_opinions'] for r in rows if r['gender_source']=='camara_api')/len(opinions):.1%})")
    print(f"party (parl. only):    {sum(r['n_opinions'] for r in rows if r['party'])/max(1,sum(r['n_opinions'] for r in rows if r['is_parliamentarian'])):.1%} of parliamentarian opinions")
    print(f"camara match (parl.):  {sum(1 for r in rows if r['is_parliamentarian'] and r.get('camara_id'))}/{sum(1 for r in rows if r['is_parliamentarian'])} speakers")
    print(f"wikipedia article:     {cov('wiki_exists'):.1%} of opinions; {sum(1 for r in rows if r['wiki_exists'])}/{len(rows)} speakers")
    print("\nrole distribution (speakers):", Counter(r["role_rule"] for r in rows))
    print("gender distribution (speakers):", Counter(r["gender"] for r in rows))
    print("wiki_exists by role (speakers):")
    for role in sorted({r["role_rule"] for r in rows}):
        sub = [r for r in rows if r["role_rule"] == role]
        print(f"  {role:16s} {sum(r['wiki_exists'] for r in sub)}/{len(sub)}")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
