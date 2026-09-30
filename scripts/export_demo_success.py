"""Export the participants that retrieve-and-regenerate fixed, as worked cases for the demo. CPU only.

A case is a participant whose baseline summary had at least one opinion the primary blind raters
(Sonnet) judged unsupported and whose rag:probe summary had none, i.e. the "fixed" participants of
the blind rounds in ROUNDS; their number is asserted against `expected_fixed`, the count reported by
synth3_results.py. Nothing is generated or judged here. A participant whose flagged line is still in
the corrected summary word for word, or who had no retrieval at all, is treated as rater noise and
listed in excluded_rater_noise.json instead. Tiers:
  A  the probe fired while the model was writing the flagged line (the partial line at the trigger
     is a prefix of it), the regenerated line is judged supported, and the Opus re-annotation
     (Qwen3-4B only) agrees on both lines and finds nothing unsupported in the corrected summary;
  B  as A, on the primary raters only;
  C  fixed at the participant level, but the flagged line changed downstream of an earlier
     regeneration rather than being regenerated itself.
Inputs: the rounds' rater files and key under data/publichearingbr/annotation/, the generations
under data/publichearingbr/longcontext_generation/, and the LDS file for each hearing's topic,
headline and date (located with the LDS glob; left out if not found).

Usage: uv run python scripts/export_demo_success.py
Output: demo_success/ at the repo root: README.md, cases.md, index.json, excluded_rater_noise.json
and cases/<case_id>.json.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from ocarandu.data.publichearingbr import DEFAULT_LDS_PATH

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "publichearingbr"
OUT = ROOT / "demo_success"
LDS = [str(DEFAULT_LDS_PATH)] if DEFAULT_LDS_PATH.exists() else []

ROUNDS = [
    dict(tag="qwen3_4b_instruct_2507", model="Qwen3-4B-Instruct-2507", round="synth3_qwen_rag480",
         judges={"sonnet": "raw", "opus": "raw_opus"}, rag="qwen3_4b_instruct_2507_rag480",
         base="qwen3_4b_instruct_2507_lc_xfitA", expected_fixed=34),
    dict(tag="llama_31_8b_instruct", model="Llama-3.1-8B-Instruct", round="synth3_rag480",
         judges={"sonnet": "raw"}, rag="llama_31_8b_instruct_rag480",
         base="llama_31_8b_instruct_lc_xfitA", expected_fixed=42),
]
POS, NEG = "sustentada", "nao_sustentada"
MARKER = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•])\s*")


def clean(s):
    return MARKER.sub("", s).strip().strip("*").strip()


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def hearings():
    if not LDS:
        return {}
    out = {}
    for r in load_jsonl(LDS[0]):
        lines = [l.strip() for l in r["materia"].splitlines() if l.strip()]
        date = re.search(r"\d{2}/\d{2}/\d{4}", r["materia"])
        out[int(r["id"])] = {"hearing_id": int(r["id"]), "topic": r["metadados"].get("assunto"),
                             "article_headline": lines[0] if lines else None,
                             "article_date": date.group(0) if date else None}
    return out


def export_round(cfg, hearing_info):
    adir = D / "annotation" / cfg["round"]
    key = [k for k in csv.DictReader(open(D / "annotation" / f"{cfg['round']}_key" / "key.csv")) if k["kind"] == "pipeline"]
    judged = {}
    for judge, sub in cfg["judges"].items():
        judged[judge] = {}
        for f in sorted((adir / sub).glob("set-*.jsonl")):
            for r in load_jsonl(f):
                judged[judge][r["id"]] = r

    # (sample_id, speaker) -> arm -> opinion_idx -> {text, judgements}; replicate 0 decides status,
    # exactly as synth3_results.py does (the replicated blocks are only for kappa)
    parts = defaultdict(lambda: defaultdict(dict))
    for k in key:
        if k["replicate"] != "0":
            continue
        j = {judge: {f: judged[judge][k["id"]].get(f) for f in ("support", "form_issue", "comment")}
             for judge in judged if k["id"] in judged[judge]}
        parts[(int(k["sample_id"]), k["speaker"])][k["arm"]][int(k["opinion_idx"])] = {
            "item_id": k["id"], "text": k["opinion"], "judges": j}

    def unsupported(arm_ops, judge):
        return any(o["judges"].get(judge, {}).get("support") == NEG for o in arm_ops.values())

    fixed = [p for p, arms in parts.items() if {"baseline", "rag"} <= set(arms)
             and unsupported(arms["baseline"], "sonnet") and not unsupported(arms["rag"], "sonnet")]
    assert len(fixed) == cfg["expected_fixed"], (cfg["round"], len(fixed), "does not reproduce synth3_results")

    gdir = D / "longcontext_generation"
    base = {(r["sample_id"], r["speaker"]): r for r in load_jsonl(gdir / cfg["base"] / "generations.jsonl") if r["condition"] == "baseline"}
    rag = {(r["sample_id"], r["speaker"]): r for r in load_jsonl(gdir / cfg["rag"] / "generations.jsonl") if r["condition"] == "rag:probe"}

    cases, excluded = [], []
    for p in fixed:
        arms, b, g = parts[p], base[p], rag[p]
        for arm, row in (("baseline", b), ("rag", g)):  # the key and the run must describe the same text
            for i, o in arms[arm].items():
                assert row["opinions"][i] == o["text"], (cfg["round"], p, arm, i)
        assert b["chunks"] == g["chunks"], (cfg["round"], p, "evidence differs between arms")
        has_opus = "opus" in cfg["judges"]
        events = []
        for e in g.get("rag_events", []):
            before, after = clean(e["before"]), clean(e["after"])
            b_idx = next((i for i, o in arms["baseline"].items() if clean(o["text"]).startswith(before)), None)
            r_idx = next((i for i, o in arms["rag"].items() if clean(o["text"]) == after), None)
            ev = {**e, "retrieved_passages": [g["chunks"][w] for w in e["windows"]],
                  "baseline_line_idx": b_idx, "rag_line_idx": r_idx, "fixes_flagged_line": False}
            if b_idx is not None and r_idx is not None:
                bo, ro = arms["baseline"][b_idx]["judges"], arms["rag"][r_idx]["judges"]
                ev["fixes_flagged_line"] = (bo.get("sonnet", {}).get("support") == NEG
                                            and ro.get("sonnet", {}).get("support") == POS
                                            and not ro.get("sonnet", {}).get("form_issue"))
                ev["opus_agrees"] = (has_opus and bo.get("opus", {}).get("support") == NEG
                                     and ro.get("opus", {}).get("support") == POS)
            events.append(ev)
        # a "fix" the intervention did not cause: the flagged line is still there word for word (or
        # nothing was retrieved, so the text is the baseline's) and a different rater simply judged it
        # supported -- rater noise, kept out of the demo and listed with its reason
        flagged = [o["text"] for o in arms["baseline"].values() if o["judges"].get("sonnet", {}).get("support") == NEG]
        rag_texts = {clean(o["text"]) for o in arms["rag"].values()}
        noise = ("no retrieval happened: same text as the baseline, judged differently" if not events else
                 "the flagged line is unchanged in the corrected summary and was judged differently"
                 if any(clean(t) in rag_texts for t in flagged) else None)
        if noise:
            excluded.append({"model": cfg["model"], "hearing_id": p[0], "speaker": p[1], "reason": noise,
                             "flagged_lines": flagged})
            continue
        key_events = [e for e in events if e["fixes_flagged_line"]]
        if key_events and has_opus and any(e.get("opus_agrees") for e in key_events) and not unsupported(arms["rag"], "opus"):
            tier = "A"
        elif key_events:
            tier = "B"
        else:
            tier = "C"
        best = next((e for e in key_events if e.get("opus_agrees")), key_events[0] if key_events else None)
        sid, spk = p
        cases.append({
            "case_id": f"{cfg['tag']}_{sid}_{slug(spk)}", "tier": tier, "model": cfg["model"],
            "blind_round": cfg["round"], "judges": list(cfg["judges"]),
            "hearing": hearing_info.get(sid, {"hearing_id": sid}),
            "speaker": {"name": b["nome"], "transcript_name": spk, "role": b["cargo"], "floor_chars": b.get("floor_chars")},
            "speaker_windows": b["chunks"],
            "headline_fix": None if best is None else {
                "probe_p": best["p"], "words_written_when_fired": best["words_written"],
                "partial_line_when_probe_fired": clean(best["before"]),
                "baseline_line": arms["baseline"][best["baseline_line_idx"]],
                "retrieved_window_idx": best["windows"], "retrieved_passages": best["retrieved_passages"],
                "regenerated_line": arms["rag"][best["rag_line_idx"]]},
            "baseline": {"generation": b["generation"],
                         "opinions": [arms["baseline"][i] for i in sorted(arms["baseline"])]},
            "rag": {"generation": g["generation"], "n_retrievals": g.get("n_retrievals"), "events": events,
                    "opinions": [arms["rag"][i] for i in sorted(arms["rag"])]},
        })
    return cases, excluded


def md_case(c):
    h, s = c["hearing"], c["speaker"]
    L = [f"### [{c['tier']}] {s['name']} — {c['model']}", "",
         f"*{s['role']}* · audiência {h['hearing_id']}" + (f" ({h['article_date']})" if h.get("article_date") else "")
         + (f" · {h['topic']}" if h.get("topic") else ""), "", f"Arquivo: `cases/{c['case_id']}.json`", ""]
    f = c["headline_fix"]
    if f:
        judges = lambda o: ", ".join(f"{j}: {v.get('support')}" + (f" — _{v['comment']}_" if v.get("comment") else "")
                                     for j, v in o["judges"].items())
        L += [f"- **Linha original (sem apoio):** {f['baseline_line']['text']}  ", f"  ({judges(f['baseline_line'])})",
              f"- **O probe disparou** (p = {f['probe_p']:.3f}) depois de: \"{f['partial_line_when_probe_fired']}\"",
              f"- **Trechos buscados da própria fala** (janelas {f['retrieved_window_idx']}): "
              + " [...] ".join(x[:220] + "…" for x in f["retrieved_passages"]),
              f"- **Linha regenerada (com apoio):** {f['regenerated_line']['text']}  ", f"  ({judges(f['regenerated_line'])})"]
    else:
        flagged = [o for o in c["baseline"]["opinions"] if o["judges"].get("sonnet", {}).get("support") == NEG]
        L += [f"- **Linha(s) original(is) sem apoio:** " + " | ".join(o["text"] for o in flagged),
              f"- A correção veio de uma regeneração anterior na mesma lista ({c['rag']['n_retrievals']} busca(s)); "
              "a linha marcada mudou por consequência, não foi ela a regenerada."]
    return "\n".join(L + [""])


def main():
    info = hearings()
    cases, excluded = [], []
    for cfg in ROUNDS:
        c, e = export_round(cfg, info)
        cases += c
        excluded += e
    cases.sort(key=lambda c: (c["tier"], c["model"] != "Qwen3-4B-Instruct-2507", -(c["headline_fix"] or {}).get("probe_p", 0)))
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "cases").glob("*.json"):
        old.unlink()
    for c in cases:
        json.dump(c, open(OUT / "cases" / f"{c['case_id']}.json", "w"), ensure_ascii=False, indent=2)
    index = [{"case_id": c["case_id"], "tier": c["tier"], "model": c["model"], "speaker": c["speaker"]["name"],
              "hearing_id": c["hearing"]["hearing_id"], "topic": c["hearing"].get("topic"),
              "baseline_line": (c["headline_fix"] or {}).get("baseline_line", {}).get("text"),
              "regenerated_line": (c["headline_fix"] or {}).get("regenerated_line", {}).get("text")} for c in cases]
    json.dump(index, open(OUT / "index.json", "w"), ensure_ascii=False, indent=2)
    json.dump(excluded, open(OUT / "excluded_rater_noise.json", "w"), ensure_ascii=False, indent=2)
    counts = defaultdict(lambda: defaultdict(int))
    for c in cases:
        counts[c["model"]][c["tier"]] += 1
    table = "\n".join(f"| {m} | {v['A']} | {v['B']} | {v['C']} | {sum(v.values())} |" for m, v in counts.items())
    open(OUT / "cases.md", "w").write("# Casos para a demo, do mais limpo ao menos limpo\n\n"
                                      + "\n".join(md_case(c) for c in cases))
    exc = defaultdict(int)
    for e in excluded:
        exc[e["model"]] += 1
    exc_txt = ", ".join(f"{v} no {m}" for m, v in exc.items())
    open(OUT / "README.md", "w").write(README.format(table=table, n=len(cases), n_exc=len(excluded), exc=exc_txt))
    print(f"{len(cases)} cases -> {OUT}; excluded as rater noise: {exc_txt}")
    print(table)


README = """# demo_success — casos em que a correção automática funcionou

Gerado por `scripts/export_demo_success.py` (CPU, nada foi regerado nem rejulgado). Serve para a
demo *mocked* da intervenção: detectar → buscar a própria fala → regenerar a linha.

## O que é um caso

Um participante de audiência cujo resumo **original** (baseline) tinha pelo menos uma opinião que
os juízes cegos marcaram como **sem apoio** na fala dele, e cujo resumo com **busca e regeneração**
(`rag:probe`) não tinha nenhuma. É exatamente a coluna "fixed" da comparação pareada que
`scripts/synth3_results.py` reporta para as rodadas cegas (o script confere essas contagens).

Desses, {n_exc} ({exc}) **ficaram de fora** e estão em `excluded_rater_noise.json`: a linha marcada
continua palavra por palavra no resumo corrigido, ou não houve busca nenhuma (texto idêntico ao
original), e outro juiz simplesmente a julgou com apoio. Isso é ruído do juiz, não correção, e não
pode aparecer na demo. Sobram {n} casos:

| modelo | A | B | C | total |
|---|---:|---:|---:|---:|
{table}

- **A (use primeiro):** o probe disparou enquanto o modelo escrevia *a própria linha* que os juízes
  marcaram; a linha regenerada foi julgada com apoio; no Qwen, o segundo juiz (Opus) concorda nas
  duas linhas e não acha nada sem apoio no resumo corrigido.
- **B:** como A, só com o juiz principal (Sonnet).
- **C:** a linha marcada sumiu porque uma regeneração *anterior* na mesma lista mudou o rumo do
  texto, e ela nunca chegou a ser escrita; não foi ela a regenerada. Menos bom para contar a história.

## Arquivos

- `cases.md`: leitura rápida de todos os casos, na ordem acima.
- `excluded_rater_noise.json`: os casos tirados e o motivo (não usar).
- `index.json`: uma linha por caso (id, nível, modelo, falante, linha original, linha regenerada).
- `cases/<case_id>.json`: tudo o que a demo precisa:
  - `hearing` (id, assunto, manchete e data da matéria), `speaker` (nome, cargo);
  - `speaker_windows`: todas as falas da pessoa em janelas de 100 palavras (o que o modelo e os juízes viram);
  - `headline_fix`: a linha original, o texto parcial no momento em que o probe disparou e a
    probabilidade dele, os trechos buscados e a linha regenerada, cada linha com o veredito e o
    comentário de cada juiz;
  - `baseline` e `rag`: o resumo inteiro dos dois lados, opinião por opinião com os vereditos, e
    todos os eventos de busca (`events`: linha, palavras escritas, p do probe, janelas, antes/depois).

## O que a demo pode e não pode dizer

- Pode: "quando o modelo ia escrever algo que a pessoa não disse, buscamos o trecho da própria fala
  e regeneramos a linha; juízes cegos julgaram a linha nova como sustentada".
- **Não** apresentar estes {n} casos como típicos: são os que funcionaram, escolhidos a dedo. O
  efeito médio e os casos quebrados (coluna "broken") estão no `results.md` de cada rodada, escrito
  por `scripts/synth3_results.py`.
- **Não** dizer que "o probe escolhe as linhas certas": estes casos mostram a busca corrigindo uma
  linha, não que o probe escolha linhas melhor do que outro gatilho. O limiar do probe é calibrado
  para disparar numa fração fixa das linhas (`OCARANDU_RAG_LINE_RATE` em `scripts/lc_retrieve_regen.py`).
- Os juízes são modelos de linguagem (Claude), cegos e calibrados contra o rótulo humano do dataset,
  não anotadores humanos.
"""

if __name__ == "__main__":
    main()
