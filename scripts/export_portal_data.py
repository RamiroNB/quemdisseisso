"""Export the data of the "Quem disse isso?" portal to app/portal/data/. CPU only, ~2 min.

Nothing is generated or judged here. Inputs, under data/publichearingbr/: the Qwen3-4B
retrieve-and-regenerate run (rag480: 100 hearings, 480 participants), its baseline (lc_xfitA) and
blind round (synth3_qwen_rag480, read as in scripts/export_demo_success.py); the relabelled omission
rows (lds_audit/llama_31_8b_instruct/stage2/omissions_v2.jsonl, scripts/lds_omission_relabel.py);
speaker_attributes.jsonl; the token-probe stream (lc_stream/<model>/). Also the LDS file, and
demo_success/cases/ at the repo root, which supplies each summary's case id and tier (run
scripts/export_demo_success.py first).
  h/<id>.json  per hearing: every floor-holder with the relabelled labels; speaking time estimated
               as floor characters / mean characters per word / 140 words per minute; up to 6 themes
               (TF-IDF uni/bigrams over the 206 transcripts, used by >= 2 floor-holders; a navigation
               aid, not a result); the baseline and rag:probe summaries, each line with its blind
               verdicts (sonnet = primary raters, opus = re-annotation), and the retrieval events.
  game.json    a few real lines replayed word by word with the layer-21 token probe that ran as the
               trigger, the per-dimension contributions to its logit (coef * standardised activation,
               quantised to int8 for the ASCII field), and the probe plane (see probe_plane).
  index.json   the hearing list and the corpus numbers, recomputed from the same files except the
               pre-registered p-value, the retrieval line rate and the model-audit figures, which are
               hard-coded.

Usage: uv run python scripts/export_portal_data.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import base64  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import unicodedata  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tagged_evidence import HEADER, name_match, parse_header  # noqa: E402

from ocarandu.data.publichearingbr import DEFAULT_LDS_PATH  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "publichearingbr"
OUT = ROOT / "app" / "portal" / "data"
LDS = str(DEFAULT_LDS_PATH)
TAG, MODEL, ROUND = "qwen3_4b_instruct_2507", "Qwen3-4B-Instruct-2507", "synth3_qwen_rag480"
RAG, BASE = f"{TAG}_rag480", f"{TAG}_lc_xfitA"
LAYER = 21
POS, NEG = "sustentada", "nao_sustentada"
WPM, CHARS_PER_WORD = 140, None  # CHARS_PER_WORD is measured on the transcripts below
MARKER = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•])\s*")
# Hearings in the carousel's front row, the showcase (38) first. Hearing 91 is exported but not featured.
FEATURED = [38, 75, 141, 81, 99, 103, 148]

# Navigation labels, assigned by reading each hearing's `assunto` (not a result; shown as filter chips).
CATS = {
    1: "Justiça Tecnologia", 4: "Infraestrutura Tecnologia", 5: "Direitos-humanos Saúde", 9: "Direitos-humanos",
    11: "Meio-ambiente", 12: "Direitos-humanos", 16: "Segurança Justiça", 17: "Justiça", 19: "Infraestrutura",
    20: "Direitos-humanos", 21: "Economia", 24: "Educação", 25: "Agropecuária Meio-ambiente", 26: "Economia Trabalho",
    32: "Direitos-humanos", 34: "Saúde Tecnologia", 36: "Economia Infraestrutura", 37: "Esporte Direitos-humanos",
    38: "Tributação Agropecuária", 49: "Infraestrutura", 51: "Educação", 53: "Cultura Direitos-humanos",
    56: "Saúde Esporte", 57: "Segurança Educação", 61: "Economia", 62: "Saúde", 63: "Direitos-humanos",
    65: "Trabalho Justiça", 67: "Trabalho Saúde", 70: "Energia Meio-ambiente", 73: "Economia", 74: "Direitos-humanos",
    75: "Trabalho Educação", 78: "Energia Economia", 79: "Tributação", 80: "Justiça", 81: "Agropecuária Economia",
    82: "Cultura", 84: "Direitos-humanos", 85: "Direitos-humanos", 86: "Economia Meio-ambiente", 87: "Trabalho Tecnologia",
    88: "Tecnologia Educação", 90: "Esporte Direitos-humanos", 91: "Energia", 92: "Trabalho Educação", 97: "Economia",
    99: "Direitos-humanos", 103: "Segurança", 104: "Energia", 106: "Agropecuária", 107: "Infraestrutura Meio-ambiente",
    108: "Economia Cultura", 112: "Trabalho Tecnologia", 114: "Saúde", 115: "Meio-ambiente", 121: "Direitos-humanos Justiça",
    122: "Agropecuária", 123: "Saúde", 125: "Educação", 126: "Saúde", 127: "Direitos-humanos Saúde",
    130: "Cultura Tecnologia", 131: "Infraestrutura", 134: "Energia Meio-ambiente", 137: "Educação", 140: "Saúde",
    141: "Educação Direitos-humanos", 142: "Economia", 144: "Infraestrutura", 146: "Saúde",
    148: "Meio-ambiente Agropecuária", 150: "Cultura", 152: "Economia", 155: "Direitos-humanos", 157: "Tributação Saúde",
    159: "Saúde", 160: "Educação", 162: "Saúde Trabalho", 164: "Infraestrutura Saúde", 166: "Tecnologia", 168: "Educação",
    169: "Meio-ambiente", 171: "Saúde", 176: "Educação Direitos-humanos", 178: "Energia Meio-ambiente", 181: "Saúde",
    183: "Meio-ambiente Energia", 184: "Tecnologia Economia", 185: "Saúde", 189: "Tecnologia Economia",
    192: "Tecnologia", 195: "Tecnologia Economia", 197: "Tecnologia", 198: "Direitos-humanos",
    201: "Justiça Direitos-humanos", 202: "Saúde Trabalho", 203: "Direitos-humanos", 204: "Saúde", 205: "Segurança",
}

# The game replays these lines (hearing, speaker, which baseline line, why it is in the game).
GAME = [
    dict(h=38, speaker="Daniel Panizzi", line=0, kind="calm",
         why="warm-up: the probe stays below the threshold while the model writes a supported line"),
    dict(h=38, speaker="Daniel Panizzi", line=1, kind="fix",
         why="the showcase: the alarm fires one word before an inverted comparison; retrieval fixes the line"),
    dict(h=None, speaker=None, line=None, kind="neutral",
         why="the alarm fires on a line that was already supported; the rewrite keeps it supported"),
    dict(h=75, speaker="Juvândia Moreira Leite", line=None, kind="fix",
         why="a second fix, tier A (both raters agree on both lines)"),
]

STOP = set("""
a à ao aos as às até com como da das de dela dele deles delas do dos e é em entre era essa esse esta este eu foi há
isso isto já la lá lhe mais mas me mesmo meu minha muito na nas nem no nos nós num numa o os ou para pela pelas pelo
pelos por porque pois qual quando que quem se sem ser seu sua são só também te tem têm tu um uma umas uns vai vão
vocês você ele ela eles elas nós nosso nossa nossos nossas aqui ali aí então assim agora ainda sobre sua suas seus
estar estamos estão está esteja estou fazer faz fazendo feito fica ficar fiz foram fosse gente hoje ir isso já
lado lugar maior mesma mesmos modo nada nesse nessa neste nesta nisso outro outra outros outras pode podem poder
porém pouco primeiro quanto quase querer quero sendo seja sempre ter tinha todo toda todos todas tudo vamos vez
vezes vou coisa coisas questão questões forma parte ponto sentido tipo exemplo caso casos momento tempo ano anos
dia dias bem bom boa boas bons grande grandes melhor muita muitas muitos obrigado obrigada senhor senhora senhores
senhoras sr sra srs presidente presidenta deputado deputada deputados deputadas comissão audiência pública públicas
público câmara casa palavra minutos minuto fala falar falou dizer disse digo tarde noite manhã cumprimentar
cumprimento agradecer agradeço convidado convidados convidada convidadas presença mesa excelência vossa v.exa
exa sessão reunião requerimento autoria colegas colega companheiro companheira querido querida prezado prezada
aqui acho creio certeza verdade realmente importante importância necessário precisamos preciso precisa temos
trabalho brasil país estado estados governo nacional federal lei projeto proposta dados número números situação
processo gostaria queria quer dá dar deu vem veio vêm tá né olha veja vejam ver visto vista falando dizendo
exatamente justamente simplesmente inclusive principalmente especialmente basicamente efetivamente absolutamente
sim não nao também ok aliás portanto porque enquanto onde cada qualquer algum alguma alguns algumas nenhum nenhuma
desta deste dessa desse daquele daquela aquele aquela aqueles aquelas outro mil milhões bilhões cento dois duas três
quatro cinco seis sete oito nove dez segundo segunda terceiro final início passo passos meio conta tanto tanta\ncontra sobre através junto frente perante durante após antes depois além dentro fora acima abaixo
""".split())


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def clean(s):
    return MARKER.sub("", s).strip().strip("*").strip()


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", fold(s)).strip("-")[:48]


def parse_turns(transcript):
    """speaker -> (list of turn texts, header details) -- the parse the omission audit used."""
    tn = re.sub(r"\s+", " ", transcript)
    heads = list(HEADER.finditer(tn))
    turns, details = defaultdict(list), defaultdict(Counter)
    for k, m in enumerate(heads):
        name, det = parse_header(m)
        stop = heads[k + 1].start() if k + 1 < len(heads) else len(tn)
        turns[name].append(tn[m.end():stop].strip())
        details[name][(det["honorific"], det["detail"])] += 1
        if det["chair_role"]:
            details[name][("chair", det["chair_role"])] += 1
    return turns, details


PARTY = re.compile(r"(?:Bloco/)?([A-Za-zÀ-ú]+(?:\s[A-Za-zÀ-ú]+)?)\s*-\s*([A-Z]{2})\b")


def pretty_party(p):
    """'REPUBLICANOS-RS' -> 'Republicanos-RS'; acronyms stay as written."""
    if not p:
        return None
    party, uf = p.rsplit("-", 1)
    keep = {"PT", "PL", "PP", "PSD", "MDB", "PSB", "PDT", "PSDB", "PSOL", "PSC", "PTB", "PV", "PMN", "PROS", "PCDOB", "DEM", "PPS"}
    party = party if party in keep else party.capitalize()
    return f"{'PCdoB' if party == 'PCDOB' else party}-{uf}"


def header_party(details):
    """most frequent 'PARTY - UF' parenthetical of a speaker's headers -> 'PT-SP' or None."""
    for (hn, det), _n in details.most_common():
        if hn == "chair":
            continue
        m = PARTY.search(det or "")
        if m:
            return f"{m.group(1).upper()}-{m.group(2)}"
    return None


WORD = re.compile(r"[A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)*|\d+")
BREAK = re.compile(r"[.,;:!?()\[\]\"“”'‘’—–/]")
TITLES = {"ministro", "ministra", "deputado", "deputada", "senador", "senadora", "doutor", "doutora", "dr", "dra",
          "professor", "professora", "presidente", "coronel", "general", "delegado", "delegada", "padre", "pastor",
          "secretario", "secretaria", "diretor", "diretora", "vereador", "vereadora", "prefeito", "prefeita"}


def stem(w):
    """crude Portuguese plural/derivation folding, only to merge near-duplicate themes (vinho/vinhos)."""
    w = re.sub(r"(oes|aes|ais|eis|is|es|s)$", "", w) if len(w) > 4 else w
    return w[:5]


def term_counts(text):
    """unigram and bigram counts of content words (bigrams never cross punctuation or a function word),
    keyed by folded form, with the surface forms seen."""
    uni, bi, surf = Counter(), Counter(), defaultdict(Counter)
    for seg in BREAK.split(text):
        prev = None
        for t in WORD.findall(seg):
            f = fold(t)
            if len(f) >= 4 and f not in STOP and not f.isdigit():
                uni[f] += 1
                surf[f][t] += 1
                if prev is not None:
                    k = prev[0] + " " + f
                    bi[k] += 1
                    surf[k][prev[1] + " " + t] += 1
                prev = (f, t)
            else:
                prev = None
    return uni, bi, surf


def theme_label(surf):
    ws = surf.split()
    if all(w[:1].isupper() for w in ws):          # a proper name or an acronym: as written
        return " ".join(w if (w.isupper() and len(w) <= 8) else w[:1].upper() + w[1:].lower() for w in ws)
    return surf[:1].upper() + surf[1:].lower()


def next_word(partial, line):
    """the word the model was writing (or about to write) when the probe fired."""
    rest = line[len(partial):]
    if rest[:1].isspace() or not partial:
        w = rest.split()[:1]
    else:  # fired inside a word ("... e redu" -> "reduzir")
        w = [partial.split()[-1] + (rest.split()[0] if rest.split() else "")]
    return w[0].strip(".,;:!?\"'()") if w else None


def main():
    global CHARS_PER_WORD
    lds = {int(r["id"]): r for r in load_jsonl(LDS)}
    om = defaultdict(list)
    for r in load_jsonl(D / "lds_audit" / "llama_31_8b_instruct" / "stage2" / "omissions_v2.jsonl"):
        om[r["hearing_id"]].append(r)
    attrs = load_jsonl(D / "speaker_attributes.jsonl")

    # ---- the experiment: runs, key, blind verdicts -------------------------------------------------
    gdir = D / "longcontext_generation"
    rag = {(r["sample_id"], r["speaker"]): r for r in load_jsonl(gdir / RAG / "generations.jsonl") if r["condition"] == "rag:probe"}
    base = {(r["sample_id"], r["speaker"]): r for r in load_jsonl(gdir / BASE / "generations.jsonl") if r["condition"] == "baseline"}
    key = [k for k in csv.DictReader(open(D / "annotation" / f"{ROUND}_key" / "key.csv")) if k["kind"] == "pipeline" and k["replicate"] == "0"]
    judged = {}
    for judge, sub in (("sonnet", "raw"), ("opus", "raw_opus")):
        judged[judge] = {}
        for f in sorted((D / "annotation" / ROUND / sub).glob("set-*.jsonl")):
            for r in load_jsonl(f):
                judged[judge][r["id"]] = r
    parts = defaultdict(lambda: defaultdict(dict))
    for k in key:
        v = {j: {"s": judged[j][k["id"]].get("support"), "f": judged[j][k["id"]].get("form_issue"),
                 "c": judged[j][k["id"]].get("comment") or ""} for j in judged if k["id"] in judged[j]}
        parts[(int(k["sample_id"]), k["speaker"])][k["arm"]][int(k["opinion_idx"])] = {"text": k["opinion"], "v": v}

    def unsupported(ops, judge="sonnet"):
        return any(o["v"].get(judge, {}).get("s") == NEG for o in ops.values())

    both = [p for p, a in parts.items() if {"baseline", "rag"} <= set(a)]
    fixed = [p for p in both if unsupported(parts[p]["baseline"]) and not unsupported(parts[p]["rag"])]
    broken = [p for p in both if not unsupported(parts[p]["baseline"]) and unsupported(parts[p]["rag"])]
    assert (len(fixed), len(broken)) == (34, 18), (len(fixed), len(broken), "does not reproduce synth3_results")
    n_b = sum(unsupported(parts[p]["baseline"]) for p in both)
    n_r = sum(unsupported(parts[p]["rag"]) for p in both)
    fixed_o = [p for p in both if unsupported(parts[p]["baseline"], "opus") and not unsupported(parts[p]["rag"], "opus")]
    broken_o = [p for p in both if not unsupported(parts[p]["baseline"], "opus") and unsupported(parts[p]["rag"], "opus")]
    demo = {}
    for p in glob.glob(str(ROOT / "demo_success" / "cases" / f"{TAG}_*.json")):
        c = json.load(open(p))
        demo[(c["hearing"]["hearing_id"], c["speaker"]["transcript_name"])] = c

    # ---- corpus: themes need document frequencies over all 206 transcripts -------------------------
    turns_all, details_all, df_uni, df_bi = {}, {}, Counter(), Counter()
    n_words = n_chars = 0
    for h, r in lds.items():
        turns, det = parse_turns(r["transcricao"])
        turns_all[h], details_all[h] = turns, det
        uni, bi, _ = term_counts(" ".join(" ".join(v) for v in turns.values()))
        df_uni.update(uni.keys())
        df_bi.update(bi.keys())
        for v in turns.values():
            for t in v:
                n_words += len(t.split())
                n_chars += len(t)
    CHARS_PER_WORD = n_chars / n_words
    N_DOC = len(lds)

    hearings = sorted({p[0] for p in rag})
    assert len(hearings) == 100 and len(rag) == 480
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "h").mkdir(exist_ok=True)
    index = []
    for h in hearings:
        r, fl = lds[h], sorted(om[h], key=lambda x: -x["floor_chars"])
        materia = [l.strip() for l in r["materia"].splitlines() if l.strip()]
        date = re.search(r"\d{2}/\d{2}/\d{4}", r["materia"]).group(0)
        body = [l for l in materia[2:] if not re.match(r"^\d{2}/\d{2}/\d{4}", l) and len(l) > 80]
        turns, det = turns_all[h], details_all[h]
        env = r["metadados"].get("envolvidos", [])

        # themes: TF-IDF uni/bigrams of the floor text, spoken by >= 2 floor-holders
        per_spk = {}
        for s in fl:
            txt = " ".join(turns.get(s["speaker"], []))
            per_spk[s["speaker"]] = term_counts(txt)
        uni_h, bi_h, surf_h = Counter(), Counter(), defaultdict(Counter)
        spread = Counter()
        for u, b, sf in per_spk.values():
            uni_h.update(u)
            bi_h.update(b)
            for k, v in sf.items():
                surf_h[k].update(v)
            spread.update(set(u) | set(b))
        name_toks = {fold(t) for s in fl for t in s["speaker"].split()} | {fold(t) for e in env for t in e["nome"].split()}
        cand = []
        art = fold(r["materia"])
        for k, c in list(uni_h.items()) + list(bi_h.items()):
            ws = k.split()
            df = (df_bi if len(ws) == 2 else df_uni)[k]
            if c < 4 or spread[k] < 2 or df < 2 or ws[0] in TITLES or any(w in name_toks for w in ws):
                continue
            if any(re.match(r"^(do|da|de|no|na|o|a|tecnologia)(.{4,})$", w) and uni_h.get(re.match(
                    r"^(do|da|de|no|na|o|a|tecnologia)(.{4,})$", w).group(2), 0) >= 2 for w in ws):
                continue  # transcription glued a preposition to the next word ("dolobby", "tecnologiablockchain")
            surf = surf_h[k].most_common(1)[0][0]
            in_art = bool(re.search(r"\b" + re.escape(k) + r"\b", art))
            proper = all(w[:1].isupper() and not w.isupper() for w in surf.split())
            if proper and not in_art:      # a person or a place nobody would call a theme of the debate
                continue
            score = c * math.log(N_DOC / (1 + df)) * (1.6 if len(ws) == 2 else 1.0)
            cand.append((score, k, surf, in_art))
        cand.sort(reverse=True)
        themes = []
        for score, k, surf, in_art in cand:
            st = {stem(w) for w in k.split()}
            if any(st & t["stems"] for t in themes):
                continue
            themes.append({"k": k, "label": theme_label(surf), "in_article": in_art, "stems": st})
            if len(themes) == 6:
                break
        for t in themes:
            t.pop("stems")

        # participants of the experiment in this hearing
        summ_keys = [p for p in rag if p[0] == h]
        speakers = []
        for rank, s in enumerate(fl):
            nm = s["speaker"]
            txt = " ".join(turns.get(nm, []))
            u, b, _ = per_spk[nm]
            tcount = [(u.get(t["k"], 0) + b.get(t["k"], 0), i) for i, t in enumerate(themes)]
            my_themes = [i for c, i in sorted(tcount, reverse=True) if c > 0][:4]
            sk = next((p for p in summ_keys if p[1] == nm), None) or next((p for p in summ_keys if name_match(p[1], nm)), None)
            hon, _ = det[nm].most_common(1)[0][0] if det.get(nm) else (None, None)
            party = header_party(det.get(nm, Counter()))
            rec = next((a for a in attrs if h in (a.get("sample_ids") or []) and name_match(a["nome"], nm)), None)
            e = next((e for e in env if name_match(e["nome"], nm)), None)
            pretty = pretty_party(party)
            if sk:
                role = rag[sk]["cargo"]
            elif s["parliamentarian"]:
                role = ("Deputada" if hon == "A SRA." else "Deputado") + (f" ({pretty})" if pretty else "")
            elif rec and rec.get("cargo"):
                role = rec["cargo"]
            elif e and e.get("cargo"):
                role = e["cargo"]
            else:
                role = "Participante da audiência"
            chair_roles = sorted({d for (hn, d), _n in det.get(nm, Counter()).items() if hn == "chair"})
            # a verbatim excerpt: the ~60-word window of the floor text densest in the hearing's themes
            words = txt.split()
            best, bi_ = -1, 0
            tk = [t["k"] for t in themes]
            for st in range(0, max(1, len(words) - 60), 20):
                w = fold(" ".join(words[st:st + 60]))
                sc = sum(w.count(t) for t in tk) + 0.001 * st / max(1, len(words))
                if sc > best:
                    best, bi_ = sc, st
            exc = " ".join(words[bi_:bi_ + 60])
            speakers.append({
                "slug": slug(rag[sk]["nome"] if sk else nm), "name": rag[sk]["nome"] if sk else nm, "tname": nm,
                "role": role, "party": pretty, "chair": bool(s["chair"]), "chair_role": chair_roles[0] if chair_roles else None,
                "parl": bool(s["parliamentarian"]), "fem": hon == "A SRA.",
                "floor": s["floor_chars"], "min": max(1, round(s["floor_chars"] / CHARS_PER_WORD / WPM)),
                "rank": rank + 1, "named": s["named_in_article"], "summary": bool(sk), "themes": my_themes,
                "excerpt": ("… " if bi_ > 0 else "") + exc + (" …" if bi_ + 60 < len(words) else ""),
                "in_article": (e["opinioes"][0] if e and e.get("opinioes") else None),
            })
            if sk:
                speakers[-1]["_key"] = sk
        # summaries
        summaries = {}
        for sp in speakers:
            sk = sp.pop("_key", None)
            if not sk:
                continue
            a, b_row, g_row = parts[sk], base[sk], rag[sk]
            assert b_row["chunks"] == g_row["chunks"]
            bl = [a["baseline"][i] for i in sorted(a["baseline"])]
            rl = [a["rag"][i] for i in sorted(a["rag"])]
            for arm, row, ops in (("baseline", b_row, bl), ("rag", g_row, rl)):
                assert [o["text"] for o in ops] == row["opinions"][:len(ops)], (h, sk, arm)
            events = []
            for e in g_row.get("rag_events", []):
                before, after = clean(e["before"]), clean(e["after"])
                b_idx = next((i for i, o in enumerate(bl) if clean(o["text"]).startswith(before)), None)
                r_idx = next((i for i, o in enumerate(rl) if clean(o["text"]) == after), None)
                nxt = None
                if b_idx is not None:
                    nxt = next_word(before, clean(bl[b_idx]["text"]))
                events.append({"b": b_idx, "r": r_idx, "p": e["p"], "ww": e["words_written"], "win": e["windows"],
                               "partial": before, "next": nxt})
            status = ("fixed" if sk in fixed else "broken" if sk in broken else
                      "bad" if unsupported(a["rag"]) else "ok")
            summaries[sp["slug"]] = {
                "baseline": [{"t": clean(o["text"]), "v": o["v"]} for o in bl],
                "rag": [{"t": clean(o["text"]), "v": o["v"]} for o in rl],
                "events": events, "windows": b_row["chunks"], "status": status,
                "case": demo[sk]["case_id"] if sk in demo else None, "tier": demo[sk]["tier"] if sk in demo else None,
            }
        n_named = sum(bool(s["named"]) for s in speakers)
        top = {s["tname"] for s in speakers[:n_named]}
        named_set = {s["tname"] for s in speakers if s["named"]}
        doc = {"id": h, "title": r["metadados"]["assunto"], "date": date, "headline": materia[0],
               "subheadline": materia[1] if len(materia) > 1 and not re.match(r"^\d{2}/", materia[1]) else None,
               "lead": body[0] if body else None, "cats": [c.replace("-", " ") for c in CATS[h].split()],
               "speakers": speakers, "themes": [{k: t[k] for k in ("label", "in_article")} for t in themes],
               "summaries": summaries, "n_named": n_named, "top_named": named_set == top and n_named > 0}
        json.dump(doc, open(OUT / "h" / f"{h}.json", "w"), ensure_ascii=False, separators=(",", ":"))
        index.append({"id": h, "title": doc["title"], "date": date, "headline": doc["headline"], "cats": doc["cats"],
                      "n": len(speakers), "n_named": n_named, "n_summ": len(summaries),
                      "people": [s["name"] for s in speakers],
                      "fixed": sum(v["status"] == "fixed" for v in summaries.values()),
                      "broken": sum(v["status"] == "broken" for v in summaries.values()),
                      "featured": FEATURED.index(h) if h in FEATURED else None})
        print(f"{h:3d} floor={len(speakers):2d} named={n_named:2d} summ={len(summaries):2d}  themes: "
              + ", ".join(t["label"] + ("*" if t["in_article"] else "") for t in themes), flush=True)

    # ---- corpus numbers (recomputed from the relabelled omission rows) ----------------------------
    rows = [x for v in om.values() for x in v if x["named_in_article"] is not None]
    fc = np.array([x["floor_chars"] for x in rows])
    omitted = np.array([not x["named_in_article"] for x in rows])
    dec = np.searchsorted(np.quantile(fc, np.linspace(0, 1, 11)[1:-1]), fc, side="right")
    by_dec = [float(omitted[dec == d].mean()) for d in range(10)]

    def floor_matched(mask):
        rates = [omitted[(dec == d) & mask].mean() for d in range(10) if ((dec == d) & mask).sum() >= 5]
        return float(np.mean(rates))
    chair = np.array([bool(x["chair"]) for x in rows])
    parl = np.array([bool(x["parliamentarian"]) for x in rows])
    corpus = {
        "omission": {"all": float(omitted.mean()), "deciles": by_dec, "n": int(len(rows)),
                     "chair_raw": float(omitted[chair].mean()), "rest_raw": float(omitted[~chair].mean()),
                     "chair_matched": floor_matched(chair), "rest_matched": floor_matched(~chair & ~parl),
                     "source": "relabelled omission rows of lds_omission_relabel.py, which also runs the tests"},
        "rag": {"participants": len(both), "fixed": len(fixed), "broken": len(broken),
                "with_unsupported_baseline": n_b / len(both), "with_unsupported_rag": n_r / len(both),
                "opus_fixed": len(fixed_o), "opus_broken": len(broken_o),
                "p_preregistered": 0.037,
                "source": "blind round synth3_qwen_rag480 (synth3_results.py); the p-value is hard-coded"},
        "audit": {"misattr_confirmed": "90 a 92%", "title_models": "2 de 4", "title_flips": "3 a 5%",
                  "source": "metadata-use and title-credibility audits (reproduce/06_audit.sh); hard-coded"},
        "tau": json.load(open(gdir / RAG / "thresholds.json"))["tau_p"], "line_rate": 0.35, "model": MODEL,
        "wpm": WPM, "chars_per_word": round(CHARS_PER_WORD, 2),
    }
    assert abs(corpus["omission"]["all"] - 0.545) < 0.002 and abs(by_dec[0] - 0.887) < 0.002 and abs(by_dec[9] - 0.151) < 0.002, corpus["omission"]
    index.sort(key=lambda x: (x["featured"] is None, x["featured"] if x["featured"] is not None else 0, -x["fixed"], x["id"]))
    cats = Counter(c for x in index for c in x["cats"])
    json.dump({"model": MODEL, "hearings": index, "categories": [c for c, _ in cats.most_common()], "corpus": corpus},
              open(OUT / "index.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n{len(index)} hearings; fixed {len(fixed)} / broken {len(broken)}; participants with an unsupported line "
          f"{n_b / len(both):.3f} -> {n_r / len(both):.3f}; omission {corpus['omission']['all']:.3f}, deciles "
          f"{by_dec[0]:.3f} .. {by_dec[9]:.3f}; chair raw {corpus['omission']['chair_raw']:.3f} matched "
          f"{corpus['omission']['chair_matched']:.3f} vs rest {corpus['omission']['rest_matched']:.3f}")
    export_game(lds, parts, base, rag, corpus["tau"])


OMV2 = load_jsonl(D / "lds_audit" / "llama_31_8b_instruct" / "stage2" / "omissions_v2.jsonl")


def probe_plane(x, tok, li, mean, scale, coef, icpt, tau, n_side=1280, n_ref=6000, pc=2, seed=0):
    """The plane the game draws when the player catches an alarm. Axis 1 is the probe's own score
    s = coef . z + intercept (z = standardised layer activation), shifted so the alarm threshold sits at 0: in
    any plane that contains the probe direction the threshold is a straight line. Axis 2 only spreads the points:
    a principal component of the activations with the probe direction removed, decorrelated from s and scaled
    to s's spread. Points: a balanced sample of baseline content-word positions (n_side whose next content word
    was absent from the speaker's turns, n_side whose word was there). The saved probe was fit on every
    baseline row, so these are training rows: the picture shows what the probe learned, not held-out accuracy."""
    rows = [i for i, t in enumerate(tok) if t["condition"] == "baseline" and t["is_content"] and not t["framing"]]
    novel = np.array([bool(tok[i]["novel"]) for i in rows])
    rng = np.random.default_rng(seed)

    def feats(sel):
        sel = np.sort(np.asarray(sel))
        v = np.asarray(x[[rows[j] for j in sel], li, :], dtype=np.float32)
        return sel, (v - mean) / scale

    lt = math.log(tau / (1 - tau))
    w = coef / np.linalg.norm(coef)
    _, zr = feats(rng.choice(len(rows), n_ref, replace=False))
    sr = zr @ coef + icpt
    zp = zr - np.outer(zr @ w, w)
    mu = zp.mean(0)
    v = np.linalg.svd(zp - mu, full_matrices=False)[2][pc]
    yr = (zr - mu) @ v
    beta = np.cov(yr, sr)[0, 1] / sr.var(ddof=1)
    yr = yr - beta * (sr - sr.mean())
    gain = sr.std() / yr.std()

    def project(zz):
        s = zz @ coef + icpt
        y = ((zz - mu) @ v - beta * (s - sr.mean())) * gain
        return np.stack([s - lt, y], 1)

    pick = np.r_[rng.choice(np.where(~novel)[0], n_side, replace=False), rng.choice(np.where(novel)[0], n_side, replace=False)]
    pick = pick[rng.permutation(len(pick))]
    sel, zs = feats(pick)
    xy = project(zs)
    lab = novel[sel]
    perm = rng.permutation(len(sel))          # feats() sorted the rows; shuffle again so particles land at random
    xy, lab = xy[perm], lab[perm]
    beyond = xy[:, 0] > 0
    print(f"plane: {len(sel)} points, beyond the threshold {int((beyond & lab).sum())} absent / {int((beyond & ~lab).sum())} present")
    plane = {"n": int(len(sel)), "x": [round(float(a), 2) for a in xy[:, 0]], "y": [round(float(a), 2) for a in xy[:, 1]],
             "novel": "".join("1" if b else "0" for b in lab), "balanced": True, "training_rows": True,
             "iso": {"0.5": round(-lt, 3), "0.9": round(math.log(9) - lt, 3)}, "pc": pc}
    return plane, project


def export_game(lds, parts, base, rag, tau):
    """Rounds of 'Seja o probe': real baseline lines with the probe that ran, word by word."""
    sdir = D / "lc_stream" / TAG
    meta = json.load(open(sdir / "meta.json"))
    li = meta["layers"].index(LAYER)
    x = np.load(sdir / "tokens.npy", mmap_mode="r")
    tok = load_jsonl(sdir / "tokens.jsonl")
    z = np.load(sdir / "probe_tokens_novel.npz")
    mean, scale, coef, icpt = (z[f"{k}_{LAYER}"] for k in ("mean", "scale", "coef", "intercept"))
    order = np.argsort(coef)  # the ASCII field lays dimensions out by probe weight: 'in the speech' on top
    plane, project = probe_plane(x, tok, li, mean, scale, coef, icpt, tau)
    lt = math.log(tau / (1 - tau))
    rows_of = defaultdict(list)
    for i, t in enumerate(tok):
        if t["run"] == BASE and t["condition"] == "baseline":
            rows_of[(t["sample_id"], t["speaker"])].append(i)

    def verdict(o):
        return {j: o["v"][j]["s"] == POS for j in o["v"]}

    def line_round(key, b_idx, ev):
        a, b_row, g_row = parts[key], base[key], rag[key]
        bl = [a["baseline"][i] for i in sorted(a["baseline"])]
        rl = [a["rag"][i] for i in sorted(a["rag"])]
        gen = b_row["generation"]
        text = bl[b_idx]["text"]
        start = gen.find(text)
        assert start >= 0, (key, text)
        line = clean(text)
        off = start + text.find(line)
        idx = [i for i in rows_of[key] if off <= tok[i]["char"] < off + len(line)]
        vec = np.asarray(x[idx, li, :], dtype=np.float32)
        zc = (vec - mean) / scale
        contrib = zc * coef
        p = 1 / (1 + np.exp(-(contrib.sum(1) + icpt)))
        xy = project(zc)
        words = [{"c": tok[i]["char"] - off, "w": tok[i]["word"], "p": round(float(pp), 4), "novel": tok[i]["novel"],
                  "content": tok[i]["is_content"] and not tok[i]["framing"],
                  "xy": [round(float(a), 2) for a in q2]} for i, pp, q2 in zip(idx, p, xy)]
        for w in words:
            assert line[w["c"]:w["c"] + len(w["w"])] == w["w"], (key, w)
        q = np.clip(np.round(contrib[:, order] / FIELD_SCALE * 127), -127, 127).astype(np.int8)
        fem = any(o["hearing_id"] == key[0] and o["speaker"] == key[1] and o.get("gender") == "F" for o in OMV2)
        date = re.search(r"\d{2}/\d{2}/\d{4}", lds[key[0]]["materia"]).group(0)
        rnd = {"h": key[0], "speaker": b_row["nome"], "role": b_row["cargo"], "line": b_idx, "n_lines": len(bl),
               "fem": fem, "date": date, "name_tokens": [fold(t) for t in (b_row["nome"] + " " + key[1]).split() if len(t) > 2],
               "prior": [clean(o["text"]) for o in bl[:b_idx]], "text": line, "verdict": verdict(bl[b_idx]),
               "comment": bl[b_idx]["v"]["sonnet"]["c"], "words": words,
               "field": base64.b64encode(q.tobytes()).decode(), "dims": int(q.shape[1]),
               "windows": b_row["chunks"], "topic": lds[key[0]]["metadados"]["assunto"], "trigger": None}
        if ev is not None:
            r_idx = next((i for i, o in enumerate(rl) if clean(o["text"]) == clean(ev["after"])), None)
            partial = clean(ev["before"])
            assert line.startswith(partial), (key, partial, line)
            # the word the model was about to write = the first stored reading at or after the stop. Its plane
            # position takes the reading that fired (logged live) on axis 1 and the re-read's axis 2.
            nxt = next(w for w in words if w["c"] >= len(partial))
            rnd["trigger"] = {"c": len(partial), "p": ev["p"], "windows": ev["windows"],
                              "regen": clean(rl[r_idx]["text"]), "regen_verdict": verdict(rl[r_idx]),
                              "regen_comment": rl[r_idx]["v"]["sonnet"]["c"],
                              "next": re.sub(r"[.,;:!?]+$", "", nxt["w"]), "next_novel": bool(nxt["novel"]),
                              "here": [round(math.log(ev["p"] / (1 - ev["p"])) - lt, 2), nxt["xy"][1]]}
            # the trigger's p is the value logged live; also count the re-read readings on content words
            # before it that already cross the threshold
            early = [w for w in words if w["c"] < len(partial) and w["p"] >= tau and w["content"]]
            rnd["early_crossings"] = len(early)
        else:
            rnd["early_crossings"] = sum(w["p"] >= tau and w["content"] for w in words)
        return rnd

    def events_of(key):
        a, g_row = parts[key], rag[key]
        bl = [a["baseline"][i] for i in sorted(a["baseline"])]
        out = []
        for e in g_row.get("rag_events", []):
            b_idx = next((i for i, o in enumerate(bl) if clean(o["text"]).startswith(clean(e["before"]))), None)
            if b_idx is not None:
                out.append((b_idx, e))
        return out

    def find_key(h, speaker):
        return next(k for k in parts if k[0] == h and (k[1] == speaker or name_match(k[1], speaker)))

    rounds = []
    for g in GAME:
        if g["kind"] == "neutral":
            # an alarm on a line both raters call supported, whose rewrite both also call supported: among the
            # featured hearings other than 38 and 91, the 90-190-character line with the highest trigger p
            best = None
            for key in sorted(parts):
                if key not in rag or key[0] not in FEATURED or key[0] in (38, 91):
                    continue
                for b_idx, e in events_of(key):
                    a = parts[key]
                    bl = [a["baseline"][i] for i in sorted(a["baseline"])]
                    rl = {clean(a["rag"][i]["text"]): a["rag"][i] for i in a["rag"]}
                    ro = rl.get(clean(e["after"]))
                    if ro is None or not all(verdict(bl[b_idx]).values()) or not all(verdict(ro).values()):
                        continue
                    L = len(clean(bl[b_idx]["text"]))
                    if 90 <= L <= 190 and (best is None or e["p"] > best[2]["p"]):
                        best = (key, b_idx, e)
            key, b_idx, e = best
            rnd = line_round(key, b_idx, e)
        else:
            key = find_key(g["h"], g["speaker"])
            evs = events_of(key)
            if g["kind"] == "calm":
                rnd = line_round(key, g["line"], None)
            else:
                cand = [(b, e) for b, e in evs if g["line"] is None or b == g["line"]]
                if g["kind"] == "fix":
                    a = parts[key]
                    rl = {clean(a["rag"][i]["text"]): a["rag"][i] for i in a["rag"]}
                    cand = [(b, e) for b, e in cand if a["baseline"][sorted(a["baseline"])[b]]["v"]["sonnet"]["s"] == NEG
                            and rl.get(clean(e["after"]), {"v": {"sonnet": {"s": None}}})["v"]["sonnet"]["s"] == POS]
                if g["kind"] == "broke":
                    a = parts[key]
                    rl = {clean(a["rag"][i]["text"]): a["rag"][i] for i in a["rag"]}
                    cand = [(b, e) for b, e in cand if a["baseline"][sorted(a["baseline"])[b]]["v"]["sonnet"]["s"] == POS
                            and rl.get(clean(e["after"]), {"v": {"sonnet": {"s": None}}})["v"]["sonnet"]["s"] == NEG]
                b_idx, e = cand[0]
                rnd = line_round(key, b_idx, e)
        rnd["kind"], rnd["why"] = g["kind"], g["why"]
        rounds.append(rnd)
        t = rnd["trigger"]
        print(f"game {g['kind']:7s} h{rnd['h']} {rnd['speaker']}: line {rnd['line']}, {len(rnd['words'])} readings, "
              f"early crossings {rnd['early_crossings']}, trigger {t and round(t['p'], 4)}, verdict {rnd['verdict']}"
              + (f" -> regen {t['regen_verdict']}" if t else ""))
    json.dump({"tau": tau, "layer": LAYER, "model": MODEL, "field_scale": FIELD_SCALE, "plane": plane, "rounds": rounds},
              open(OUT / "game.json", "w"), ensure_ascii=False, separators=(",", ":"))


FIELD_SCALE = 0.5   # |coef * z| mapped to the full int8 range; a few dimensions saturate, which reads as texture

if __name__ == "__main__":
    main()
