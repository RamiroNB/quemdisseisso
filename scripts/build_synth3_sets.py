"""Build blind annotation sets that compare generation arms on the same participants.

An arm is a (run tag, condition) of a longcontext_generation run, given with --arms.
Design:

  * a block is one (participant, arm): the whole generated summary (all parsed
    opinions) plus all of the participant's 100-word windows, exactly the text
    the model was given. Participants missing an arm, or whose arms were given
    different text, are dropped, so the evidence shown is identical across arms.
  * arms are balanced inside every set and no participant appears twice in a
    set, so a rater never sees two arms of the same person and rater strictness
    cannot load onto one arm.
  * `--replicate` participants have each of their blocks placed a second time,
    in another set, for inter-rater agreement.
  * `--calib` items per set are dataset (GPT-4) opinions, drawn from a pool that
    is half hallucinated and half not, shown with their own four `chunks_proximos`
    (the evidence the benchmark's human annotator saw); that human label is kept
    in the key.

Raters are not told that arms exist; they judge support in the evidence only, and
form problems have a separate field. The arms can still differ in surface form
(e.g. how often a line opens with the speaker's name), which a rater who sees a
whole summary may notice.

Usage: uv run python scripts/build_synth3_sets.py --arms LABEL=RUN_TAG:CONDITION ... --name NAME
         [--n-participants N] [--blocks-per-set 10] [--calib 4] [--replicate 20] [--seed 20260922]
Output: data/publichearingbr/annotation/<name>/
  sets/set-NN.md        blind set for one rater
  sets/set-NN.ids.txt   the item ids of that set, in order
  raw/                  where the raters write set-NN.jsonl
and, in a sibling directory the raters are never pointed at,
  <name>_key/key.csv    id -> set, kind (pipeline|calib), arm, replicate, participant,
                        opinion, referee containment and flag (< 0.3), dataset human label
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from attribution_generation import containment  # noqa: E402  (the runs' own referee)

from ocarandu.data.publichearingbr import load_nli_opinions  # noqa: E402

GEN = ROOT / "data" / "publichearingbr" / "longcontext_generation"
ANNOT = ROOT / "data" / "publichearingbr" / "annotation"
HEADER = """# {name} — {n_items} itens para anotar

Você recebe {n_blocks} resumos automáticos. Cada resumo é de UMA pessoa que falou em uma
audiência pública da Câmara dos Deputados, e vem acompanhado dos TRECHOS LITERAIS da fala
dessa pessoa nessa audiência — recortes de 100 palavras, em ordem cronológica, que juntos
são tudo o que essa pessoa disse. Os recortes podem começar e terminar no meio de uma frase.

Para CADA item (cada linha numerada de cada resumo), responda:

1. `support`: a afirmação é sustentada pelos trechos dessa pessoa?
   - `sustentada`: o conteúdo da afirmação está nos trechos (paráfrase vale; não precisa ser
     o ponto principal da fala, nem estar em uma única passagem).
   - `nao_sustentada`: a afirmação acrescenta, contradiz, inverte, exagera ou distorce algo
     em relação aos trechos — inclusive números, nomes, instituições, datas, negações ou
     generalizações que não estejam ali.
   - `nao_da_para_dizer`: os trechos não permitem decidir.
2. `form_issue`: 1 se o item não é uma opinião/posição atribuível à pessoa (é texto
   procedimental, uma citação solta, uma linha vazia ou repetida), 0 caso contrário.
3. `comment`: uma frase curta só quando o caso for duvidoso; pode ficar vazio.

Julgue APENAS o apoio nos trechos. Não julgue estilo, tamanho, ordem, se a frase começa
pelo nome da pessoa, nem quão bem escrita ela é — nada disso conta como falta de apoio.
Não compare resumos entre si: cada item é julgado sozinho, contra os trechos dele.
"""


def load_blocks():
    rows = defaultdict(dict)
    windows = {}
    for arm, tag, cond in ARMS:
        path = GEN / tag / "generations.jsonl"
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["condition"] != cond or not r["opinions"]:
                continue
            key = (r["sample_id"], r["speaker"])
            rows[key][arm] = r
            windows.setdefault(key, r["chunks"])
    arms = [a for a, _, _ in ARMS]
    full, dropped = {}, 0
    for key, by_arm in rows.items():
        if not all(a in by_arm for a in arms):
            dropped += 1
            continue
        # all arms must have been given the same text, or the evidence is not comparable
        if any(by_arm[a]["chunks"] != windows[key] for a in arms):
            dropped += 1
            continue
        full[key] = by_arm
    print(f"{len(full)} participants with all three arms and identical evidence ({dropped} dropped)", file=sys.stderr)
    return full, windows


def calibration_pool(rng, n):
    ops = [o for o in load_nli_opinions() if len(o.context_chunks) == 4 and o.opinion.strip()]
    rng.shuffle(ops)
    # half hallucinated, half faithful, so rater accuracy is measurable on both classes
    pos = [o for o in ops if o.is_hallucination][: (n + 1) // 2]
    neg = [o for o in ops if not o.is_hallucination][: n // 2]
    out = pos + neg
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks-per-set", type=int, default=10)
    ap.add_argument("--calib", type=int, default=4)
    ap.add_argument("--replicate", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--arms", nargs="+", required=True, metavar="LABEL=RUN_TAG:CONDITION",
                    help="arms as (label in the key, run tag, condition in that run); e.g. "
                         "baseline=qwen3_4b_instruct_2507_lc_xfitA:baseline rag=qwen3_4b_instruct_2507_rag480:rag:probe "
                         "(the condition may itself contain ':')")
    ap.add_argument("--name", required=True, help="output directory name under data/publichearingbr/annotation/")
    ap.add_argument("--n-participants", type=int, default=0, help="subsample this many participants (0 = all)")
    a = ap.parse_args()
    global ARMS, OUT, KEY_DIR
    ARMS = []
    for spec in a.arms:
        label, rest = spec.split("=", 1)
        tag, cond = rest.split(":", 1)
        ARMS.append((label, tag, cond))
    name = a.name
    OUT, KEY_DIR = ANNOT / name, ANNOT / f"{name}_key"
    rng = random.Random(a.seed)
    blocks_by_part, windows = load_blocks()
    parts = sorted(blocks_by_part)
    rng.shuffle(parts)
    if a.n_participants:
        parts = parts[: a.n_participants]
        blocks_by_part = {k: v for k, v in blocks_by_part.items() if k in set(parts)}
    arms = [x for x, _, _ in ARMS]

    # ---- assignment: every (participant, arm) once; `--replicate` participants twice ----
    n_sets = (len(parts) * len(arms) + a.blocks_per_set - 1) // a.blocks_per_set
    reps = parts[: a.replicate]
    n_sets += (len(reps) * len(arms) + a.blocks_per_set - 1) // a.blocks_per_set
    sets = [[] for _ in range(n_sets)]
    used = [set() for _ in range(n_sets)]  # participants already in that set

    def place(part, arm, rep):
        order = sorted(range(n_sets), key=lambda i: (len(sets[i]), rng.random()))
        for i in order:
            if part in used[i] or len(sets[i]) >= a.blocks_per_set:
                continue
            if sum(1 for _, ar, _ in sets[i] if ar == arm) > (a.blocks_per_set // len(arms)):
                continue  # keep the arms balanced inside the set
            sets[i].append((part, arm, rep))
            used[i].add(part)
            return True
        for i in order:  # relax the balance constraint rather than drop a block
            if part not in used[i] and len(sets[i]) < a.blocks_per_set:
                sets[i].append((part, arm, rep))
                used[i].add(part)
                return True
        return False

    plan = [(p, arm, 0) for p in parts for arm in arms] + [(p, arm, 1) for p in reps for arm in arms]
    rng.shuffle(plan)
    unplaced = [(p, arm) for p, arm, rep in plan if not place(p, arm, rep)]
    if unplaced:
        print(f"WARNING: {len(unplaced)} blocks could not be placed", file=sys.stderr)

    calib = calibration_pool(rng, a.calib * n_sets)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sets").mkdir(exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    key_rows, n_items_total, item_no = [], 0, 0

    for si, blocks in enumerate(sets, 1):
        name = f"set-{si:02d}"
        rng.shuffle(blocks)
        # every block of this set becomes a numbered section; calibration blocks are interleaved
        sections = []
        for part, arm, rep in blocks:
            r = blocks_by_part[part][arm]
            wins = windows[part]
            items = []
            for j, op in enumerate(r["opinions"]):
                item_no += 1
                iid = f"s3-{item_no:05d}"
                items.append((iid, op))
                key_rows.append(dict(id=iid, set=name, kind="pipeline", arm=arm, replicate=rep,
                                     sample_id=part[0], speaker=part[1], nome=r["nome"], cargo=r["cargo"],
                                     opinion_idx=j, opinion=op,
                                     referee_containment=round(containment(op, wins), 4),
                                     referee_flag=int(containment(op, wins) < 0.3),
                                     human_label_dataset="", n_windows=len(wins)))
            sections.append(dict(nome=r["nome"], cargo=r["cargo"], windows=wins, items=items))
        for _ in range(a.calib):
            if not calib:
                break
            o = calib.pop()
            item_no += 1
            iid = f"s3-{item_no:05d}"
            key_rows.append(dict(id=iid, set=name, kind="calib", arm="", replicate=0,
                                 sample_id=o.sample_id, speaker=o.person_name, nome=o.person_name,
                                 cargo=o.person_role, opinion_idx=0, opinion=o.opinion,
                                 referee_containment=round(containment(o.opinion, list(o.context_chunks)), 4),
                                 referee_flag=int(containment(o.opinion, list(o.context_chunks)) < 0.3),
                                 human_label_dataset=int(o.is_hallucination), n_windows=4))
            sections.append(dict(nome=o.person_name, cargo=o.person_role,
                                 windows=list(o.context_chunks), items=[(iid, o.opinion)]))
        rng.shuffle(sections)

        n_items = sum(len(s["items"]) for s in sections)
        n_items_total += n_items
        lines = [HEADER.format(name=name, n_items=n_items, n_blocks=len(sections)), ""]
        for bi, s in enumerate(sections, 1):
            lines.append(f"## Resumo {bi} de {len(sections)}")
            lines.append(f"PESSOA: {s['nome']} — {s['cargo']}")
            lines.append("")
            lines.append("AFIRMAÇÕES A JULGAR:")
            for iid, op in s["items"]:
                lines.append(f"  [{iid}] {op}")
            lines.append("")
            lines.append(f"TRECHOS DA FALA DESSA PESSOA ({len(s['windows'])} recortes, em ordem cronológica):")
            for wi, w in enumerate(s["windows"], 1):
                lines.append(f"  trecho {wi}: {w}")
            lines.append("")
        (OUT / "sets" / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
        (OUT / "sets" / f"{name}.ids.txt").write_text(
            "\n".join(iid for s in sections for iid, _ in s["items"]) + "\n", encoding="utf-8")

    fields = ["id", "set", "kind", "arm", "replicate", "sample_id", "speaker", "nome", "cargo", "opinion_idx",
              "opinion", "referee_containment", "referee_flag", "human_label_dataset", "n_windows"]
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    with open(KEY_DIR / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(key_rows)
    by_arm = defaultdict(int)
    for r in key_rows:
        by_arm[r["arm"] or "calib"] += 1
    print(f"{n_sets} sets, {n_items_total} items ({dict(by_arm)}); "
          f"{len(reps)} participants double-annotated; key -> {KEY_DIR / 'key.csv'}", file=sys.stderr)
    print(f"sets -> {OUT / 'sets'}; raters write {OUT / 'raw'}/<set>.jsonl", file=sys.stderr)


if __name__ == "__main__":
    main()
