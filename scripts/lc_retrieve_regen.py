"""Probe-triggered retrieve-and-regenerate on the per-participant summarisation task.

Greedy decoding with the plain prompt of longcontext_speaker_generation.py. When a
trigger fires on the current line, the OCARANDU_RAG_TOPK windows of the speaker's turns
that share the most content words with the line so far are added to the prompt, and
that line is regenerated from its start; decoding then continues with the original
prompt. This follows FLARE ("Active Retrieval Augmented Generation", EMNLP 2023), with a
residual-stream token probe (P(next content word is absent from the source), fitted by
lc_stream_stats.py) as the trigger and the speaker's own turns as the retrieval corpus.

Arms, through the same decode loop (at most one retrieval per line, and only once the
line has >= 2 content words that are not the speaker's name or a framing verb):
  rag:none     never retrieve (baseline on the same code path)
  rag:probe    retrieve when the probe's P(next content word novel) exceeds tau_p
tau_p is fitted on calibration participants disjoint from the evaluated ones (by default
the OCARANDU_RAG_N_CAL right after them) so that the probe fires on OCARANDU_RAG_LINE_RATE
of their lines. The realised rate on the evaluated participants is logged per row
(n_retrievals).

Usage: OCARANDU_DEVICE=cuda:0 OCARANDU_LC_FOLD=A OCARANDU_LC_SPLIT=all \\
       OCARANDU_LC_N=220 OCARANDU_LC_PER_HEARING=100 uv run python scripts/lc_retrieve_regen.py
Env: OCARANDU_MODEL_REPO, OCARANDU_GEN_LAYER (probe layer), OCARANDU_RAG_ARMS
     (none,probe), OCARANDU_RAG_N_EVAL (200), OCARANDU_RAG_N_CAL (20),
     OCARANDU_RAG_EVAL_FROM (0), OCARANDU_RAG_CAL_FROM (N_EVAL), OCARANDU_RAG_LINE_RATE (0.35),
     OCARANDU_RAG_TOPK (2), OCARANDU_RAG_PROBE (token-probe .npz, default
     data/publichearingbr/lc_stream/<model>/probe_tokens_novel.npz), OCARANDU_RAG_TAG,
     OCARANDU_LC_MAXNEW (300).
Writes data/publichearingbr/longcontext_generation/<model>_rag/{generations.jsonl,thresholds.json}.
Resumable; an existing thresholds.json fitted on other calibration participants is refused.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attribution_generation import content_words  # noqa: E402
from content_hallucination_generation import DEVICE, LAYER, MODEL_TAG, REPO_ID, ROOT, hearing_split  # noqa: E402
from lc_stream_extract import FRAMING  # noqa: E402
from longcontext_speaker_generation import (  # noqa: E402
    FOLD, LINE_PREFIX, PREAMBLE, PROMPTS, SPLIT, build_items, score_generation,
)

GEN = ROOT / "data" / "publichearingbr" / "longcontext_generation"
OUT_DIR = GEN / os.environ.get("OCARANDU_RAG_TAG", f"{MODEL_TAG}_rag")
ARMS = os.environ.get("OCARANDU_RAG_ARMS", "none,probe").split(",")
N_EVAL = int(os.environ.get("OCARANDU_RAG_N_EVAL", "200"))
N_CAL = int(os.environ.get("OCARANDU_RAG_N_CAL", "20"))
# EVAL_FROM / CAL_FROM let a later slice of fold A be evaluated with thresholds already fitted on an
# earlier calibration slice (e.g. EVAL_FROM=220, CAL_FROM=200), so calibration participants are never evaluated.
EVAL_FROM = int(os.environ.get("OCARANDU_RAG_EVAL_FROM", "0"))
CAL_FROM = int(os.environ.get("OCARANDU_RAG_CAL_FROM", str(N_EVAL)))
LINE_RATE = float(os.environ.get("OCARANDU_RAG_LINE_RATE", "0.35"))
TOPK = int(os.environ.get("OCARANDU_RAG_TOPK", "2"))
MAX_NEW = int(os.environ.get("OCARANDU_LC_MAXNEW", "300"))
LINE_MAX = 90  # tokens for one regenerated line
MIN_QUERY_WORDS = 2
PROBE_NPZ = os.environ.get("OCARANDU_RAG_PROBE", str(ROOT / "data" / "publichearingbr" / "lc_stream" / MODEL_TAG / "probe_tokens_novel.npz"))
NOTE = ("\n\nPara a próxima opinião, estes são os trechos da fala de {nome} mais relevantes. Escreva essa "
        "opinião de forma fiel a eles, sem inverter, exagerar ou acrescentar nada, mantendo o formato: uma "
        "única frase curta na terceira pessoa, sem citar a fala literalmente.\n\"{passage}\"")


def query_words(line, nome):
    return content_words(LINE_PREFIX.sub("", line)) - content_words(nome) - FRAMING


def top_windows(q, windows, k):
    scored = sorted(range(len(windows)), key=lambda i: (-len(q & content_words(windows[i])), i))[:k]
    return sorted(scored)  # chronological order in the note


def main():
    fit_h, eval_pool, dev_h, test_h = hearing_split()
    if FOLD != "A" or SPLIT != "all":
        raise SystemExit("run with OCARANDU_LC_FOLD=A OCARANDU_LC_SPLIT=all (the protocol participants)")
    tok = AutoTokenizer.from_pretrained(REPO_ID)
    items = build_items(tok, set(eval_pool))
    if len(items) < max(CAL_FROM + N_CAL, EVAL_FROM + 1):
        raise SystemExit(f"need at least {max(CAL_FROM + N_CAL, EVAL_FROM + 1)} items, got {len(items)}; set OCARANDU_LC_N")
    evals, cals = items[EVAL_FROM:EVAL_FROM + N_EVAL], items[CAL_FROM:CAL_FROM + N_CAL]
    ev_keys = {(it["sample_id"], it["speaker"]) for it in evals}
    if ev_keys & {(it["sample_id"], it["speaker"]) for it in cals}:
        raise SystemExit("evaluation and calibration participants overlap; refusing")
    if len(evals) < N_EVAL:
        print(f"note: only {len(evals)} evaluation participants available from index {EVAL_FROM}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(REPO_ID, dtype=torch.bfloat16, device_map=DEVICE).eval()
    z = np.load(PROBE_NPZ)
    pr = {k: torch.tensor(z[f"{k}_{LAYER}"], dtype=torch.float32, device=DEVICE) for k in ("mean", "scale", "coef")}
    pr_b = float(z[f"intercept_{LAYER}"])
    state = {"h": None}

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        state["h"] = h[0, -1].float()
        return output

    model.model.layers[LAYER].register_forward_hook(hook)
    gc_eos = getattr(model.generation_config, "eos_token_id", None)
    eos = {t for t in ([tok.eos_token_id] + (gc_eos if isinstance(gc_eos, list) else [gc_eos])) if t is not None}

    def encode(user, assistant=""):
        ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, return_tensors="pt")
        ids = ids["input_ids"] if not isinstance(ids, torch.Tensor) else ids
        if assistant:
            ids = torch.cat([ids, tok(assistant, add_special_tokens=False, return_tensors="pt")["input_ids"]], dim=1)
        return ids.to(DEVICE)

    @torch.no_grad()
    def forward(ids, past):
        out = model(input_ids=ids, past_key_values=past, use_cache=True)
        return out.logits[0, -1].float(), out.past_key_values

    def probe_p():
        zc = (state["h"] - pr["mean"]) / pr["scale"]
        return float(torch.sigmoid(zc @ pr["coef"] + pr_b))

    def entropy(logits):
        lp = torch.log_softmax(logits, -1)
        return float(-(lp.exp() * lp).sum())

    @torch.no_grad()
    def gen_line(user, done):
        """Greedy decode of one line after `done`, stopping at newline / eos / LINE_MAX tokens."""
        logits, past = forward(encode(user, done), None)
        seg = []
        while len(seg) < LINE_MAX:
            t = int(logits.argmax())
            if t in eos:
                return tok.decode(seg), len(seg), True
            seg.append(t)
            s = tok.decode(seg)
            if "\n" in s:
                return s[: s.index("\n") + 1], len(seg), False
            logits, past = forward(torch.tensor([[t]], device=DEVICE), past)
        return tok.decode(seg) + "\n", len(seg), False

    def run(it, arm, thr, record=None):
        user = PROMPTS["plain"].format(nome=it["nome"], cargo=it["cargo"], falas=it["falas"])
        text, n_tok, done_lines, events, finished = "", 0, set(), [], False
        while n_tok < MAX_NEW and not finished:
            logits, past = forward(encode(user, text), None)
            base, seg = text, []
            while n_tok < MAX_NEW:
                cur = base + tok.decode(seg)
                line = cur.split("\n")[-1]
                li = cur.count("\n")
                q = query_words(line, it["nome"])
                eligible = (li not in done_lines and len(q) >= MIN_QUERY_WORDS
                            and not PREAMBLE.match(LINE_PREFIX.sub("", line).strip()))
                if eligible:
                    p, ent = probe_p(), entropy(logits)
                    if record is not None:
                        record.setdefault(li, []).append((p, ent))
                    if arm == "probe" and p > thr["tau_p"]:
                        wins = top_windows(q, it["windows"], TOPK)
                        passage = " [...] ".join(it["windows"][i] for i in wins)
                        head = cur[: len(cur) - len(line)]
                        new, used, fin = gen_line(user + NOTE.format(nome=it["nome"], passage=passage), head)
                        events.append(dict(line=li, words_written=len(line.split()), p=round(p, 4), ent=round(ent, 3),
                                           windows=wins, before=line, after=new.strip()))
                        text, n_tok, finished = head + new, n_tok + used, fin
                        done_lines.add(li)
                        break
                t = int(logits.argmax())
                if t in eos:
                    text, finished = cur, True
                    break
                seg.append(t)
                n_tok += 1
                logits, past = forward(torch.tensor([[t]], device=DEVICE), past)
            else:
                text = base + tok.decode(seg)
        return text.strip(), events

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    thr_path = OUT_DIR / "thresholds.json"
    if thr_path.exists():
        thr = json.loads(thr_path.read_text())
        # reused thresholds must come from these calibration participants
        if [list(x) for x in thr.get("cal_participants", [])] != [[c["sample_id"], c["speaker"]] for c in cals]:
            raise SystemExit("thresholds.json was fit on different calibration participants; refusing")
    else:
        # calibration on participants disjoint from the evaluated ones: per eligible line, the max probe p;
        # tau_p puts LINE_RATE of the lines above it
        mx_p, lens = [], []
        for it in cals:
            rec = {}
            run(it, "none", None, record=rec)
            for li, v in rec.items():
                mx_p.append(max(a for a, _ in v)); lens.append(len(v))
        L = float(np.mean(lens))
        thr = dict(tau_p=float(np.quantile(mx_p, 1 - LINE_RATE)), line_rate=LINE_RATE,
                   n_cal_participants=len(cals), n_cal_lines=len(mx_p), mean_eligible_positions=L,
                   cal_participants=[[c["sample_id"], c["speaker"]] for c in cals])
        thr_path.write_text(json.dumps(thr, indent=1))
    print(f"thresholds: {thr}", file=sys.stderr, flush=True)

    out = OUT_DIR / "generations.jsonl"
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["condition"], r["sample_id"], r["speaker"]))
    f = open(out, "a", encoding="utf-8")
    for arm in ARMS:
        cond = f"rag:{arm}"
        for i, it in enumerate(evals):
            if (cond, it["sample_id"], it["speaker"]) in done:
                continue
            gen, events = run(it, arm, thr)
            row = {"condition": cond, "direction": "rag", "alpha": 0.0, "sample_id": it["sample_id"],
                   "speaker": it["speaker"], "nome": it["nome"], "cargo": it["cargo"],
                   "floor_chars": it["floor_chars"], "ctx_tokens": it["ctx_tokens"],
                   "opinion": " ".join(it["gold"]), "gold": it["gold"], "true_label": 0, "chunks": it["windows"],
                   "generation": gen, "n_chars": len(gen), "prompt_kind": "plain",
                   "n_retrievals": len(events), "rag_events": events,
                   **score_generation(gen, it["windows"], it["gold"])}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"  [{cond}] {i + 1}/{len(evals)}", file=sys.stderr, flush=True)
        rows = [json.loads(l) for l in open(out, encoding="utf-8")]
        rows = [r for r in rows if r["condition"] == cond]
        lines = sum(max(1, r["n_opinions"]) for r in rows)
        print(f"{cond:12s} coverage={np.mean([r['coverage'] or 0 for r in rows]):.3f} "
              f"ungrounded={np.mean([r['low_share'] or 0 for r in rows]):.3f} "
              f"retrievals/line={sum(r['n_retrievals'] for r in rows) / lines:.3f} "
              f"n_op={np.mean([r['n_opinions'] for r in rows]):.2f}", file=sys.stderr, flush=True)
    print(f"wrote {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
