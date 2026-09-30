"""The PublicHearingBR detector recipe on an external benchmark: RAGTruth summarisation.

Linear probe on the reader model's sentence-mean residual activations (one teacher-forced
pass over article and response; the model is not fine-tuned), reported like RAGTruth's own
detector tables (Niu et al., ACL 2024, Table 5 response level, Table 6 span level,
Summarization column). Protocol:
  * train on the official train split (all six generators), content-bearing sentences,
    sentence labels from extract_ragtruth_sentence_mean.py;
  * response score = max over its sentences; response label = at least one human span of
    any type (the benchmark's response-level definition);
  * layer by out-of-fold response ROC-AUC on train (5-fold GroupKFold by source article),
    threshold by max F1 on the same out-of-fold scores; the official test split is scored
    once, and the other layers' test AUCs are printed but never used to choose;
  * non-neural baseline: share of a sentence's content words absent from the article,
    max over the response, threshold chosen on train;
  * span level: characters of flagged sentences vs characters of human spans, micro P/R/F1
    (sentence granularity caps precision); 95% CIs from 2,000 bootstrap draws over test
    articles, an article's six responses resampled together.

Usage: uv run python scripts/ragtruth_external_eval.py <reader_tag>   (a key of READERS; CPU)
       uv run python scripts/ragtruth_external_eval.py --report       (writes data/ragtruth_sentence/ragtruth_external.md)
The first form writes data/ragtruth_sentence/<reader_tag>/external_eval.json and
external_eval_test_responses.csv; OCARANDU_RT_EVAL_LAYERS (comma list) limits the layers tried.
"""
import json
import os
import re
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ocarandu.data.ragtruth import load_summarization_subset
from ocarandu.data.sentence_labels import segment_sentences

ROOT = Path(__file__).resolve().parents[1]
RT_ROOT = ROOT / "data" / "ragtruth_sentence"
DOC = RT_ROOT / "ragtruth_external.md"
SEED = 0
N_BOOT = 2000
READERS = {"llama_2_7b_chat_hf": ("Llama-2-7B-chat", "llama-2-7b-chat"),
           "llama_31_8b_instruct": ("Llama-3.1-8B-Instruct", None),
           "qwen25_7b_instruct": ("Qwen2.5-7B-Instruct", None)}

# Published rows, Niu et al., RAGTruth (ACL 2024, arXiv:2401.00396), Table 5 (response) and Table 6 (span),
# Summarization column: (detector, P, R, F1).
PUBLISHED_RESPONSE = [("Prompt, GPT-3.5-turbo", 23.4, 89.2, 37.1), ("Prompt, GPT-4-turbo", 31.5, 97.6, 47.6),
                      ("SelfCheckGPT, GPT-3.5-turbo", 31.1, 56.5, 40.1), ("LMvLM, GPT-4-turbo", 23.3, 81.9, 36.2),
                      ("Fine-tuned Llama-2-13B", 64.0, 54.9, 59.1)]
PUBLISHED_SPAN = [("Prompt, GPT-3.5-turbo", 6.1, 33.7, 10.3), ("Prompt, GPT-4-turbo", 14.7, 65.4, 24.1),
                  ("Fine-tuned Llama-2-13B", 52.4, 30.8, 38.8)]

_TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _norm(t):
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def content_words(text):
    return [_norm(t) for t in _TOK.findall(text.lower()) if t not in ENGLISH_STOP_WORDS and (len(t) > 2 or t.isdigit())]


def f1_at(y, s, tau):
    pred = s >= tau
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def best_threshold(y, s):
    cands = np.unique(s)
    f1s = [f1_at(y, s, t)[2] for t in cands]
    return float(cands[int(np.argmax(f1s))])


def to_response(sent_df, col):
    return sent_df.groupby("response_id", sort=False)[col].max()


def bootstrap(resp, cols, tau, rng):
    """resp: DataFrame indexed by response with y, source_id and score columns. Returns CI dicts."""
    by_src = {s: g.index.values for s, g in resp.groupby("source_id")}
    srcs = np.array(list(by_src))
    out = {c: {"auc": [], "f1": []} for c in cols}
    out["delta_auc"] = []
    for _ in range(N_BOOT):
        pick = np.concatenate([by_src[s] for s in rng.choice(srcs, len(srcs), replace=True)])
        b = resp.loc[pick]
        if b.y.nunique() < 2:
            continue
        aucs = {}
        for c in cols:
            aucs[c] = roc_auc_score(b.y, b[c])
            out[c]["auc"].append(aucs[c]); out[c]["f1"].append(f1_at(b.y.values, b[c].values, tau[c])[2])
        out["delta_auc"].append(aucs[cols[0]] - aucs[cols[1]])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res = {c: {"auc_ci": ci(out[c]["auc"]), "f1_ci": ci(out[c]["f1"])} for c in cols}
    d = np.array(out["delta_auc"])
    res["delta_auc_ci"] = ci(d)
    res["delta_auc_p_le0"] = float((d <= 0).mean())
    return res


def char_prf(test_resp_sents, examples_by_id, score_col, tau):
    tp = pp = gp = 0
    for rid, g in test_resp_sents.groupby("response_id"):
        ex = examples_by_id[rid]
        sents = segment_sentences(ex.response)
        gold = np.zeros(len(ex.response), bool)
        for sp in ex.spans:
            gold[sp.start:sp.end] = True
        pred = np.zeros(len(ex.response), bool)
        for si, sc in zip(g.sent_i.values, g[score_col].values):
            if sc >= tau:
                pred[sents[si].start:sents[si].end] = True
        tp += int((gold & pred).sum()); pp += int(pred.sum()); gp += int(gold.sum())
    p = tp / pp if pp else 0.0
    r = tp / gp if gp else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(tag):
    rt = RT_ROOT / tag
    meta = json.load(open(rt / "meta.json"))
    idx = pd.read_json(rt / "index.jsonl", lines=True, dtype={"response_id": str, "source_id": str})
    idx = idx[(idx.row < meta["n_rows"]) & (idx.filler == "content_bearing")].reset_index(drop=True)
    acts = np.load(rt / "sent_mean.npy", mmap_mode="r")
    examples = load_summarization_subset()
    ex_by_id = {e.response_id: e for e in examples}

    # response-level truth and the lexical baseline, both from the raw benchmark
    src_vocab, resp_sents = {}, {}
    lex = np.zeros(len(idx))
    for k, (rid, si) in enumerate(zip(idx.response_id.values, idx.sent_i.values)):
        ex = ex_by_id[rid]
        if ex.source_id not in src_vocab:
            src_vocab[ex.source_id] = set(content_words(ex.source_info))
        if rid not in resp_sents:
            resp_sents[rid] = segment_sentences(ex.response)
        cw = content_words(resp_sents[rid][si].text)
        lex[k] = (sum(w not in src_vocab[ex.source_id] for w in cw) / len(cw)) if cw else 0.0
    idx["lexical"] = lex
    resp = (idx.groupby("response_id", sort=False)
               .agg(source_id=("source_id", "first"), split=("split", "first"), gen_model=("gen_model", "first")))
    resp["y"] = [int(len(ex_by_id[r].spans) > 0) for r in resp.index]

    tr_s = idx[idx.split == "train"].reset_index(drop=True)
    te_s = idx[idx.split == "test"].reset_index(drop=True)
    tr_r = resp[resp.split == "train"]
    te_r = resp[resp.split == "test"].copy()
    folds = list(GroupKFold(n_splits=5).split(tr_s, groups=tr_s.source_id.values))

    layer_rows, fits = [], {}
    only = {int(x) for x in os.environ.get("OCARANDU_RT_EVAL_LAYERS", "").split(",") if x}  # for smoke tests
    for li, layer in enumerate(meta["layers"]):
        if only and layer not in only:
            continue
        x_tr = np.asarray(acts[tr_s.row.values, li, :], dtype=np.float32)
        oof = np.zeros(len(tr_s))
        for fi, (a, b) in enumerate(folds):
            sc = StandardScaler().fit(x_tr[a])
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED).fit(sc.transform(x_tr[a]), tr_s.label.values[a])
            oof[b] = clf.predict_proba(sc.transform(x_tr[b]))[:, 1]
        tr_s[f"oof{layer}"] = oof
        oof_r = to_response(tr_s, f"oof{layer}").loc[tr_r.index]
        sc = StandardScaler().fit(x_tr)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED).fit(sc.transform(x_tr), tr_s.label.values)
        te_s[f"p{layer}"] = clf.predict_proba(sc.transform(np.asarray(acts[te_s.row.values, li, :], dtype=np.float32)))[:, 1]
        te_r[f"p{layer}"] = to_response(te_s, f"p{layer}").loc[te_r.index]
        layer_rows.append({"layer": layer, "train_oof_auc": float(roc_auc_score(tr_r.y, oof_r)),
                           "test_auc_not_used": float(roc_auc_score(te_r.y, te_r[f"p{layer}"]))})
        print(layer_rows[-1], file=sys.stderr, flush=True)

    best = max(layer_rows, key=lambda r: r["train_oof_auc"])["layer"]
    tau = {"probe": best_threshold(tr_r.y.values, to_response(tr_s, f"oof{best}").loc[tr_r.index].values),
           "lexical": best_threshold(tr_r.y.values, to_response(tr_s, "lexical").loc[tr_r.index].values)}
    te_r["probe"] = te_r[f"p{best}"]; te_s["probe"] = te_s[f"p{best}"]
    te_r["lexical"] = to_response(te_s, "lexical").loc[te_r.index]

    def block(frame):
        out = {"n": int(len(frame)), "n_pos": int(frame.y.sum())}
        for c in ("probe", "lexical"):
            p, r, f = f1_at(frame.y.values, frame[c].values, tau[c])
            out[c] = {"auc": float(roc_auc_score(frame.y, frame[c])) if frame.y.nunique() > 1 else None,
                      "pr_auc": float(average_precision_score(frame.y, frame[c])) if frame.y.nunique() > 1 else None,
                      "p": p, "r": r, "f1": f}
        return out

    res = {"tag": tag, "reader": READERS[tag][0], "layers": layer_rows, "chosen_layer": best, "tau": tau,
           "test": block(te_r), "by_generator": {g: block(f) for g, f in te_r.groupby("gen_model")},
           "boot": bootstrap(te_r, ["probe", "lexical"], tau, np.random.default_rng(SEED)),
           "span": {c: dict(zip(("p", "r", "f1"), char_prf(te_s, ex_by_id, c, tau[c]))) for c in ("probe", "lexical")},
           "n_train_sent": int(len(tr_s)), "n_train_pos": int(tr_s.label.sum())}
    self_gen = READERS[tag][1]
    if self_gen:
        res["self_generated"] = block(te_r[te_r.gen_model == self_gen])
    (rt / "external_eval.json").write_text(json.dumps(res, indent=1))
    te_r.to_csv(rt / "external_eval_test_responses.csv")
    print(json.dumps({k: res[k] for k in ("chosen_layer", "tau", "test", "span")}, indent=1))


def report():
    L = ["# RAGTruth (external benchmark): the detector on English news summarisation, benchmark format", "",
         "Generated by `scripts/ragtruth_external_eval.py --report`. Official RAGTruth split, Summary task: "
         "train 793 articles x 6 generators, TEST 150 articles x 6 generators = 900 responses, "
         "22.7% with at least one human hallucination span. Probe = logistic regression on the sentence-mean "
         "residual stream of the READER model (one teacher-forced pass over the article + response; the model "
         "is never fine-tuned). Layer by out-of-fold train AUC, threshold by out-of-fold train F1; test read once. "
         "95% CIs by bootstrap over the 150 test articles.", ""]
    runs = [json.load(open(RT_ROOT / t / "external_eval.json")) for t in READERS if (RT_ROOT / t / "external_eval.json").exists()]
    L += ["## Response level (RAGTruth Table 5 format, Summarization column)", "",
          "| detector | P | R | **F1** | ROC-AUC | PR-AUC |", "|---|---:|---:|---:|---:|---:|"]
    for name, p, r, f in PUBLISHED_RESPONSE:
        L.append(f"| {name} (published) | {p:.1f} | {r:.1f} | {f:.1f} | – | – |")
    for k, R in enumerate(runs):
        t, b = R["test"], R["boot"]
        # the lexical baseline does not depend on the reader: print it once
        for c, lab in ((("lexical", "novel-content-word share (non-neural)"),) if k == 0 else ()) + (("probe", f"**linear probe, {R['reader']} L{R['chosen_layer']}**"),):
            m = t[c]
            L.append(f"| {lab} | {100*m['p']:.1f} | {100*m['r']:.1f} | **{100*m['f1']:.1f}** "
                     f"[{100*b[c]['f1_ci'][0]:.1f}, {100*b[c]['f1_ci'][1]:.1f}] | {m['auc']:.3f} "
                     f"[{b[c]['auc_ci'][0]:.3f}, {b[c]['auc_ci'][1]:.3f}] | {m['pr_auc']:.3f} |")
    L += ["", "Probe minus lexical ROC-AUC on the same 900 responses (bootstrap by article):", ""]
    for R in runs:
        b = R["boot"]
        d = R["test"]["probe"]["auc"] - R["test"]["lexical"]["auc"]
        L.append(f"- {R['reader']}: {d:+.3f} [{b['delta_auc_ci'][0]:+.3f}, {b['delta_auc_ci'][1]:+.3f}], "
                 f"share of draws <= 0: {b['delta_auc_p_le0']:.4f}")
    L += ["", "## Span level (RAGTruth Table 6 format, Summarization column; character overlap, micro)", "",
          "| detector | P | R | F1 |", "|---|---:|---:|---:|"]
    for name, p, r, f in PUBLISHED_SPAN:
        L.append(f"| {name} (published) | {p:.1f} | {r:.1f} | {f:.1f} |")
    for k, R in enumerate(runs):
        for c, lab in ((("lexical", "novel-content-word share"),) if k == 0 else ()) + (("probe", f"linear probe, {R['reader']} L{R['chosen_layer']}"),):
            s = R["span"][c]
            L.append(f"| {lab} (flagged sentences) | {100*s['p']:.1f} | {100*s['r']:.1f} | {100*s['f1']:.1f} |")
    L += ["", "## By generator model (probe / lexical ROC-AUC, 150 responses each)", "",
          "| reader | " + " | ".join(sorted(runs[0]["by_generator"])) + " |", "|---|" + "---:|" * len(runs[0]["by_generator"])]
    for R in runs:
        L.append(f"| {R['reader']} | " + " | ".join(
            f"{R['by_generator'][g]['probe']['auc']:.3f} / {R['by_generator'][g]['lexical']['auc']:.3f}" for g in sorted(R["by_generator"])) + " |")
    for R in runs:
        if "self_generated" in R:
            s = R["self_generated"]
            L += ["", f"Self-generated subset ({R['reader']} reading its own RAGTruth responses, {s['n']} responses, "
                      f"{s['n_pos']} hallucinated): probe ROC-AUC {s['probe']['auc']:.3f}, F1 {100*s['probe']['f1']:.1f}; "
                      f"lexical {s['lexical']['auc']:.3f} / {100*s['lexical']['f1']:.1f}. Closest to ReDeEP's setting "
                      "(ICLR 2025, Table 1: SAPLMA 0.704, ITI 0.716, ReDeEP 0.746 AUC on Llama-2-7B) but NOT comparable: "
                      "theirs pools QA, data-to-text and summarisation."]
    L += ["", "## Layer selection (train out-of-fold AUC decides; test column shown only for transparency)", ""]
    for R in runs:
        L += [f"**{R['reader']}** (chosen L{R['chosen_layer']}; {R['n_train_sent']} train sentences, {R['n_train_pos']} positive):", "",
              "| layer | train OOF AUC | test AUC (not used) |", "|---:|---:|---:|"]
        L += [f"| {r['layer']} | {r['train_oof_auc']:.3f} | {r['test_auc_not_used']:.3f} |" for r in R["layers"]] + [""]
    L += ["## Caveats", "",
          "- Summarisation task only (our activations cover RAGTruth's Summary subset); the published rows are the "
          "same task and split, the pooled 'Overall' column is not comparable.",
          "- Response label counts every human span, including `implicit_true` ones (true but unsupported by the source).",
          "- The published fine-tuned Llama-2-13B was trained on all three tasks' train data and outputs spans; our "
          "probe sees the Summary train split only and flags whole sentences.",
          "- A handful of very long articles were cut at 6,000 tokens at extraction; sentences past the cut have no vector "
          "and do not contribute to the response score (the response label still comes from all its spans)."]
    DOC.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "--report":
        report()
    else:
        evaluate(sys.argv[1])
