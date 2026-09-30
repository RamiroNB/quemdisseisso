"""Cross-lingual zero-shot transfer of the hallucination probe, RAGTruth (en) -> PublicHearingBR (pt-BR).

Per layer, a balanced logistic probe is trained on RAGTruth sentence activations
(official train split, content-bearing sentences; extract_ragtruth_sentence_mean.py)
and applied without adaptation to the PublicHearingBR opinion activations of the same
model (extract_publichearingbr_activations.py), scored against the dataset's human
`verificacao_manual` labels. Each probe's RAGTruth test-split score is reported too,
so a failed transfer can be told apart from a weak probe.

Usage: uv run python scripts/crosslingual_probe.py [model_tag] [phbr_act_dir]
  model_tag     dir under data/ragtruth_sentence/ (default llama_31_8b_instruct)
  phbr_act_dir  dir under data/publichearingbr/ (default activations_<model_tag>)
Writes data/publichearingbr/crosslingual_probe_<model_tag>.md. CPU only.
"""
import json
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TAG = sys.argv[1] if len(sys.argv) > 1 else "llama_31_8b_instruct"
RT = ROOT / "data" / "ragtruth_sentence" / TAG
PHBR = ROOT / "data" / "publichearingbr" / (sys.argv[2] if len(sys.argv) > 2 else f"activations_{TAG}")
OUT = ROOT / "data" / "publichearingbr" / f"crosslingual_probe_{TAG}.md"
SEED = 0


def main():
    meta = json.load(open(RT / "meta.json"))
    idx = pd.read_json(RT / "index.jsonl", lines=True)
    idx = idx[idx.row < meta["n_rows"]]
    acts = np.load(RT / "sent_mean.npy", mmap_mode="r")
    keep = idx.filler == "content_bearing"
    tr = idx[keep & (idx.split == "train")]
    te = idx[keep & (idx.split == "test")]

    p_idx = pd.read_json(PHBR / "index.jsonl", lines=True)
    p_acts = np.load(PHBR / "opinion_mean.npy", mmap_mode="r")
    y_p = p_idx.label.values.astype(int)

    lines = [f"# Cross-lingual zero-shot transfer: RAGTruth (en) -> PublicHearingBR (pt-BR), {TAG}", "",
             f"Train: {len(tr)} content-bearing RAGTruth sentences ({int(tr.label.sum())} hallucinated), official train split. "
             f"RAGTruth test: {len(te)} sentences. Target: {len(p_idx)} PublicHearingBR opinions "
             f"({int(y_p.sum())} = {y_p.mean():.1%} hallucinated), human labels, no adaptation.", "",
             "| layer | RAGTruth test ROC / PR | **PHBR zero-shot ROC / PR** |", "|---|---:|---:|"]
    best = None
    for li, layer in enumerate(meta["layers"]):
        if layer >= p_acts.shape[1]:
            continue
        x_tr = np.asarray(acts[tr.row.values, li, :], dtype=np.float32)
        x_te = np.asarray(acts[te.row.values, li, :], dtype=np.float32)
        x_p = np.asarray(p_acts[:, layer, :], dtype=np.float32)
        sc = StandardScaler().fit(x_tr)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED).fit(sc.transform(x_tr), tr.label.values)
        s_te = clf.predict_proba(sc.transform(x_te))[:, 1]
        s_p = clf.predict_proba(sc.transform(x_p))[:, 1]
        r_te, a_te = roc_auc_score(te.label.values, s_te), average_precision_score(te.label.values, s_te)
        r_p, a_p = roc_auc_score(y_p, s_p), average_precision_score(y_p, s_p)
        lines.append(f"| {layer} | {r_te:.3f} / {a_te:.3f} | **{r_p:.3f} / {a_p:.3f}** |")
        if best is None or r_p > best[1]:
            best = (layer, r_p, a_p)
    lines += ["", f"Best transfer layer {best[0]}: ROC-AUC {best[1]:.3f}, PR-AUC {best[2]:.3f} "
                  f"(random PR-AUC baseline {y_p.mean():.3f})."]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
