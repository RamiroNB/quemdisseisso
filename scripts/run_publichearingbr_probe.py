"""In-domain hallucination probe on PublicHearingBR (human labels), one per layer.

Reads {opinion_mean,name_last}.npy and index.jsonl written by
extract_publichearingbr_activations.py. For every layer: balanced logistic
regression, 5-fold GroupKFold by hearing (sample_id) so opinions of one hearing
never straddle train and test, and out-of-fold probabilities -> ROC-AUC, PR-AUC
and the best-F1 threshold (picked on the same out-of-fold scores, so optimistic).

Usage: uv run python scripts/run_publichearingbr_probe.py [--vector opinion_mean|name_last] [--C 1.0]
Env: OCARANDU_ACT_DIR, activation dir under data/publichearingbr/ (default "activations").
Writes data/publichearingbr/probe_layer_sweep_<vector><suffix>.csv and, for opinion_mean,
probe_scores<suffix>.jsonl (out-of-fold probability at the best layer by ROC-AUC, one row
per opinion; read by mitigation_headroom.py). <suffix> is "" for the default dir, else "_"
plus the dir name without "activations_". CPU only.
"""

import argparse
import json
import os
from pathlib import Path

# cap BLAS threads; must run before numpy/sklearn are imported
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ocarandu.probing.steering import best_f1_threshold

ROOT = Path(__file__).resolve().parents[1]
ACT_NAME = os.environ.get("OCARANDU_ACT_DIR", "activations")
ACT_DIR = ROOT / "data" / "publichearingbr" / ACT_NAME
SUFFIX = "" if ACT_NAME == "activations" else "_" + ACT_NAME.replace("activations_", "")
N_FOLDS = 5
SEED = 0


def load(vector="opinion_mean"):
    index = pd.read_json(ACT_DIR / "index.jsonl", lines=True)
    n_done = int((ACT_DIR / "done.txt").read_text())
    index = index[index.row < n_done].reset_index(drop=True)
    x = np.load(ACT_DIR / f"{vector}.npy", mmap_mode="r")[: len(index)]
    return index, x


def oof_probas(x_layer, y, groups, seed=SEED, C=1.0):
    proba = np.zeros(len(y))
    gkf = GroupKFold(n_splits=N_FOLDS)
    for tr, te in gkf.split(x_layer, y, groups):
        scaler = StandardScaler().fit(x_layer[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C, random_state=seed)
        clf.fit(scaler.transform(x_layer[tr]), y[tr])
        proba[te] = clf.predict_proba(scaler.transform(x_layer[te]))[:, 1]
    return proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vector", default="opinion_mean", choices=["opinion_mean", "name_last"])
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    index, x = load(args.vector)
    y = index.label.values.astype(int)
    groups = index.sample_id.values
    n, n_layers, _ = x.shape
    print(f"{n} opinions, {n_layers} layers, positive rate {y.mean():.3f}, {len(set(groups))} hearings, vector={args.vector}")

    rows, all_probas = [], {}
    for layer in range(n_layers):
        proba = oof_probas(np.asarray(x[:, layer, :], dtype=np.float32), y, groups, C=args.C)
        f1, thr = best_f1_threshold(y, proba)
        pred = proba >= thr
        rows.append({"layer": layer, "roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
                     "f1_best": f1, "threshold": thr,
                     "precision": (pred & (y == 1)).sum() / max(1, pred.sum()),
                     "recall": (pred & (y == 1)).sum() / max(1, (y == 1).sum())})
        all_probas[layer] = proba
        r = rows[-1]
        print(f"layer {layer:2d}  ROC-AUC {r['roc_auc']:.3f}  PR-AUC {r['pr_auc']:.3f}  F1@best {r['f1_best']:.3f} "
              f"(P {r['precision']:.3f} R {r['recall']:.3f} thr {thr:.3f})", flush=True)

    sweep = pd.DataFrame(rows)
    sweep.to_csv(ROOT / "data" / "publichearingbr" / f"probe_layer_sweep_{args.vector}{SUFFIX}.csv", index=False)
    best = int(sweep.loc[sweep.roc_auc.idxmax(), "layer"])
    print(f"\nbest layer by ROC-AUC: {best}  (ROC-AUC {sweep.roc_auc.max():.3f}, PR-AUC {sweep.loc[best, 'pr_auc']:.3f}; "
          f"chance PR-AUC = {y.mean():.3f})")

    if args.vector == "opinion_mean":
        out = ROOT / "data" / "publichearingbr" / f"probe_scores{SUFFIX}.jsonl"
        with open(out, "w") as f:
            for i, r in index.iterrows():
                f.write(json.dumps({"row": int(r.row), "sample_id": int(r.sample_id), "nome": r.nome, "opinion": r.opinion,
                                    "label": int(r.label), "layer": best, "proba": float(all_probas[best][i])}, ensure_ascii=False) + "\n")
        print(f"wrote out-of-fold probe scores at layer {best} to {out}")


if __name__ == "__main__":
    main()
