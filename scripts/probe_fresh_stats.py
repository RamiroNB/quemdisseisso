"""Statistics for the output of probe_fresh_corruptions.py.

Probe = balanced logistic regression on the in-domain activations of the dataset's
labelled opinions (activations_<model_tag>/opinion_mean.npy, human label
verificacao_manual), one per saved layer, hearing-disjoint: each fresh row is scored
by a probe that never saw the labelled opinions of its own hearing. Referee =
1 - content-word containment of the claim in the four windows the probe saw.
Reported in stats.md:
  1. paired detection per corruption type: share of pairs where the corrupted claim
     scores higher than its original (probe; referee, with its ties), mean shift,
     sign-test p; and pooled ROC-AUC corrupted vs original per layer;
  2. the probe on the original opinions by arm: mean P, share above the best-F1
     threshold of run_publichearingbr_probe.py's sweep (0.5 if missing), referee
     ungrounded share (containment < 0.3 in all windows) and fabricated numbers,
     paired with the baseline arm by participant (sign-flip permutation by hearing).

Usage: uv run python scripts/probe_fresh_stats.py <probe_fresh dir tag>
Env: OCARANDU_ACT_DIR (default activations_<model_tag>), OCARANDU_GEN_LAYER (layer of the
paired tables; default the middle saved layer). Writes <dir>/stats.md and <dir>/scores.csv.
CPU only.
"""
import json
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content_hallucination_stats import block_perm_p  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ACT_NAME = os.environ.get("OCARANDU_ACT_DIR")
SEED = 0


def train_probe(act_dir, layer, exclude_hearings=()):
    """Balanced logistic probe on the dataset's labelled opinions at `layer`, leaving out the hearings in
    exclude_hearings."""
    index = pd.read_json(act_dir / "index.jsonl", lines=True)
    n_done = int((act_dir / "done.txt").read_text())
    index = index[index.row < n_done]
    keep = ~index.sample_id.isin(set(exclude_hearings)).values
    x = np.asarray(np.load(act_dir / "opinion_mean.npy", mmap_mode="r")[: len(index), layer, :], dtype=np.float32)[keep]
    y = index.label.values.astype(int)[keep]
    sc = StandardScaler().fit(x)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED).fit(sc.transform(x), y)
    return sc, clf, y.mean()


def grouped_scores(act_dir, layer, acts_layer, hearings, n_folds=5):
    """Score fresh rows with hearing-disjoint probes: the rows' hearings are split into n_folds folds, and rows
    of fold k are scored by a probe trained without the labelled opinions of fold k's hearings."""
    uniq = sorted(set(hearings))
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    fold_of = {h: i % n_folds for i, h in enumerate(uniq)}
    folds = np.array([fold_of[h] for h in hearings])
    out = np.zeros(len(hearings))
    for k in range(n_folds):
        m = folds == k
        if not m.any():
            continue
        sc, clf, _ = train_probe(act_dir, layer, exclude_hearings=[h for h in uniq if fold_of[h] == k])
        out[m] = clf.predict_proba(sc.transform(acts_layer[m]))[:, 1]
    return out


def best_f1_threshold(model_tag):
    p = ROOT / "data" / "publichearingbr" / f"probe_layer_sweep_opinion_mean_{model_tag}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).set_index("layer")["threshold"].to_dict()


def main():
    tag = sys.argv[1]
    d = ROOT / "data" / "publichearingbr" / "probe_fresh" / tag
    meta = json.load(open(d / "meta.json"))
    df = pd.read_json(d / "rows.jsonl", lines=True)
    df = df[df.idx < meta["n"]].reset_index(drop=True)
    acts = np.load(d / "acts.npy", mmap_mode="r")
    layers = meta["layers"]
    model_tag = meta["model"].split("/")[-1].lower().replace(".", "").replace("-", "_")
    act_dir = ROOT / "data" / "publichearingbr" / (ACT_NAME or f"activations_{model_tag}")
    thr_by_layer = best_f1_threshold(model_tag) or {}
    main_layer = int(os.environ.get("OCARANDU_GEN_LAYER", layers[len(layers) // 2]))
    if main_layer not in layers:
        main_layer = layers[len(layers) // 2]

    L = [f"# The probe on fresh text with rule-based corruptions ({tag})", "",
         f"Model {meta['model']}; runs {', '.join(meta['tags'])}; arms {', '.join(meta['conditions'])}; "
         f"{meta['n_original']} original opinions, {meta['n']} rows (variants: "
         + ", ".join(f"{k} {v}" for k, v in df.variant.value_counts().items()) + f"). Probe layer for the paired tables: L{main_layer}.", ""]

    scores = {}
    for li, layer in enumerate(layers):
        x = np.asarray(acts[: len(df), li, :], dtype=np.float32)
        scores[layer] = grouped_scores(act_dir, layer, x, df.sample_id.values)
        df[f"p{layer}"] = scores[layer]
    df["ref"] = 1 - df.containment_shown

    # ---------- 1. paired detection per corruption ----------
    orig = df[df.variant == "original"].set_index("pair_id")
    L += ["## 1. Paired detection: does the score rise from the original to its corrupted twin?", "",
          f"| corruption | pairs | probe L{main_layer}: corrupted > original | mean ΔP | sign-test p | referee: corrupted > original | mean Δ(1-containment) | referee ties | Δ content words |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in ("polarity", "number", "entity", "scope", "cross"):
        c = df[df.variant == kind]
        if c.empty:
            continue
        o = orig.loc[c.pair_id]
        dp = c[f"p{main_layer}"].values - o[f"p{main_layer}"].values
        dr = c.ref.values - o.ref.values
        win = (dp > 0).mean()
        p = binomtest(int((dp > 0).sum()), int((dp != 0).sum()), 0.5).pvalue if (dp != 0).sum() else 1.0
        ties = (dr == 0).mean()
        rwin = (dr > 0).mean()
        L.append(f"| {kind} | {len(c)} | {win:.3f} | {dp.mean():+.3f} | {p:.2g} | {rwin:.3f} | {dr.mean():+.3f} | {ties:.2f} | "
                 f"{(c.n_response_tokens.values - o.n_response_tokens.values).mean():+.1f} tok |")
    L += ["", "Referee = 1 - containment of the claim in the four shown windows (the same windows as the probe saw). "
              "A tie means the corruption did not change the content-word set, so the referee cannot see it."]

    # per layer, pooled AUC corrupted vs original within type
    L += ["", "### Pooled ROC-AUC corrupted vs original, by layer", "",
          "| corruption | " + " | ".join(f"L{l}" for l in layers) + " | referee |", "|---|" + "---:|" * (len(layers) + 1)]
    for kind in ("polarity", "number", "entity", "scope", "cross"):
        c = df[df.variant == kind]
        if c.empty:
            continue
        o = orig.loc[c.pair_id]
        y = np.r_[np.ones(len(c)), np.zeros(len(o))]
        cells = [f"{roc_auc_score(y, np.r_[c[f'p{l}'].values, o[f'p{l}'].values]):.3f}" for l in layers]
        cells.append(f"{roc_auc_score(y, np.r_[c.ref.values, o.ref.values]):.3f}")
        L.append(f"| {kind} | " + " | ".join(cells) + " |")

    # ---------- 2. firing on originals by arm ----------
    thr = thr_by_layer.get(main_layer, 0.5)
    L += ["", f"## 2. The probe on the pipeline's own opinions (originals), by arm — L{main_layer}, threshold {thr:.3f} (dataset best-F1)", "",
          "| arm | opinions | mean P(halluc) | share flagged | referee ungrounded (<0.3, all windows) | fabricated number |",
          "|---|---:|---:|---:|---:|---:|"]
    og = df[df.variant == "original"].copy()
    og["flag"] = og[f"p{main_layer}"] > thr
    og["ung"] = og.containment_all < 0.3
    per = og.groupby(["run", "sample_id", "speaker", "condition"]).agg(p=(f"p{main_layer}", "mean"), flag=("flag", "mean"),
                                                                     ung=("ung", "mean"), fab=("fabricated_number", "mean")).reset_index()
    base = per[per.condition == "baseline"].set_index(["run", "sample_id", "speaker"])
    for arm in meta["conditions"]:
        a = og[og.condition == arm]
        pa = per[per.condition == arm].set_index(["run", "sample_id", "speaker"])
        pa = pa.loc[pa.index.intersection(base.index)]
        b = base.loc[pa.index]
        sids = np.array([i[1] for i in pa.index])
        def cell(col, fmt=".3f"):
            v = a[{"p": f"p{main_layer}", "flag": "flag", "ung": "ung", "fab": "fabricated_number"}[col]].mean()
            if arm == "baseline" or pa.empty:
                return f"{v:{fmt}}"
            dlt = pa[col].values - b[col].values
            return f"{v:{fmt}} ({dlt.mean():+.3f}, p={block_perm_p(dlt, sids):.3g})"
        L.append(f"| {arm} | {len(a)} | {cell('p')} | {cell('flag')} | {cell('ung')} | {cell('fab')} |")
    L.append("")
    L.append("Deltas: per-participant means paired with the baseline arm, sign-flip permutation by hearing.")
    # agreement probe vs referee on originals
    for arm in meta["conditions"]:
        a = og[og.condition == arm]
        if a.ung.sum() >= 10:
            L.append(f"{arm}: probe ROC-AUC for the referee's ungrounded label {roc_auc_score(a.ung, a[f'p{main_layer}']):.3f}; "
                     f"for fabricated numbers {roc_auc_score(a.fabricated_number, a[f'p{main_layer}']):.3f}" if a.fabricated_number.sum() >= 10 else
                     f"{arm}: probe ROC-AUC for the referee's ungrounded label {roc_auc_score(a.ung, a[f'p{main_layer}']):.3f}.")

    text = "\n".join(L) + "\n"
    (d / "stats.md").write_text(text)
    df[["idx", "pair_id", "variant", "run", "condition", "sample_id", "speaker", "sample", "opinion_idx", "text",
        "containment_shown", "containment_all", "fabricated_number"] + [f"p{l}" for l in layers]].to_csv(d / "scores.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()
