"""Probe and baseline statistics for the output of lc_stream_extract.py.

One probe protocol throughout: logistic regression on the saved residual vector, 5-fold GroupKFold
by hearing, fit on baseline rows of the training hearings and scored on every arm's rows of the
held-out hearings. Competitors from the same forward pass: next-token entropy, 1 - max probability,
1 - copy mass. Targets (labels defined in lc_stream_extract.py):
  tokens  the next content word is novel (framing verbs excluded); the next number is novel
  lines   the opinion about to be written is ungrounded (containment < 0.3), has a novel content word,
          has a novel number
  items   read at the last prompt token: the output will contain an ungrounded opinion / a fabricated number
Also reports copy mass, entropy and novel-word share per arm, paired with baseline by participant
(sign-flip permutation by hearing).

Usage: uv run python scripts/lc_stream_stats.py <lc_stream dir name, e.g. llama_31_8b_instruct>
Env: OCARANDU_LS_MAX_FIT (default 60000) caps the training rows of each probe fit. CPU only.
Writes <dir>/stats.md, and probes fit on all baseline rows (scaler + weights per layer):
<dir>/probe_tokens_novel.npz (read by the gate conditions of longcontext_speaker_generation.py and by
lc_retrieve_regen.py) and <dir>/probe_lines_ungrounded.npz.
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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content_hallucination_stats import block_perm_p  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 0
MAX_FIT = int(os.environ.get("OCARANDU_LS_MAX_FIT", "60000"))  # cap on training rows per probe fit (random subsample)


def load(d, name):
    meta = json.load(open(d / "meta.json"))
    n = meta[f"n_{name}"]
    df = pd.read_json(d / f"{name}.jsonl", lines=True)
    df = df[df.idx < n].reset_index(drop=True)
    x = np.load(d / f"{name}.npy", mmap_mode="r")
    return meta, df, x


def probe_scores(x, df, y, layers, n_splits=5):
    """One probe per layer, trained on baseline rows of the training hearings,
    scored on all rows of the test hearings. Returns {layer: scores array}."""
    groups = df.sample_id.values
    is_base = (df.condition == "baseline").values
    out = {li: np.full(len(df), np.nan) for li in range(len(layers))}
    rng = np.random.default_rng(SEED)
    for tr, te in GroupKFold(n_splits).split(np.zeros(len(df)), y, groups):
        tr = tr[is_base[tr]]
        if len(tr) > MAX_FIT:
            tr = rng.choice(tr, MAX_FIT, replace=False)
        if len(set(y[tr])) < 2:
            continue
        for li in range(len(layers)):
            xtr = np.asarray(x[np.sort(tr), li, :], dtype=np.float32)
            ytr = y[np.sort(tr)]
            sc = StandardScaler().fit(xtr)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED).fit(sc.transform(xtr), ytr)
            xte = np.asarray(x[np.sort(te), li, :], dtype=np.float32)
            out[li][np.sort(te)] = clf.predict_proba(sc.transform(xte))[:, 1]
    return out


def save_probes(x, df, y, layers, out_path):
    """Fit one probe per layer on all baseline rows (subsampled to MAX_FIT) and save scaler + weights, so a
    generation-time gate (intervene only where the probe fires) can reuse them without refitting."""
    is_base = np.where((df.condition == "baseline").values)[0]
    rng = np.random.default_rng(SEED)
    if len(is_base) > MAX_FIT:
        is_base = np.sort(rng.choice(is_base, MAX_FIT, replace=False))
    if len(set(y[is_base])) < 2:
        return
    out = {"layers": np.array(layers)}
    for li, layer in enumerate(layers):
        xtr = np.asarray(x[is_base, li, :], dtype=np.float32)
        sc = StandardScaler().fit(xtr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED).fit(sc.transform(xtr), y[is_base])
        out[f"mean_{layer}"], out[f"scale_{layer}"] = sc.mean_.astype(np.float32), sc.scale_.astype(np.float32)
        out[f"coef_{layer}"], out[f"intercept_{layer}"] = clf.coef_[0].astype(np.float32), np.float32(clf.intercept_[0])
    np.savez(out_path, **out)


def auc_table(df, y, scores, layers, arms, title, lines):
    lines += ["", f"### {title}", "",
              "| arm | n | positives | " + " | ".join(f"probe L{l}" for l in layers) + " | entropy | 1-maxp | 1-copy mass |",
              "|---|---:|---:|" + "---:|" * (len(layers) + 3)]
    for arm in arms:
        m = (df.condition == arm).values & ~np.isnan(scores[0])
        ya = y[m]
        if len(set(ya)) < 2 or ya.sum() < 10:
            lines.append(f"| {arm} | {int(m.sum())} | {int(ya.sum())} | " + " | ".join(["—"] * (len(layers) + 3)) + " |")
            continue
        cells = []
        for li in range(len(layers)):
            s = scores[li][m]
            cells.append(f"{roc_auc_score(ya, s):.3f} / {average_precision_score(ya, s):.3f}")
        for comp in (df.entropy.values[m], 1 - df.maxp.values[m], 1 - df.copy_mass.values[m]):
            cells.append(f"{roc_auc_score(ya, comp):.3f} / {average_precision_score(ya, comp):.3f}")
        lines.append(f"| {arm} | {int(m.sum())} | {int(ya.sum())} ({ya.mean():.3f}) | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Cells: ROC-AUC / PR-AUC (chance PR-AUC = positive rate). Probe trained on baseline rows of other hearings.")


def main():
    tag = sys.argv[1]
    d = ROOT / "data" / "publichearingbr" / "lc_stream" / tag
    meta, tok, xt = load(d, "tokens")
    _, lin, xl = load(d, "lines")
    _, itm, xi = load(d, "items")
    layers = meta["layers"]
    arms = meta["conditions"]
    L = [f"# Reading the stream before the word ({tag})", "",
         f"Model {meta['model']}; runs {', '.join(meta['tags'])}; arms {', '.join(arms)}; layers {layers}. "
         f"{len(itm)} generations re-read, {len(tok)} content/number tokens, {len(lin)} opinion lines.", ""]

    # ---------- anchoring mechanism by arm (item level, paired by participant) ----------
    L += ["## Anchoring at the mechanism level (per generation, paired by participant)", "",
          "| arm | copy mass at generated positions | entropy | novel content words / content words | opinions | ungrounded items |",
          "|---|---:|---:|---:|---:|---:|"]
    # two gold participants can map to the same transcript speaker: keep one row
    dedup = itm.drop_duplicates(["run", "condition", "sample_id", "speaker"])
    base = dedup[dedup.condition == "baseline"].set_index(["run", "sample_id", "speaker"])
    for arm in arms:
        a = dedup[dedup.condition == arm].set_index(["run", "sample_id", "speaker"])
        a = a.loc[a.index.intersection(base.index)]
        if a.empty:
            L.append(f"| {arm} | — | — | — | — | — |")
            continue
        b = base.loc[a.index]
        nov = a.n_novel_words / a.n_content_words.clip(lower=1)
        novb = b.n_novel_words / b.n_content_words.clip(lower=1)
        sids = np.array([i[1] for i in a.index])
        def cell(va, vb):
            if arm == "baseline":
                return f"{va.mean():.3f}"
            dlt = (va.values - vb.values)
            return f"{va.mean():.3f} ({dlt.mean():+.3f}, p={block_perm_p(dlt, sids):.3g})"
        L.append(f"| {arm} | {cell(a.mean_copy_mass_gen, b.mean_copy_mass_gen)} | {cell(a.mean_entropy_gen, b.mean_entropy_gen)} | "
                 f"{cell(nov, novb)} | {a.n_opinions.mean():.2f} | {a.any_ungrounded.mean():.3f} |")
    L.append("")
    L.append("Copy mass = probability mass the unsteered network puts on tokens of the participant's own turns, "
             "averaged over the generated positions of that arm's text (teacher-forced). Deltas vs baseline, "
             "sign-flip permutation by hearing.")

    # ---------- token level ----------
    ct = tok[tok.is_content & ~tok.framing].reset_index(drop=True)
    idx_ct = tok.index[tok.is_content & ~tok.framing].values
    y = ct.novel.values.astype(int)
    L += ["", "## Token level: is the next content word going to be novel?", "",
          f"{len(ct)} content-word positions (framing verbs excluded); novel share by arm: " +
          ", ".join(f"{arm} {ct[ct.condition == arm].novel.mean():.3f}" for arm in arms) + "."]
    sc = probe_scores(_Sub(xt, idx_ct), ct, y, layers)
    auc_table(ct, y, sc, layers, arms, "Novel content word, read one token before", L)
    save_probes(_Sub(xt, idx_ct), ct, y, layers, d / "probe_tokens_novel.npz")
    # numbers
    nm = tok[tok.is_number].reset_index(drop=True)
    idx_nm = tok.index[tok.is_number].values
    yn = nm.novel_number.values.astype(int)
    if yn.sum() >= 30 and (1 - yn).sum() >= 30:
        scn = probe_scores(_Sub(xt, idx_nm), nm, yn, layers)
        auc_table(nm, yn, scn, layers, arms, "Novel number (fabricated), read one token before", L)
    else:
        L.append(f"\nNumbers: {len(nm)} number tokens, {int(yn.sum())} novel — too few for a probe table.")

    # ---------- line level ----------
    L += ["", "## Line level: read before the opinion starts"]
    lin["ungrounded"] = (lin.containment < 0.3).astype(int)
    for target in ("ungrounded", "has_novel_word", "has_novel_number"):
        yl = lin[target].values.astype(int)
        if yl.sum() < 20:
            L.append(f"\n{target}: only {int(yl.sum())} positives; skipped.")
            continue
        scl = probe_scores(xl, lin, yl, layers)
        auc_table(lin, yl, scl, layers, arms, f"Opinion will be {target}", L)
        if target == "ungrounded":
            save_probes(xl, lin, yl, layers, d / "probe_lines_ungrounded.npz")

    # ---------- item level ----------
    L += ["", "## Item level: read at the last prompt token, before any output"]
    for target in ("any_ungrounded", "fabricated_number"):
        yi = itm[target].values.astype(int)
        if yi.sum() < 20:
            L.append(f"\n{target}: only {int(yi.sum())} positives; skipped.")
            continue
        sci = probe_scores(xi, itm, yi, layers)
        auc_table(itm, yi, sci, layers, arms, f"Output will have {target}", L)
        # length as the trivial competitor
        for arm in arms:
            m = (itm.condition == arm).values
            if len(set(yi[m])) == 2:
                L.append(f"prompt length as a score ({arm}): ROC-AUC {roc_auc_score(yi[m], -itm.n_prompt_tokens.values[m]):.3f} (score = −prompt tokens: a shorter floor scores higher).")

    text = "\n".join(L) + "\n"
    (d / "stats.md").write_text(text)
    print(text)


class _Sub:
    """Row-subset view of a memmap: x[rows_sorted, li, :] without loading everything."""
    def __init__(self, x, idx):
        self.x, self.idx = x, np.asarray(idx)

    def __getitem__(self, key):
        rows, li, _ = key
        return self.x[self.idx[rows], li, :]

    def __len__(self):
        return len(self.idx)


if __name__ == "__main__":
    main()
