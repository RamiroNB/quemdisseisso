"""Steering in free generation: one-sentence summaries of the dataset's four evidence chunks.

Each item is an NLI opinion's four untagged `chunks_proximos`; the model writes one sentence on what was
said, once per condition. The referee is lexical, independent of the probe behind the direction:
  coverage           share of the generation's content words found in the best-matching chunk
                     (None when abstained)
  fabricated_number  the generation contains a number not found among the chunks' numbers
  abstained          refusal regex (ABSTAIN_RE of attribution_generation.py)

With OCARANDU_CONTENT_HOLDOUT=1 (default) hearings are split into a direction-fit half and an eval half,
and the eval half again into dev and test (OCARANDU_GEN_SPLIT=dev|test; unset = the whole eval half).
Directions (unit vectors; the PT-BR label-based ones use the fit half only): halluc = faithful minus
hallucinated mean opinion activation (human labels); probe_w = logistic-regression weights; misattr
(identity-swap activations, if present); halluc_en = the halluc difference on English RAGTruth train
sentences (if extracted); random / random<k> = seeded controls.
Conditions: baseline or <direction>[@add|@nrm|@rot]:<amount> (see intervene()).

Usage: uv run python scripts/content_hallucination_generation.py   (one GPU; configured by env)
Env: OCARANDU_MODEL_REPO, OCARANDU_DEVICE, OCARANDU_ACT_DIR, OCARANDU_GEN_LAYER (14), OCARANDU_GEN_N (300),
  OCARANDU_GEN_PER_HEARING (1), OCARANDU_GEN_MAXNEW (60), OCARANDU_GEN_CONDITIONS,
  OCARANDU_GEN_PROMPT=sentence|grounded|paragraph, OCARANDU_GEN_SAMPLE=1 + OCARANDU_GEN_SAMPLE_SEED
  (nucleus sampling, T 0.7, top-p 0.9, instead of greedy), OCARANDU_GEN_TAG (output suffix).
Output: data/publichearingbr/content_hallucination_generation/<model_tag>[_<tag>]/{generations.jsonl,summary.csv}
"""

import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attribution_generation import ABSTAIN_RE, containment, content_words, norm  # noqa: E402
from ocarandu.data.publichearingbr import load_nli_opinions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = os.environ.get("OCARANDU_MODEL_REPO", "meta-llama/Llama-3.1-8B-Instruct")
DEVICE = os.environ.get("OCARANDU_DEVICE", "cuda:0")
MODEL_TAG = REPO_ID.split("/")[-1].lower().replace(".", "").replace("-", "_")
ACT_DIR = ROOT / "data" / "publichearingbr" / os.environ.get("OCARANDU_ACT_DIR", f"activations_{MODEL_TAG}")
GEN_TAG = os.environ.get("OCARANDU_GEN_TAG", "")
OUT_DIR = ROOT / "data" / "publichearingbr" / "content_hallucination_generation" / (MODEL_TAG + (("_" + GEN_TAG) if GEN_TAG else ""))
# opinions drawn per eval hearing; with >1 the stats script permutes in blocks of one hearing
PER_HEARING = int(os.environ.get("OCARANDU_GEN_PER_HEARING", "1"))
LAYER = int(os.environ.get("OCARANDU_GEN_LAYER", "14"))
N_ITEMS = int(os.environ.get("OCARANDU_GEN_N", "300"))
MAX_NEW = int(os.environ.get("OCARANDU_GEN_MAXNEW", "60"))
CONDITIONS = os.environ.get(
    "OCARANDU_GEN_CONDITIONS",
    "baseline,halluc:0.5,halluc:1.0,halluc:-0.5,misattr:1.0,random:1.0"
).split(",")
HOLDOUT = os.environ.get("OCARANDU_CONTENT_HOLDOUT", "1") == "1"
# decoding: greedy (default) or seeded nucleus sampling
SAMPLE = os.environ.get("OCARANDU_GEN_SAMPLE", "0") == "1"
SAMPLE_SEED = int(os.environ.get("OCARANDU_GEN_SAMPLE_SEED", "0"))
# prompt: sentence (default), grounded (explicit use-only-the-excerpts instruction) or paragraph
PROMPT_KIND = os.environ.get("OCARANDU_GEN_PROMPT", "sentence")
# English RAGTruth sentence activations of the same model (extract_ragtruth_sentence_mean.py),
# source of the cross-lingual direction `halluc_en`
RT_DIR = ROOT / "data" / "ragtruth_sentence" / MODEL_TAG
SEED = 0

PROMPTS = {
    "sentence": (
        "A seguir estão trechos de uma transcrição de audiência pública da Câmara dos Deputados.\n\n{chunks}\n\n"
        "Com base apenas nos trechos acima, escreva uma única frase relatando o que foi afirmado."
    ),
    # prompt-only competitor of steering: an explicit instruction to use only the excerpts
    "grounded": (
        "A seguir estão trechos de uma transcrição de audiência pública da Câmara dos Deputados.\n\n{chunks}\n\n"
        "Com base apenas nos trechos acima, escreva uma única frase relatando o que foi afirmado. "
        "Use somente informações que estejam escritas nos trechos: não acrescente nomes, números, datas, "
        "instituições, cargos ou conclusões que não apareçam acima. Se algo não estiver nos trechos, não mencione."
    ),
    "paragraph": (
        "A seguir estão trechos de uma transcrição de audiência pública da Câmara dos Deputados.\n\n{chunks}\n\n"
        "Com base apenas nos trechos acima, escreva um parágrafo de três frases, em terceira pessoa, "
        "relatando o que foi afirmado."
    ),
}
PROMPT = PROMPTS[PROMPT_KIND]
NUMBER_RE = re.compile(r"\b\d[\d.,]*\s?%?\b")


def fabricated_numbers(gen, chunks):
    """Numbers/percentages in the generation absent from every shown chunk."""
    shown = set()
    for c in chunks:
        shown |= set(NUMBER_RE.findall(c))
    gen_nums = set(NUMBER_RE.findall(gen))
    return bool(gen_nums) and not (gen_nums <= shown)


def build_eval_set(rng, restrict_to=None):
    ops = [o for o in load_nli_opinions() if len(o.context_chunks) == 4]
    by_hearing = {}
    for o in ops:
        by_hearing.setdefault(o.sample_id, []).append(o)
    hearings = sorted(by_hearing)
    if restrict_to is not None:
        hearings = [h for h in hearings if h in restrict_to]
    rng.shuffle(hearings)
    items = []
    for h in hearings:
        ops_h = list(by_hearing[h])
        rng.shuffle(ops_h)
        items.extend(ops_h[:PER_HEARING])
        if len(items) >= N_ITEMS:
            break
    return items[:N_ITEMS]


def hearing_split():
    """Deterministic split of the hearings with a 4-chunk opinion into disjoint direction-fit and
    eval halves, fixed up front so the fit set can never be empty (an empty one would give NaN
    directions without an error). The eval half is split again into dev and test (independent
    shuffle, SEED+1) so layer and alpha can be chosen on dev and reported on test; the directions
    depend only on the fit half, so they are identical across dev, test and no-split runs.
    Returns (fit, eval_pool, dev, test)."""
    ops = [o for o in load_nli_opinions() if len(o.context_chunks) == 4]
    all_h = sorted({o.sample_id for o in ops})
    rng = random.Random(SEED)
    rng.shuffle(all_h)
    half = len(all_h) // 2
    fit_hearings, eval_pool = set(all_h[:half]), all_h[half:]
    dev_test = list(eval_pool)
    random.Random(SEED + 1).shuffle(dev_test)
    dev_half = len(dev_test) // 2
    return fit_hearings, eval_pool, dev_test[:dev_half], dev_test[dev_half:]  # fit, eval_pool, dev, test


def directions(layer, fit_hearings):
    """-> ({name: unit vector, "hidden": size}, mean activation norm). halluc = faithful minus
    hallucinated mean opinion activation; probe_w = logistic-regression weights; misattr and
    halluc_en only when their inputs exist; random = fixed seed. The PT-BR label-based directions
    use fit_hearings only when HOLDOUT; the norm is the mean over all opinions."""
    out = {}
    index = pd.read_json(ACT_DIR / "index.jsonl", lines=True)
    mask = index.sample_id.isin(fit_hearings).values if HOLDOUT else np.ones(len(index), dtype=bool)
    x_all = np.asarray(np.load(ACT_DIR / "opinion_mean.npy", mmap_mode="r")[:, layer, :], dtype=np.float32)
    x, y = x_all[mask], index.label.values.astype(int)[mask]
    v = x[y == 0].mean(0) - x[y == 1].mean(0)
    out["halluc"] = v / np.linalg.norm(v)
    act_norm = float(np.linalg.norm(x_all).mean() if x_all.ndim == 1 else np.linalg.norm(x_all, axis=1).mean())

    swap_dir = ROOT / "data" / "publichearingbr" / "identity_swap" / (MODEL_TAG + "_tagged")
    if swap_dir.exists():
        meta = json.load(open(swap_dir / "meta.json"))
        if layer in meta["save_layers"]:
            li = meta["save_layers"].index(layer)
            sw = pd.read_json(swap_dir / "results.jsonl", lines=True)
            sw = sw[sw.row < meta["n_rows"]]
            if HOLDOUT:
                sw = sw[sw.sample_id.isin(fit_hearings)]
            acts = np.load(swap_dir / "acts_opinion_mean.npy", mmap_mode="r")
            w = (np.asarray(acts[sw[sw.variant == "original"].row.values, li, :], dtype=np.float32).mean(0)
                 - np.asarray(acts[sw[sw.variant == "within_hearing"].row.values, li, :], dtype=np.float32).mean(0))
            out["misattr"] = w / np.linalg.norm(w)

    # logistic-regression weight direction on the same fit hearings
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scl = StandardScaler().fit(x)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED).fit(scl.transform(x), 1 - y)
    pw = clf.coef_[0] / scl.scale_  # raises P(faithful) in raw activation space
    out["probe_w"] = pw / np.linalg.norm(pw)

    # English direction: faithful minus hallucinated RAGTruth sentence means at the
    # same layer, content-bearing sentences, official train split only
    if (RT_DIR / "meta.json").exists():
        rt = json.load(open(RT_DIR / "meta.json"))
        if layer in rt["layers"]:
            ridx = pd.read_json(RT_DIR / "index.jsonl", lines=True)
            ridx = ridx[(ridx.row < rt["n_rows"]) & (ridx.split == "train") & (ridx.filler == "content_bearing")]
            racts = np.load(RT_DIR / "sent_mean.npy", mmap_mode="r")
            li = rt["layers"].index(layer)
            xr = np.asarray(racts[ridx.row.values, li, :], dtype=np.float32)
            yr = ridx.label.values.astype(int)
            ve = xr[yr == 0].mean(0) - xr[yr == 1].mean(0)
            out["halluc_en"] = ve / np.linalg.norm(ve)
            print(f"cos(halluc_pt, halluc_en) = {float(out['halluc'] @ out['halluc_en']):+.3f}", file=sys.stderr)

    out["hidden"] = x_all.shape[1]
    out["random"] = random_direction(0, x_all.shape[1])
    return out, act_norm


def random_direction(seed, hidden):
    """Unit vector from a fixed seed; conditions random0..randomK give independent controls."""
    r = np.random.default_rng(seed).standard_normal(hidden).astype(np.float32)
    return r / np.linalg.norm(r)


def main():
    rng = random.Random(SEED)
    fit_hearings, eval_pool, dev_hearings, test_hearings = hearing_split() if HOLDOUT else (None, None, None, None)
    # "" -> the whole eval half; "dev"/"test" -> disjoint halves of it (choose layer/alpha on dev)
    SPLIT = os.environ.get("OCARANDU_GEN_SPLIT", "")
    if SPLIT not in ("", "dev", "test"):
        raise SystemExit(f"OCARANDU_GEN_SPLIT must be '', 'dev' or 'test', got {SPLIT!r}")
    restrict = {"": eval_pool, "dev": dev_hearings, "test": test_hearings}[SPLIT] if HOLDOUT else None
    eval_items = build_eval_set(random.Random(SEED), restrict_to=restrict)
    eval_hearings = [o.sample_id for o in eval_items]
    print(f"{len(eval_items)} held-out hearings for generation (split={SPLIT or 'none'}); holdout direction fit = {HOLDOUT}"
          + (f" ({len(fit_hearings)} hearings reserved for direction fit)" if HOLDOUT else ""),
          file=sys.stderr, flush=True)

    dirs, act_norm = directions(LAYER, fit_hearings)
    hidden = dirs.pop("hidden")
    for k, v in dirs.items():
        if np.isnan(v).any() or not (0.99 < np.linalg.norm(v) < 1.01):
            raise SystemExit(f"direction {k!r} is invalid (nan={np.isnan(v).any()}, norm={np.linalg.norm(v):.3f}) -- refusing to generate")
    print(f"directions available: {list(dirs)}; act_norm={act_norm:.3f}; "
          f"cos(halluc,probe_w)={float(dirs['halluc'] @ dirs['probe_w']):+.3f}", file=sys.stderr, flush=True)

    tok = AutoTokenizer.from_pretrained(REPO_ID)
    model = AutoModelForCausalLM.from_pretrained(REPO_ID, dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    layer_mod = model.model.layers[LAYER]
    state = {"dir": None, "mode": "add", "amt": 0.0}

    def intervene(h):
        """Move the residual stream h toward the unit direction d (N = mean activation norm).
        add     h + a*N*d                       (changes the norm)
        nrm     (h + a*N*d) rescaled to |h|     (norm-preserving addition)
        rot     rotate h toward d by a degrees, in the plane of h and the part of d
                orthogonal to h (|h| preserved)"""
        mode, d, a = state["mode"], state["dir"], state["amt"]
        if mode == "add":
            return h + a * act_norm * d
        if mode == "nrm":
            hn = h.norm(dim=-1, keepdim=True)
            out = h + a * act_norm * d
            return out * (hn / out.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        if mode == "rot":
            hf = h.float()
            hn = hf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            hu = hf / hn
            df = d.float()
            u = df - (hu * df).sum(-1, keepdim=True) * hu  # part of d orthogonal to h
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            th = torch.tensor(a * 3.141592653589793 / 180.0, device=hf.device)
            return (hn * (torch.cos(th) * hu + torch.sin(th) * u)).to(h.dtype)
        raise ValueError(mode)

    def hook(module, inputs, output):
        if state["dir"] is None:
            return output
        if isinstance(output, tuple):
            return (intervene(output[0]),) + tuple(output[1:])
        return intervene(output)

    layer_mod.register_forward_hook(hook)

    def generate(user):
        if tok.chat_template:
            ids = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, return_tensors="pt")
            ids = ids["input_ids"] if not isinstance(ids, torch.Tensor) else ids
        else:
            ids = tok(f"<s>[INST] {user.strip()} [/INST]", add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = ids.to(DEVICE)
        with torch.no_grad():
            if SAMPLE:
                torch.manual_seed(SAMPLE_SEED * 100003 + ids.shape[1])  # per-prompt reproducible
                out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.7, top_p=0.9,
                                     pad_token_id=tok.eos_token_id)
            else:
                out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    f = open(OUT_DIR / "generations.jsonl", "w")
    for cond in CONDITIONS:
        # condition grammar: <direction>[@add|@nrm|@rot]:<amount>; amount is alpha in
        # units of the mean activation norm for add/nrm and degrees for rot
        mode = "add"
        if cond == "baseline":
            dname, alpha = "baseline", 0.0
        else:
            head, a = cond.split(":")
            alpha = float(a)
            if "@" in head:
                dname, mode = head.split("@")
            else:
                dname = head
            m_rand = re.fullmatch(r"random(\d+)", dname)
            if m_rand:
                dirs[dname] = random_direction(int(m_rand.group(1)), hidden)
            if dname not in dirs:
                print(f"  skip {cond} (no '{dname}' direction available)", file=sys.stderr)
                continue
        state["mode"], state["amt"] = mode, alpha
        state["dir"] = None if alpha == 0.0 else torch.tensor(dirs[dname], dtype=torch.bfloat16, device=DEVICE)
        for i, o in enumerate(eval_items):
            user = PROMPT.format(chunks="\n\n".join(o.context_chunks))
            gen = generate(user)
            abst = bool(ABSTAIN_RE.search(gen))
            cov = containment(gen, o.context_chunks) if not abst else None
            fab = fabricated_numbers(gen, o.context_chunks) if not abst else False
            row = {"condition": cond, "direction": dname, "alpha": alpha, "sample_id": o.sample_id,
                   "opinion": o.opinion, "chunks": list(o.context_chunks),  # what the model saw, for re-scoring
                   "true_label": int(o.is_hallucination), "generation": gen, "abstained": abst,
                   "coverage": cov, "fabricated_number": fab, "n_chars": len(gen)}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                f.flush()
                print(f"  [{cond}] {i + 1}/{len(eval_items)}", file=sys.stderr, flush=True)
    f.close()

    df = pd.DataFrame(rows)
    summ = []
    for (cond, dname, alpha), g in df.groupby(["condition", "direction", "alpha"]):
        cov = g.coverage.dropna()
        summ.append({
            "condition": cond, "direction": dname, "alpha": alpha, "n": len(g),
            "abstain_rate": float(g.abstained.mean()),
            "mean_coverage": float(cov.mean()) if len(cov) else float("nan"),
            "low_coverage_rate": float((cov < 0.3).mean()) if len(cov) else float("nan"),
            "fabricated_number_rate": float(g.fabricated_number.mean()),
            "mean_len_chars": float(g.n_chars.mean()),
        })
        r = summ[-1]
        print(f"{cond:14s} coverage={r['mean_coverage']:.3f} low_cov={r['low_coverage_rate']:.2f} "
              f"fab_num={r['fabricated_number_rate']:.3f} abstain={r['abstain_rate']:.2f} len={r['mean_len_chars']:.0f}",
              flush=True)
    pd.DataFrame(summ).to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
