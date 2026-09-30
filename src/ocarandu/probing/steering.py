"""Per-layer mean-difference steering vectors and logistic regression probes.

Works on externally extracted per-token activations: one .npz per example with
`activations` [n_layers, n_tokens, hidden], `labels` [n_tokens], `source_id` and,
when known, RAGTruth's `split`. `best_f1_threshold` is also used by
scripts/run_publichearingbr_probe.py.
"""

import os

# Cap BLAS thread pools at 8 (unless already set): an uncapped LogisticRegression
# over multi-GB float32 arrays takes every core on the machine. Only takes
# effect if set before numpy's first import.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Example:
    source_id: str
    activations: np.ndarray  # [n_layers, n_tokens, hidden]
    labels: np.ndarray  # [n_tokens], 0/1
    split: str = ""  # RAGTruth's official train/test split, when known


def load_activations(activations_dir: Path) -> list[Example]:
    examples = []
    for path in sorted(activations_dir.glob("*.npz")):
        z = np.load(path)
        examples.append(
            Example(
                source_id=str(z["source_id"]),
                activations=z["activations"],
                labels=z["labels"],
                split=str(z["split"]) if "split" in z.files else "",
            )
        )
    return examples


def split_by_official(examples: list[Example]) -> tuple[list[Example], list[Example]]:
    """Split using RAGTruth's own per-response train/test split field, which
    is never reshuffled. Use this instead of split_by_example whenever
    examples carry a real RAGTruth split (gold-label data, not fresh
    generations, which have no official split)."""
    train = [ex for ex in examples if ex.split == "train"]
    test = [ex for ex in examples if ex.split == "test"]
    missing = [ex for ex in examples if ex.split not in ("train", "test")]
    if missing:
        raise ValueError(
            f"{len(missing)} examples have no official RAGTruth split (split_by_official "
            "requires real 'train'/'test' values on every example) -- use split_by_example instead"
        )
    return train, test


def split_by_example(
    examples: list[Example], test_frac: float = 0.2, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Split by example (not token) so tokens from the same document never
    leak across train/test."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(examples))
    n_test = int(len(examples) * test_frac)
    test_idx = set(indices[:n_test].tolist())
    train = [ex for i, ex in enumerate(examples) if i not in test_idx]
    test = [ex for i, ex in enumerate(examples) if i in test_idx]
    return train, test


def _flatten_layer(examples: list[Example], layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
    xs = [ex.activations[layer_idx] for ex in examples if ex.labels.size > 0]
    ys = [ex.labels for ex in examples if ex.labels.size > 0]
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def mean_diff_vector(examples: list[Example], layer_idx: int) -> np.ndarray:
    """v = mean(faithful activations) - mean(hallucinated activations), at one layer."""
    x, y = _flatten_layer(examples, layer_idx)
    faithful = x[y == 0]
    hallucinated = x[y == 1]
    if len(faithful) == 0 or len(hallucinated) == 0:
        raise ValueError(f"layer {layer_idx}: need both classes present to compute mean-diff")
    return faithful.mean(axis=0) - hallucinated.mean(axis=0)


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Best (F1, threshold) over the full precision-recall curve.

    Chosen against the same set passed in — this is an optimistic upper
    bound (the threshold is picked with knowledge of that set's labels),
    useful for telling apart "the default 0.5 threshold is a bad fit for
    an imbalanced problem" from "the classes aren't actually separable."
    Not a substitute for tuning the threshold on a held-out validation set.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # last precision/recall point has no corresponding threshold
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1s = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    if len(f1s) == 0:
        return 0.0, 0.5
    best_idx = int(np.argmax(f1s))
    return float(f1s[best_idx]), float(thresholds[best_idx])


@dataclass
class LayerResult:
    layer: int
    f1_default: float
    auc: float
    pr_auc: float  # average precision -- doesn't share ROC-AUC's optimism under class imbalance
    accuracy: float
    f1_best_threshold: float
    best_threshold: float
    precision_at_best_threshold: float
    recall_at_best_threshold: float
    n_train_tokens: int
    n_test_tokens: int


def evaluate_layer(
    train: list[Example], test: list[Example], layer_idx: int, seed: int = 0
) -> LayerResult:
    x_train, y_train = _flatten_layer(train, layer_idx)
    x_test, y_test = _flatten_layer(test, layer_idx)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)[:, 1]
    has_both_classes = len(set(y_test.tolist())) > 1
    f1_best, threshold = best_f1_threshold(y_test, y_proba) if has_both_classes else (0.0, 0.5)
    y_pred_at_best = (y_proba >= threshold).astype(int)

    return LayerResult(
        layer=layer_idx,
        f1_default=f1_score(y_test, y_pred, zero_division=0),
        auc=roc_auc_score(y_test, y_proba) if has_both_classes else float("nan"),
        pr_auc=average_precision_score(y_test, y_proba) if has_both_classes else float("nan"),
        accuracy=accuracy_score(y_test, y_pred),
        f1_best_threshold=f1_best,
        best_threshold=threshold,
        precision_at_best_threshold=precision_score(y_test, y_pred_at_best, zero_division=0),
        recall_at_best_threshold=recall_score(y_test, y_pred_at_best, zero_division=0),
        n_train_tokens=len(y_train),
        n_test_tokens=len(y_test),
    )


def sweep_layers(train: list[Example], test: list[Example], n_layers: int, seed: int = 0) -> list[LayerResult]:
    return [evaluate_layer(train, test, layer_idx, seed=seed) for layer_idx in range(n_layers)]
