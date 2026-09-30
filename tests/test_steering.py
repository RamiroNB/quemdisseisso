import numpy as np

import pytest

from ocarandu.probing.steering import (
    Example,
    best_f1_threshold,
    evaluate_layer,
    mean_diff_vector,
    split_by_example,
    split_by_official,
    sweep_layers,
)


def _synthetic_examples(n_examples=20, n_layers=3, hidden=8, tokens_per_example=10, seed=0):
    """Faithful tokens cluster near 0, hallucinated tokens near +5 in one
    layer, so the mean-diff/probe should trivially separate them."""
    rng = np.random.default_rng(seed)
    examples = []
    for i in range(n_examples):
        labels = rng.integers(0, 2, size=tokens_per_example).astype(np.int8)
        activations = rng.normal(0, 0.1, size=(n_layers, tokens_per_example, hidden)).astype(np.float32)
        for layer in range(n_layers):
            activations[layer, labels == 1] += 5.0
        examples.append(Example(source_id=f"s{i}", activations=activations, labels=labels))
    return examples


def test_split_by_example_no_leakage():
    examples = _synthetic_examples(n_examples=10)
    train, test = split_by_example(examples, test_frac=0.3, seed=1)
    train_ids = {ex.source_id for ex in train}
    test_ids = {ex.source_id for ex in test}
    assert train_ids.isdisjoint(test_ids)
    assert len(train) + len(test) == 10


def test_split_by_official_uses_ragtruth_split_field():
    examples = _synthetic_examples(n_examples=4)
    examples[0].split = "train"
    examples[1].split = "train"
    examples[2].split = "test"
    examples[3].split = "test"
    train, test = split_by_official(examples)
    assert {ex.source_id for ex in train} == {"s0", "s1"}
    assert {ex.source_id for ex in test} == {"s2", "s3"}


def test_split_by_official_rejects_missing_split():
    examples = _synthetic_examples(n_examples=2)
    examples[0].split = "train"
    # examples[1].split left as default ""
    with pytest.raises(ValueError):
        split_by_official(examples)


def test_mean_diff_vector_points_toward_faithful():
    examples = _synthetic_examples(n_examples=20)
    v = mean_diff_vector(examples, layer_idx=0)
    # faithful ~0, hallucinated ~5 -> v = faithful - hallucinated should be very negative
    assert (v < -3).all()


def test_evaluate_layer_separates_easy_synthetic_data():
    examples = _synthetic_examples(n_examples=40)
    train, test = split_by_example(examples, test_frac=0.25, seed=2)
    result = evaluate_layer(train, test, layer_idx=0)
    assert result.f1_default > 0.9
    assert result.accuracy > 0.9
    assert result.f1_best_threshold >= result.f1_default
    assert result.pr_auc > 0.9
    assert result.precision_at_best_threshold > 0.9
    assert result.recall_at_best_threshold > 0.9


def test_best_f1_threshold_beats_default_on_imbalanced_scores():
    rng = np.random.default_rng(7)
    # 950 faithful (low score), 50 hallucinated (high score) -- imbalanced,
    # perfectly separable, but a naive 0.5 threshold on raw scores would
    # still work here; use scores centered so 0.5 default cutoff is a poor
    # separator to exercise the search logic instead.
    y = np.array([0] * 950 + [1] * 50)
    proba = np.concatenate([rng.uniform(0.3, 0.6, 950), rng.uniform(0.55, 0.9, 50)])
    f1, threshold = best_f1_threshold(y, proba)
    default_f1 = f1_score_at(y, proba, 0.5)
    assert f1 >= default_f1


def f1_score_at(y_true, y_proba, threshold):
    from sklearn.metrics import f1_score

    return f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)


def test_sweep_layers_returns_one_result_per_layer():
    examples = _synthetic_examples(n_examples=20, n_layers=4)
    train, test = split_by_example(examples, test_frac=0.3, seed=3)
    results = sweep_layers(train, test, n_layers=4)
    assert len(results) == 4
    assert [r.layer for r in results] == [0, 1, 2, 3]
