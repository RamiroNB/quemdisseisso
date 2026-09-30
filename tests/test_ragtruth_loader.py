import pytest

from ocarandu.data.ragtruth import DEFAULT_DATASET_DIR, load_summarization_subset

pytestmark = pytest.mark.skipif(not DEFAULT_DATASET_DIR.exists(), reason="RAGTruth not cloned into vendor/ragtruth")


def test_loads_only_summary_task_type():
    examples = load_summarization_subset()
    assert len(examples) > 0
    assert all(ex.source_info for ex in examples)


def test_preserves_official_split():
    examples = load_summarization_subset()
    splits = {ex.split for ex in examples}
    assert splits <= {"train", "test"}
    assert "train" in splits and "test" in splits


def test_model_filter():
    examples = load_summarization_subset(model="llama-2-7b-chat")
    assert len(examples) > 0
    assert all(ex.model == "llama-2-7b-chat" for ex in examples)


def test_spans_have_expected_fields():
    examples = load_summarization_subset(model="mistral-7B-instruct")
    labeled = [ex for ex in examples if ex.spans]
    assert labeled, "expected at least one hallucinated example"
    span = labeled[0].spans[0]
    assert span.end > span.start
    assert span.text
