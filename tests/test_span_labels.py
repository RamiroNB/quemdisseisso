from ocarandu.data.ragtruth import HallucinationSpan, RagTruthExample
from ocarandu.data.span_labels import token_labels_for_response, token_overlaps_span


class FakeWhitespaceTokenizer:
    """Minimal stand-in for a fast HF tokenizer's offset mapping.

    Splits on whitespace and reports each token's (start, end) char span,
    so tests don't depend on downloading a real tokenizer.
    """

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        offsets = []
        pos = 0
        for word in text.split(" "):
            start = text.index(word, pos)
            end = start + len(word)
            offsets.append((start, end))
            pos = end
        return {"offset_mapping": offsets}


def _example(response: str, spans: tuple[HallucinationSpan, ...]) -> RagTruthExample:
    return RagTruthExample(
        source_id="s1",
        response_id="r1",
        model="test-model",
        split="test",
        source_info="irrelevant source text",
        prompt="summarize",
        response=response,
        spans=spans,
    )


def test_no_spans_all_faithful():
    ex = _example("the cat sat on the mat", spans=())
    labels = token_labels_for_response(ex, FakeWhitespaceTokenizer())
    assert labels == [0] * 6


def test_span_marks_overlapping_tokens_only():
    response = "the cat sat on the mat"
    # "sat on" -> chars 8-14
    span = HallucinationSpan(start=8, end=14, text="sat on", label_type="Evident Conflict")
    ex = _example(response, spans=(span,))
    labels = token_labels_for_response(ex, FakeWhitespaceTokenizer())
    # tokens: the(0-3) cat(4-7) sat(8-11) on(12-14) the(15-18) mat(19-22)
    assert labels == [0, 0, 1, 1, 0, 0]


def test_majority_overlap_labels_token():
    response = "hello world"
    # "world" spans 6-11 (5 chars); span 7-10 covers "orl" (3/5 = 60%) -- majority overlap
    span = HallucinationSpan(start=7, end=10, text="wor", label_type="Evident Conflict")
    ex = _example(response, spans=(span,))
    labels = token_labels_for_response(ex, FakeWhitespaceTokenizer())
    assert labels == [0, 1]


def test_boundary_touch_does_not_label_token():
    response = "hello world"
    # "world" spans 6-11 (5 chars); span 10-11 covers only "d" (1/5 = 20%) --
    # a bare boundary touch, should NOT flip the whole token positive
    span = HallucinationSpan(start=10, end=11, text="d", label_type="Evident Conflict")
    ex = _example(response, spans=(span,))
    labels = token_labels_for_response(ex, FakeWhitespaceTokenizer())
    assert labels == [0, 0]


def test_token_overlaps_span_threshold():
    # token spans 0-10 (len 10); overlap of 4 chars = 40%, below default 0.5
    assert not token_overlaps_span(0, 10, 6, 20)
    # overlap of 6 chars = 60%, above default 0.5
    assert token_overlaps_span(0, 10, 4, 20)
    # no overlap at all
    assert not token_overlaps_span(0, 10, 10, 20)
