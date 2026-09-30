"""Convert RAGTruth character-offset hallucination spans into per-token labels."""

from .ragtruth import RagTruthExample

# A token counts as "in" a span only if the span covers at least this
# fraction of the token's characters -- a bare touch/overlap-by-one-char
# (e.g. a span boundary landing mid-token, or a token that's mostly
# punctuation straddling the edge) shouldn't flip the whole token positive.
DEFAULT_MIN_OVERLAP_FRAC = 0.5


def token_overlaps_span(
    tok_start: int, tok_end: int, span_start: int, span_end: int, min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC
) -> bool:
    overlap = min(tok_end, span_end) - max(tok_start, span_start)
    if overlap <= 0:
        return False
    tok_len = tok_end - tok_start
    if tok_len <= 0:
        return False
    return (overlap / tok_len) >= min_overlap_frac


def token_labels_for_response(
    example: RagTruthExample, tokenizer, min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC
) -> list[int]:
    """Return one binary label per token of `example.response` (1 = hallucinated).

    A token is labeled hallucinated only if a hallucination span covers at
    least `min_overlap_frac` of the token's characters (not just any touch
    at the boundary -- see `token_overlaps_span`). Requires a fast
    tokenizer (one that supports `return_offsets_mapping`).
    """
    encoding = tokenizer(example.response, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    labels = [0] * len(offsets)
    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        for span in example.spans:
            if token_overlaps_span(tok_start, tok_end, span.start, span.end, min_overlap_frac):
                labels[i] = 1
                break
    return labels
