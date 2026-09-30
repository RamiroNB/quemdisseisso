"""Sentence-level (not token-level) hallucination labeling and a
content-bearing vs. filler classifier, both for RAGTruth (English).

Token-level labels are noisy for single function-word tokens (a lone
"the" carries no hallucination signal by itself); this operates one
spaCy sentence at a time instead, pooling that sentence's token
activations into one vector for the probe. Needs spaCy's en_core_web_sm.
"""

import re
from dataclasses import dataclass

import spacy

from .span_labels import token_overlaps_span

_NLP = None

# English discourse markers -- Rule 1 of the filler classifier (English only).
DISCOURSE_MARKERS = [
    "good morning",
    "good afternoon",
    "good evening",
    "hello",
    "hi there",
    "thank you",
    "thanks for",
    "let's begin",
    "let's start",
    "let's move on",
    "moving on",
    "in conclusion",
    "to summarize",
    "let's talk about",
    "i'd like to",
    "i would like to",
    "welcome to",
    "welcome back",
]

_ENTITY_LABELS_FOR_CONTENT = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "PERCENT", "CARDINAL", "ORDINAL"}
_PAST_TENSE_TAGS = {"VBD", "VBN"}
_NUMERAL_RE = re.compile(r"\d")


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int  # char offset into the response
    end: int
    filler_label: str  # "content_bearing" or "filler"


def _strip_punct_lower(text: str) -> str:
    return re.sub(r"[^\w\s']", "", text).strip().lower()


def classify_filler(sent_doc) -> str:
    """Rules applied in order; first match wins. `sent_doc` is a spaCy Span
    (one sentence) from a parsed Doc, so NER/POS tags are already available.
    """
    text_norm = _strip_punct_lower(sent_doc.text)

    # Rule 1: discourse marker, only if the sentence carries no entity/numeral
    starts_with_marker = any(text_norm == m or text_norm.startswith(m) for m in DISCOURSE_MARKERS)
    has_entity_or_numeral = bool(sent_doc.ents) or bool(_NUMERAL_RE.search(sent_doc.text))
    if starts_with_marker and not has_entity_or_numeral:
        return "filler"

    # Rule 2: factual content check
    if any(ent.label_ in _ENTITY_LABELS_FOR_CONTENT for ent in sent_doc.ents):
        return "content_bearing"
    if _NUMERAL_RE.search(sent_doc.text):
        return "content_bearing"
    for token in sent_doc:
        if token.tag_ in _PAST_TENSE_TAGS:
            subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
            if any(subj.ent_type_ or subj.pos_ == "PRON" for subj in subjects):
                return "content_bearing"

    # Rule 3: length fallback
    n_tokens = sum(1 for t in sent_doc if not t.is_punct)
    if n_tokens < 6:
        return "filler"

    # Rule 4: default
    return "content_bearing"


def segment_sentences(response: str) -> list[Sentence]:
    """spaCy-sentence-segment `response`, classifying each sentence
    content_bearing/filler in the same pass (NER/POS already computed)."""
    doc = get_nlp()(response)
    sentences = []
    for sent in doc.sents:
        label = classify_filler(sent)
        sentences.append(Sentence(text=sent.text, start=sent.start_char, end=sent.end_char, filler_label=label))
    return sentences


def sentence_gold_label(sentence: Sentence, spans: tuple[tuple[int, int], ...], min_overlap_frac: float = 0.5) -> int:
    """A sentence is hallucinated (1) if at least `min_overlap_frac` (default
    half) of some span's own characters fall inside it. Reuses
    token_overlaps_span with the span playing the "token" role and the
    sentence playing the "span" role, so a span that merely touches a
    sentence boundary doesn't count, and a span split across two sentences
    gets assigned to whichever one holds most of it."""
    for span_start, span_end in spans:
        if token_overlaps_span(span_start, span_end, sentence.start, sentence.end, min_overlap_frac):
            return 1
    return 0


def sentence_evident_subtle_categories(
    sentence: Sentence,
    typed_spans: tuple[tuple[int, int, str], ...],
    min_overlap_frac: float = 0.5,
) -> set[str]:
    """Which of RAGTruth's Evident/Subtle categories (its `label_type`
    field, e.g. "Evident Conflict", "Subtle Baseless Info") triggered this
    sentence's positive label, using the same majority-overlap rule as
    `sentence_gold_label`. `typed_spans` is (start, end, label_type).
    Returns a subset of {"Evident", "Subtle"} (empty if no span overlaps).
    """
    categories = set()
    for span_start, span_end, label_type in typed_spans:
        if token_overlaps_span(span_start, span_end, sentence.start, sentence.end, min_overlap_frac):
            if label_type.startswith("Evident"):
                categories.add("Evident")
            elif label_type.startswith("Subtle"):
                categories.add("Subtle")
    return categories
