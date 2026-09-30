import importlib.util

import pytest

from ocarandu.data.sentence_labels import (
    Sentence,
    segment_sentences,
    sentence_evident_subtle_categories,
    sentence_gold_label,
)

needs_spacy_model = pytest.mark.skipif(importlib.util.find_spec("en_core_web_sm") is None,
                                       reason="spaCy model en_core_web_sm not installed")


@needs_spacy_model
def test_greeting_with_no_entity_is_filler():
    sentences = segment_sentences("Good morning. The report covers three main topics.")
    assert sentences[0].filler_label == "filler"
    assert sentences[1].filler_label == "content_bearing"


@needs_spacy_model
def test_greeting_with_entity_is_not_auto_filler():
    sentences = segment_sentences("Good morning, this is John's third award this year.")
    # contains a PERSON entity + numeral -> should not be blindly filler
    assert sentences[0].filler_label == "content_bearing"


@needs_spacy_model
def test_short_fragment_with_no_content_is_filler():
    sentences = segment_sentences("Sure, no problem. The company reported $10 million in Q3 revenue.")
    assert sentences[0].filler_label == "filler"
    assert sentences[1].filler_label == "content_bearing"


@needs_spacy_model
def test_sentence_with_numeral_is_content_bearing():
    sentences = segment_sentences("It happened in 1999.")
    assert sentences[0].filler_label == "content_bearing"


def test_sentence_gold_label_majority_overlap():
    # sentence spans chars 0-20; a span covering >=50% of itself inside the
    # sentence should mark it positive
    sent = Sentence(text="x" * 20, start=0, end=20, filler_label="content_bearing")
    span_inside = ((5, 15),)  # fully inside, 100% of span in sentence
    assert sentence_gold_label(sent, span_inside) == 1


def test_sentence_gold_label_ignores_barely_touching_span():
    # span mostly outside the sentence (only 1 of 10 chars land inside)
    sent = Sentence(text="x" * 20, start=0, end=20, filler_label="content_bearing")
    span_mostly_outside = ((19, 29),)  # only 1 char (19-20) inside a 10-char span
    assert sentence_gold_label(sent, span_mostly_outside) == 0


def test_sentence_gold_label_no_spans_is_faithful():
    sent = Sentence(text="x" * 20, start=0, end=20, filler_label="content_bearing")
    assert sentence_gold_label(sent, ()) == 0


def test_evident_subtle_categories_both_present():
    sent = Sentence(text="x" * 20, start=0, end=20, filler_label="content_bearing")
    typed_spans = ((0, 10, "Evident Conflict"), (10, 20, "Subtle Baseless Info"))
    assert sentence_evident_subtle_categories(sent, typed_spans) == {"Evident", "Subtle"}


def test_evident_subtle_categories_only_overlapping_span_counts():
    sent = Sentence(text="x" * 20, start=0, end=20, filler_label="content_bearing")
    typed_spans = ((0, 10, "Evident Conflict"), (25, 35, "Subtle Conflict"))
    assert sentence_evident_subtle_categories(sent, typed_spans) == {"Evident"}


def test_evident_subtle_categories_empty_when_no_overlap():
    sent = Sentence(text="x" * 20, start=0, end=20, filler_label="content_bearing")
    typed_spans = ((25, 35, "Evident Conflict"),)
    assert sentence_evident_subtle_categories(sent, typed_spans) == set()
