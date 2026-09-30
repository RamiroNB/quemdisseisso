import pytest

from ocarandu.data.publichearingbr import DEFAULT_NLI_PATH, load_nli_opinions

pytestmark = pytest.mark.skipif(
    not DEFAULT_NLI_PATH.exists(), reason="PublicHearingBR data lives outside the repo (shared cache), not always present"
)


def test_loads_opinions_with_labels():
    opinions = load_nli_opinions()
    assert len(opinions) > 4000
    assert all(isinstance(o.is_hallucination, bool) for o in opinions)
    # nearly all have exactly 4 context chunks; one known outlier has 0
    assert sum(1 for o in opinions if len(o.context_chunks) == 4) > len(opinions) - 5


def test_hallucination_rate_matches_known_figure():
    opinions = load_nli_opinions()
    rate = sum(1 for o in opinions if o.is_hallucination) / len(opinions)
    assert 0.10 < rate < 0.14  # ~11.9% observed
