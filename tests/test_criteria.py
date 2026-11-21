from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

from calico.criteria import (
    SCORERS,
    BaselineScorer,
    CriteriaScores,
    PartsScorer,
    blend_examples,
    criteria_scorer,
    example_similarity,
    scorer_key,
)
from calico.errors import UnknownPluginError
from calico.looks import BUILT_IN, load_looks
from calico.selection.context import Trace
from calico.settings import Settings
from tests import factories as f

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray
    from PIL import Image

    from calico.domain import Photo
    from calico.selection.context import FeatureView


class Words:
    """A text encoder over a 3-word vocabulary that reads only the first four words."""

    name, version, dim = "words", "1", 3
    columns: tuple[str, ...] | None = None
    supports_text = True

    def extract(self, images: Sequence[Image.Image]) -> NDArray[np.float32]:
        raise NotImplementedError

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vocab = ["fog", "water", "stairs"]
        rows = [[float(w in t.split()) for w in vocab] for t in texts]
        return f.unit(rows)

    def readable_prefix(self, text: str) -> str:
        return " ".join(text.split()[:4])

    def match_probability(self, similarities: NDArray[np.float32]) -> NDArray[np.float32] | None:
        return None


class Blind(Words):
    name = "blind"
    supports_text = False


def ranking(result: CriteriaScores | None) -> NDArray[np.float32]:
    assert result is not None
    return result.scores


def pool() -> tuple[list[Photo], FeatureView]:
    photos = [f.photo(f"{i}.jpg") for i in range(3)]
    return photos, f.view([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_baseline_scores_each_photo_by_cosine_with_the_text() -> None:
    photos, v = pool()
    trace = Trace()
    scores = ranking(BaselineScorer(Words()).score("fog", photos, v, trace))
    assert scores is not None
    assert scores.dtype == np.float32
    np.testing.assert_allclose(scores, [1, 0, 0], atol=1e-6)
    assert trace.notes == []


def test_no_text_means_no_scores_and_no_note() -> None:
    photos, v = pool()
    trace = Trace()
    assert BaselineScorer(Words()).score("   ", photos, v, trace) is None
    assert trace.notes == []


def test_an_embedder_that_cannot_read_text_is_noted() -> None:
    photos, v = pool()
    trace = Trace()
    assert BaselineScorer(Blind()).score("fog", photos, v, trace) is None
    assert any("cannot read text" in n for n in trace.notes)


def test_text_beyond_what_the_encoder_reads_is_noted() -> None:
    photos, v = pool()
    trace = Trace()
    BaselineScorer(Words()).score("fog over still water at dawn", photos, v, trace)
    assert any("“fog over still water”" in n and "“at dawn”" in n for n in trace.notes)


def test_scorers_are_found_by_name_and_versioned() -> None:
    from pathlib import Path  # noqa: PLC0415

    assert "baseline" in SCORERS
    settings = Settings(home=Path("/unused"))
    scorer = criteria_scorer("baseline", Words(), settings)
    assert scorer_key(scorer) == "baseline@1 (words)"
    with pytest.raises(UnknownPluginError, match="baseline"):
        criteria_scorer("nope", Words(), settings)


# -- the parts scorer -------------------------------------------------------------------------


class Vocab(Words):
    """Encodes text as the vocabulary words it contains (reads everything)."""

    vocab = ("fog", "water", "stairs", "sky", "clouds")
    dim = 5

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        rows = [[float(w in t.split()) for w in self.vocab] for t in texts]
        return f.unit([r if any(r) else [1e-3] * len(self.vocab) for r in rows])

    def readable_prefix(self, text: str) -> str:
        return text


def parts(embedder: Words | None = None) -> PartsScorer:
    return PartsScorer(embedder or Vocab(), hemisphere="north")


def test_things_to_avoid_count_against_a_photo() -> None:
    photos = [f.photo(f"{i}.jpg") for i in range(3)]
    v = f.view([[0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])  # sky+clouds, sky, clouds
    trace = Trace()
    scores = ranking(parts().score("a sky without clouds", photos, v, trace))
    assert scores is not None
    assert list(np.argsort(-scores)) == [1, 0, 2]  # the clear sky first
    baseline = ranking(BaselineScorer(Vocab()).score("a sky without clouds", photos, v, Trace()))
    assert baseline is not None
    assert int(np.argmax(baseline)) == 0  # one blurred meaning: the cloudy sky wins
    assert any("without clouds" in n for n in trace.notes)


def test_when_ranks_photos_that_fit_first_and_undated_ones_between() -> None:
    photos = [
        f.photo("summer.jpg", captured_at=datetime(2021, 7, 1)),
        f.photo("winter.jpg", captured_at=datetime(2021, 1, 9)),
        f.photo("undated.jpg"),
    ]
    v = f.view(
        np.array([[1, 0, 0, 0, 0], [0.6, 0.8, 0, 0, 0], [0.6, 0.8, 0, 0, 0]])
    )  # summer: most fog
    trace = Trace()
    scores = ranking(parts().score("fog in winter", photos, v, trace))
    assert scores is not None
    assert list(np.argsort(-scores)) == [1, 2, 0]
    assert any("1 photo has no capture date" in n for n in trace.notes)


def test_where_ranks_photos_from_the_named_place_first() -> None:
    photos = [f.photo("a.jpg", location="Harbor"), f.photo("b.jpg", location="Old Mill")]
    v = f.view(np.array([[1, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0]]))
    scores = ranking(parts().score("fog at Old Mill", photos, v, Trace()))
    assert scores is not None
    assert int(np.argmax(scores)) == 1


def test_dates_and_places_still_work_when_the_text_cannot_be_read() -> None:
    photos = [
        f.photo("summer.jpg", captured_at=datetime(2021, 7, 1)),
        f.photo("winter.jpg", captured_at=datetime(2021, 1, 9)),
    ]
    trace = Trace()
    scores = ranking(parts(Blind()).score("fog in winter", photos, f.view(np.eye(2, 3)), trace))
    assert scores is not None
    assert int(np.argmax(scores)) == 1
    assert any("“fog” was ignored" in n for n in trace.notes)


def test_a_plain_description_scores_like_the_baseline_with_no_extra_note() -> None:
    photos, v = pool()
    trace = Trace()
    plain = ranking(parts(Words()).score("fog", photos, v, trace))
    baseline = ranking(BaselineScorer(Words()).score("fog", photos, v, Trace()))
    assert plain is not None
    assert baseline is not None
    np.testing.assert_allclose(plain, baseline, atol=1e-6)
    assert trace.notes == []


def test_parts_is_registered_and_takes_the_hemisphere_from_settings(calico_home: Path) -> None:
    settings = Settings(home=calico_home, hemisphere="south")
    scorer = criteria_scorer("parts", Vocab(), settings)
    assert scorer_key(scorer) == "parts@1 (words)"
    assert isinstance(scorer, PartsScorer)
    assert scorer.hemisphere == "south"


# -- looks: light, colour and mood measured, not read ----------------------------------------

TECH = ("brightness", "contrast", "warmth", "saturation")


def measured_view(embeddings: list[list[float]], tech: list[list[float]]) -> FeatureView:
    return f.view(np.array(embeddings), {"technical": np.array(tech)}, {"technical": TECH})


def with_looks(embedder: Words | None = None) -> PartsScorer:
    return PartsScorer(embedder or Vocab(), "north", tuple(load_looks(BUILT_IN)))


def test_a_look_alone_is_answered_by_measurement() -> None:
    photos = [f.photo(f"{i}.jpg") for i in range(3)]
    v = measured_view(
        [[1, 0, 0, 0, 0]] * 3,
        [[0.5, 0.3, 0.0, 0.1], [0.5, 0.3, 0.4, 0.6], [0.5, 0.3, 0.2, 0.3]],  # warm+saturated: 1
    )
    trace = Trace()
    scores = ranking(with_looks().score("golden hour", photos, v, trace))
    assert scores is not None
    assert list(np.argsort(-scores)) == [1, 2, 0]
    assert any("look golden hour (warmer, more saturated)" in n for n in trace.notes)


def test_a_look_and_a_subject_count_equally() -> None:
    photos = [f.photo(f"{i}.jpg") for i in range(3)]
    v = measured_view(
        [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]],  # stairs, stairs, fog
        [[0.5, 0.3, 0.0, 0.1], [0.5, 0.3, 0.4, 0.6], [0.5, 0.3, 0.5, 0.7]],
    )
    scores = ranking(with_looks().score("golden hour on stairs", photos, v, Trace()))
    assert scores is not None
    assert int(np.argmax(scores)) == 1  # the golden stairs beat the dull stairs and golden fog


def test_looks_work_without_a_text_reader() -> None:
    photos = [f.photo(f"{i}.jpg") for i in range(2)]
    v = measured_view([[1, 0, 0], [0, 1, 0]], [[0.8, 0.3, 0.1, 0.9], [0.1, 0.3, 0.1, 0.1]])
    trace = Trace()
    scores = ranking(with_looks(Blind()).score("moody stairs", photos, v, trace))
    assert scores is not None
    assert int(np.argmax(scores)) == 1  # the darker, more muted photo
    assert any("“stairs” was ignored" in n for n in trace.notes)


def test_without_measurements_a_look_is_noted_and_skipped() -> None:
    photos, v = pool()
    trace = Trace()
    scores = ranking(with_looks(Words()).score("moody fog", photos, v, trace))
    assert scores is not None
    assert int(np.argmax(scores)) == 0
    assert any("cannot be measured" in n for n in trace.notes)


def test_measured_looks_are_a_separate_scorer_not_the_default(calico_home: Path) -> None:
    settings = Settings(home=calico_home)
    assert settings.criteria_scorer == "parts"
    assert scorer_key(criteria_scorer("parts", Vocab(), settings)) == "parts@1 (words)"
    looks = criteria_scorer("parts-looks", Vocab(), settings)
    assert scorer_key(looks) == "parts-looks@1 (words)"
    assert isinstance(looks, PartsScorer)
    assert looks.looks


# -- how likely each photo matches (for "clear match") ----------------------------------------


class Calibrated(Vocab):
    """Says how likely a photo matches, as SigLIP does: sigmoid(10 * cosine - 5)."""

    def match_probability(self, similarities: NDArray[np.float32]) -> NDArray[np.float32] | None:
        return (1 / (1 + np.exp(-(10 * similarities - 5)))).astype(np.float32)


def match(result: CriteriaScores | None) -> NDArray[np.float32]:
    assert result is not None
    assert result.match is not None
    return result.match


def test_a_model_that_can_tell_says_how_likely_each_photo_matches() -> None:
    photos = [f.photo(f"{i}.jpg") for i in range(3)]
    v = f.view([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 1, 0, 0, 0]])  # fog, water, both
    result = BaselineScorer(Calibrated()).score("fog", photos, v, Trace())
    np.testing.assert_allclose(match(result), [0.993, 0.007, 0.888], atol=1e-3)
    assert result is not None
    assert result.clear_matches is not None
    assert result.clear_matches.tolist() == [True, False, True]


def test_a_model_that_only_ranks_cannot_say_what_clearly_matches() -> None:
    photos, v = pool()
    for scorer in (BaselineScorer(Words()), parts(Words())):
        result = scorer.score("fog", photos, v, Trace())
        assert result is not None
        assert result.match is None
        assert result.clear_matches is None


def test_a_thing_to_avoid_makes_a_photo_unlikely_to_match() -> None:
    photos = [f.photo(f"{i}.jpg") for i in range(3)]
    v = f.view([[0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])  # sky+clouds, sky, clouds
    result = parts(Calibrated()).score("a sky without clouds", photos, v, Trace())
    np.testing.assert_allclose(match(result), [0.1, 0.99, 0.0], atol=0.01)


def test_when_and_where_must_fit_for_a_clear_match() -> None:
    photos = [
        f.photo("summer.jpg", captured_at=datetime(2021, 7, 1)),
        f.photo("winter.jpg", captured_at=datetime(2021, 1, 9)),
        f.photo("undated.jpg"),
        f.photo("elsewhere.jpg", location="Harbor", captured_at=datetime(2021, 1, 9)),
    ]
    v = f.view([[1, 0, 0, 0, 0]] * 4)  # all fog
    result = parts(Calibrated()).score("fog at Old Mill in winter", photos, v, Trace())
    np.testing.assert_allclose(match(result), [0.0, 0.99, 0.5, 0.0], atol=0.01)
    assert result is not None
    assert result.clear_matches is not None
    assert result.clear_matches.tolist() == [False, True, False, False]  # unknown is not clear


def test_when_and_where_alone_can_clearly_match_without_reading_text() -> None:
    photos = [
        f.photo("summer.jpg", captured_at=datetime(2021, 7, 1)),
        f.photo("winter.jpg", captured_at=datetime(2021, 1, 9)),
    ]
    result = parts(Blind()).score("in winter", photos, f.view(np.eye(2, 3)), Trace())
    assert match(result).tolist() == [0.0, 1.0]


def test_text_that_cannot_be_read_leaves_clear_matches_unknown() -> None:
    photos = [f.photo("winter.jpg", captured_at=datetime(2021, 1, 9))]
    result = parts(Blind()).score("fog in winter", photos, f.view(np.eye(1, 3)), Trace())
    assert result is not None
    assert result.match is None


def test_measured_looks_cannot_say_what_clearly_matches() -> None:
    photos = [f.photo(f"{i}.jpg") for i in range(2)]
    v = measured_view([[1, 0, 0, 0, 0]] * 2, [[0.5, 0.3, 0.0, 0.1], [0.5, 0.3, 0.4, 0.6]])
    result = with_looks(Calibrated()).score("golden hour fog", photos, v, Trace())
    assert result is not None
    assert result.match is None


def test_the_describe_model_reads_its_own_vectors_when_they_are_not_the_main_embedding() -> None:
    """With SigLIP for "Describe it" and CLIP for the rest, the photos' SigLIP vectors are a
    feature of their own; the scorer reads those, not the CLIP embedding."""
    photos = [f.photo(f"{i}.jpg") for i in range(2)]
    v = f.view(
        [[1, 0], [0, 1]],  # the main embedding: some other model, other dimensions
        {"words": f.unit([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])},  # water, fog
    )
    result = BaselineScorer(Calibrated()).score("fog", photos, v, Trace())
    assert int(np.argmax(ranking(result))) == 1


# -- examples -------------------------------------------------------------------------------


def test_example_similarity_is_the_mean_cosine_to_the_examples() -> None:
    _, v = pool()
    examples = f.view([[1, 0, 0], [0, 1, 0]])
    assert example_similarity(Words(), v, examples).tolist() == pytest.approx([0.5, 0.5, 0.0])


def test_example_similarity_uses_the_readers_own_vectors() -> None:
    photos = [f.photo(f"{i}.jpg") for i in range(2)]
    v = f.view([[1, 0], [0, 1]], {"words": np.array([[0, 1, 0], [1, 0, 0]])})
    examples = f.view([[1, 0]], {"words": np.array([[1, 0, 0]])})
    assert len(photos) == len(v)
    assert example_similarity(Words(), v, examples).tolist() == pytest.approx([0.0, 1.0])


def test_examples_add_to_the_text_by_weight_and_keep_its_match() -> None:
    text = CriteriaScores(
        np.array([0.3, 0.1], dtype=np.float32), np.array([0.9, 0.1], dtype=np.float32)
    )
    similarity = np.array([0.2, 0.8], dtype=np.float32)
    blended = blend_examples(text, similarity, weight=0.5)
    assert blended.scores.tolist() == pytest.approx([0.4, 0.5])
    assert blended.match is not None
    assert blended.match.tolist() == pytest.approx([0.9, 0.1])


def test_examples_alone_rank_by_similarity_and_cannot_say_what_matches() -> None:
    similarity = np.array([0.2, 0.8], dtype=np.float32)
    alone = blend_examples(None, similarity, weight=0.5)
    assert alone.scores.tolist() == pytest.approx([0.2, 0.8])
    assert alone.match is None


def test_a_description_that_reads_to_nothing_says_it_was_ignored() -> None:
    photos = [f.photo(f"{i}.jpg") for i in range(2)]
    v = f.view([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    trace = Trace()
    assert parts().score(".", photos, v, trace) is None
    assert any("nothing Calico can look for" in n for n in trace.notes)
