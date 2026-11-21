from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from calico.domain import MAX_DESCRIPTION, SelectionRequest


def test_a_description_longer_than_any_model_reads_is_refused_with_why() -> None:
    """Pasted by mistake, 60,000 characters held the one web worker for most of a minute;
    the image model reads about 77 words."""
    SelectionRequest(task="best", folders=(Path("/a"),), criteria="x" * MAX_DESCRIPTION)
    with pytest.raises(ValidationError, match=f"{MAX_DESCRIPTION} characters"):
        SelectionRequest(task="best", folders=(Path("/a"),), criteria="x" * (MAX_DESCRIPTION + 1))
