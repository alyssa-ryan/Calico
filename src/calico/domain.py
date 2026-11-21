"""The vocabulary of the tool: photos, requests, and results.

Two rules keep this module trustworthy:

* Nothing here performs I/O. These are values.
* Anything that crosses a process boundary (CLI JSON, web API, run records on
  disk) is a frozen pydantic model, so it validates on the way in and
  serialises losslessly on the way out. ``Photo`` is a plain frozen dataclass
  because it is created in bulk on hot paths and never parsed from user input.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

MAX_DESCRIPTION = 500
"""The longest "Describe it": the image model reads about 77 words, and a description
pasted by mistake must not hold the one web worker for a minute."""


def place_key(location: str) -> str:
    """What makes two location names the same place: case, spacing and Unicode form (a
    name pasted from Finder is often decomposed) do not count."""
    return " ".join(unicodedata.normalize("NFC", location).split()).casefold()


@dataclass(frozen=True, slots=True)
class Photo:
    """One image file in the catalog.

    ``path`` identifies the file; ``content_key`` identifies its bytes, so two
    copies of the same image share a key (and their computed features).
    ``location`` is the place the photo belongs to, derived from its folder by
    the collection's rules (see ``calico.catalog.locations``).
    """

    path: Path
    content_key: str
    collection: str
    location: str
    captured_at: datetime | None
    width: int
    height: int
    camera: str | None = None
    utc_offset: int | None = None
    """Minutes east of UTC that ``captured_at`` was recorded in, when the camera wrote
    its time zone; ``captured_at`` stays as the camera's clock showed it."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SelectionRequest(_Frozen):
    """What the user asked for. Identical for every entry point (CLI, web, Python)."""

    task: str = Field(description="Registered task name, e.g. 'best' or 'triptych'.")
    folders: Annotated[tuple[Path, ...], Field(min_length=1)] = Field(
        description="The pool: every catalogued photo under any of these folders."
    )
    criteria: str = Field(
        default="",
        description=f"Free-text description of what to look for (at most {MAX_DESCRIPTION} "
        "characters).",
    )
    only_clear_matches: bool = Field(
        default=False,
        description="Consider only photos that clearly match the criteria. If none do, "
        "all photos are considered and the run says so.",
    )
    describe_weight: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="The description's share of each photo's score, from 0 (ignored) to 1 "
        "(all that counts); unset means the task's own default.",
    )
    template: Literal["general", "yours"] | None = Field(
        default=None,
        description="Whose taste ranks photos: 'general' (the zero-shot aesthetic) or 'yours' "
        "(learned from your ratings); unset means the template in calico.toml.",
    )
    examples: tuple[Path, ...] = Field(
        default=(),
        description="Photos in your collections that show what you mean: part of the "
        "description, left out of the results.",
    )
    references: tuple[Path, ...] = Field(
        default=(), description="Reference images; need not be in the catalog."
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description="How many results to return; unset means the task's own default.",
    )
    options: dict[str, JsonValue] = Field(
        default_factory=dict, description="Task-specific options, validated by the task."
    )

    @field_validator("criteria")
    @classmethod
    def _readable_length(cls, criteria: str) -> str:
        if len(criteria) > MAX_DESCRIPTION:
            raise ValueError(
                f"the description is {len(criteria)} characters; Calico reads at most "
                f"{MAX_DESCRIPTION} characters (the image model reads about 77 words), so "
                "shorten it"
            )
        return criteria


class Evidence(_Frozen):
    """One reason a pick scored as it did.

    ``value`` is the signal in [0, 1]; ``weight`` is how much it counted. The
    pick's score is the sum of ``value * weight`` over its evidence, so the
    evidence list *is* the explanation, not a narrative written beside it.
    """

    signal: str
    value: float
    weight: float
    detail: str = ""

    @property
    def contribution(self) -> float:
        return self.value * self.weight


class Pick(_Frozen):
    """One selected photo, with why it was selected."""

    path: Path
    content_key: str
    location: str
    captured_at: datetime | None
    score: float
    evidence: tuple[Evidence, ...] = ()
    burst_size: int = Field(
        default=1, ge=1, description="Frames this pick stands for: itself and its alternates."
    )
    alternates: tuple[Path, ...] = Field(
        default=(),
        description="Other frames of the pick's burst, or of its view from other occasions, "
        "best first.",
    )
    clear_match: bool | None = Field(
        default=None,
        description="Whether the photo clearly matches the criteria; null when there are "
        "none, or the model cannot tell.",
    )
    probability: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="How likely you are to like it, from what you said at places your taste "
        "never learned (Yours only); null for General, for a taste not yet calibrated, and "
        "at a place your taste learned from.",
    )


class Marker(_Frozen):
    """Where a list stops being good: the line Yours draws above the first pick you are less
    likely than ``probability`` to like."""

    before: int = Field(
        ge=0,
        description="The picks before this one are at least that likely to be liked and this "
        "one is the first that is not (the number of picks when every one is).",
    )
    probability: float = Field(ge=0, le=1, description="The line: how likely a pick must be.")


class Group(_Frozen):
    """An ordered set of picks shown together.

    ``best`` returns one group; ``same_view`` one group per matched view;
    ``triptych`` one group of three per story.
    """

    title: str
    score: float
    picks: tuple[Pick, ...]
    evidence: tuple[Evidence, ...] = ()
    marker: Marker | None = Field(
        default=None,
        description="Where the list stops being good. Null when picks carry no probability, "
        "or a description reordered them.",
    )


class StageStat(_Frozen):
    """How one pipeline stage changed the candidate count, and how long it took."""

    name: str
    count_in: int
    count_out: int
    seconds: float


class SelectionResult(_Frozen):
    """Everything a request produced. Also the payload of a saved run."""

    run_id: str
    created_at: datetime
    request: SelectionRequest
    groups: tuple[Group, ...]
    example_keys: tuple[str, ...] = Field(
        default=(), description="Content keys of the request's examples, in the same order."
    )
    notes: tuple[str, ...] = Field(
        default=(), description="Things the user should know: fallbacks, skipped files."
    )
    stages: tuple[StageStat, ...] = ()
    versions: dict[str, str] = Field(
        default_factory=dict, description="calico + every extractor/model version used."
    )
