"""Tasks: the things a user can ask for. Each entry in the UI's menu is one task.

A task is a class with a name, a description, an options model, and a ``run``
method that turns a ``TaskContext`` into groups of picks. The options model is
a pydantic model whose fields carry ``description``s and bounds; the CLI and the
web form are generated from it, so adding a task never means editing the UI.

To add a task: subclass ``Task``, register it under ``[project.entry-points.
"calico.tasks"]`` in ``pyproject.toml``, and write its tests against a
hand-built ``TaskContext`` (see ``tests/factories.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

from calico.errors import InvalidRequestError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from numpy.typing import NDArray

    from calico.domain import Group
    from calico.selection.context import TaskContext


RETIRED_OPTIONS = {
    "criteria_weight": "How much the description counts is now one setting for every task: "
    "describe_weight on the request, from 0 to 1 (--describe-weight in the CLI, the slider "
    "under “Describe it” on the web).",
}
"""Options that were removed, and what to use instead, so old scripts get a way forward."""


class TaskOptions(BaseModel):
    """Base for every task's options. Unknown keys are an error, not ignored."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ReferenceUse(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    UNUSED = "unused"


class Task[O: TaskOptions](ABC):
    name: ClassVar[str]
    title: ClassVar[str]
    description: ClassVar[str]
    reference_use: ClassVar[ReferenceUse] = ReferenceUse.UNUSED
    default_limit: ClassVar[int] = 12
    """Results returned when the request does not say how many."""
    default_describe_weight: ClassVar[float] = 0.5
    """The description's share of each photo's score when the request does not say."""
    uses_taste: ClassVar[bool] = False
    """Whether the template (General or Yours) can change what this task picks: true for
    tasks that rank by strength (``selection.strength``), which reads the aesthetic."""
    reference_label: ClassVar[str] = ""
    """What the UI calls the reference photo for this task ("" = no reference input)."""
    reference_help: ClassVar[str] = ""
    options_model: ClassVar[type[TaskOptions]]

    def parse_options(self, raw: Mapping[str, Any]) -> O:
        for key, hint in RETIRED_OPTIONS.items():
            if key in raw:
                raise InvalidRequestError(f"{key!r} is no longer a task option. {hint}")
        try:
            return self.options_model.model_validate(dict(raw))  # type: ignore[return-value]
        except ValidationError as exc:
            raise InvalidRequestError(f"Invalid options for task {self.name!r}:\n{exc}") from exc

    def describe_weight(self, ctx: TaskContext) -> float:
        """How much the description counts: the request's share, else this task's default."""
        weight = ctx.request.describe_weight
        return self.default_describe_weight if weight is None else weight

    def patch_shortlist(self, ctx: TaskContext, options: O) -> NDArray[np.intp] | None:
        """Photos (indices into ``ctx.photos``) to compare with the references part by
        part before ``run``; the engine fills ``ctx.patch_scores`` for them. ``None`` (the
        default) asks for nothing: it costs about 0.1 s a photo."""
        return None

    def patch_pairs(self, ctx: TaskContext, options: O) -> NDArray[np.intp] | None:
        """Pairs of photos (rows of two indices into ``ctx.photos``) to compare with each
        other part by part before ``run``; the engine fills ``ctx.pair_shares`` for them and
        remembers them. ``None`` (the default) asks for nothing: a photo not yet compared
        costs about 0.2 s, and each pair 4 ms."""
        return None

    @abstractmethod
    def run(self, ctx: TaskContext, options: O) -> list[Group]:
        """Select from ``ctx.photos``.

        ``ctx.limit`` bounds the number of *results*; each task's
        docstring says what one result is (a photo for ``best``, a story of three
        for ``triptych``).
        """
