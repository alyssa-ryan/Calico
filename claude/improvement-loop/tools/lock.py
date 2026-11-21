from __future__ import annotations

import ast
import hashlib
from typing import TYPE_CHECKING

from guard import LOOP, REPO, Refused, git

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

MEASURE_FILES = ("tools/summarize.py", "tools/calibrate_repeats.py")
HARNESS = "tools/harness.py"
MEASURES_IN_HARNESS = (
    "Grid",
    "null_configs",
    "Jittered",
    "Nulled",
    "low_priority_words",
    "is_low_priority",
    "Labels",
    "fold_map",
    "window",
    "pack",
    "same_subject",
    "repeat_share",
    "random_repeat",
    "auc",
    "burst_mates",
    "strongest_eval",
    "unfit_located",
    "story_eval",
    "set_measures",
    "random_sets",
    "evaluate",
)
BEFORE_LOCK = "278be2922eb16be33f31d93303134db1e68f2c4e3f1bf55d7a360588b3fe7b45"


def digest(read: Callable[[str], str]) -> str:
    parts = [f"{name}\n{read(name)}" for name in MEASURE_FILES]
    text = read(HARNESS)
    found = {
        node.name: ast.get_source_segment(text, node) or ""
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }
    missing = [name for name in MEASURES_IN_HARNESS if name not in found]
    if missing:
        raise Refused(f"{HARNESS} no longer defines {missing}, which tools/lock.py names")
    parts += [f"{name}\n{found[name]}" for name in MEASURES_IN_HARNESS]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def repo_path(name: str) -> str:
    return (LOOP / name).relative_to(REPO).as_posix()


def at(commit: str) -> str:
    return digest(lambda name: git("show", f"{commit}:{repo_path(name)}"))


def now() -> str:
    return digest(lambda name: (LOOP / name).read_text())


def with_files(commit: str, chosen: Iterable[str]) -> str:
    picked = set(chosen)

    def read(name: str) -> str:
        path = repo_path(name)
        if path in picked:
            return (REPO / path).read_text()
        return git("show", f"{commit}:{path}")

    return digest(read)


def refuse_change(before: str, after: str, open_attempts: list[str]) -> None:
    if before != after and open_attempts:
        raise Refused(
            f"this changes the measures (the files and harness functions tools/lock.py names) "
            f"while {open_attempts} are open: decide or record them first. A candidate's "
            "option is built and committed before its attempt is registered"
        )


def refuse_stale(
    *, registered: str | None, ladder: str | None, current: str, track: str, tier: str
) -> None:
    if registered is not None and registered != current:
        raise Refused(
            "the measures changed since this attempt was registered; it cannot be judged by "
            "them. Close it (`loop record`) and register again"
        )
    if (ladder or BEFORE_LOCK) != current:
        flag = "" if tier == "validation" else f" --tier {tier}"
        raise Refused(
            f"{track}'s {tier} null ladder was measured with other measures: "
            f"run `loop null {track}{flag}` first"
        )
