from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from calico.errors import ConfigError
from calico.looks import BUILT_IN, discover_looks, load_looks

if TYPE_CHECKING:
    from pathlib import Path


def test_the_built_in_vocabulary_loads_and_describes_itself() -> None:
    looks = {look.name: look for look in load_looks(BUILT_IN)}
    golden = looks["golden hour"]
    assert "magic hour" in golden.words
    assert golden.signals == (("warmth", 1), ("saturation", 1))
    assert golden.describe() == "warmer, more saturated"
    assert looks["moody"].describe() == "darker, more muted"


def test_a_personal_file_overrides_a_look_by_name_and_adds_new_ones(tmp_path: Path) -> None:
    personal = tmp_path / "looks.toml"
    personal.write_text(
        '[looks."golden hour"]\nwords = ["golden hour"]\nsignals = { warmth = 1 }\n'
        '[looks.noir]\nwords = ["noir"]\nsignals = { contrast = 1, saturation = -1 }\n',
        encoding="utf-8",
    )
    looks = {look.name: look for look in discover_looks((BUILT_IN, personal, tmp_path / "absent"))}
    assert looks["golden hour"].signals == (("warmth", 1),)
    assert looks["noir"].describe() == "more contrast, more muted"
    assert "moody" in looks


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        ('[looks.x]\nwords = ["x"]\nsignals = { glow = 1 }\n', "glow"),
        ('[looks.x]\nwords = ["x"]\nsignals = { warmth = 2 }\n', r"\+1 or -1"),
        ("[looks.x]\nwords = []\nsignals = { warmth = 1 }\n", "words"),
        ('[looks.x]\nwords = [" "]\nsignals = { warmth = 1 }\n', "empty"),
        ('[looks.x]\nwords = ["x"]\nsignals = { warmth = true }\n', r"\+1 or -1"),
        ("[looks.x\n", "not valid TOML"),
    ],
)
def test_mistakes_in_a_vocabulary_say_how_to_fix_them(
    tmp_path: Path, body: str, problem: str
) -> None:
    path = tmp_path / "looks.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError, match=problem):
        load_looks(path)
