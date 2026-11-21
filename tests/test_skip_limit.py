"""A run fails when more tests skip than CI expects (A09-3).

The ML tests skip without PyTorch and the browser tests without Chrome, so a CI image that
lost either would pass with whole layers untested. CI states how many skips it expects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.skip_limit import MAX_SKIPPED_ENV

if TYPE_CHECKING:
    import pytest

TWO_SKIPS = """
import pytest

def test_runs(): pass

@pytest.mark.skip(reason="first")
def test_first(): pass

@pytest.mark.skip(reason="second")
def test_second(): pass
"""


def run(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, limit: str | None
) -> pytest.RunResult:
    if limit is None:
        monkeypatch.delenv(MAX_SKIPPED_ENV, raising=False)
    else:
        monkeypatch.setenv(MAX_SKIPPED_ENV, limit)
    pytester.makepyfile(TWO_SKIPS)
    return pytester.runpytest("-p", "tests.skip_limit")


def test_more_skips_than_expected_fail_the_run_and_say_how_many(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = run(pytester, monkeypatch, "1")
    assert result.ret == 1
    result.stdout.fnmatch_lines([f"*2 tests skipped, more than {MAX_SKIPPED_ENV}=1*"])


def test_as_many_skips_as_expected_pass(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert run(pytester, monkeypatch, "2").ret == 0


def test_without_a_limit_any_number_may_skip(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a laptop missing Chrome, skipping is the right answer."""
    assert run(pytester, monkeypatch, None).ret == 0
