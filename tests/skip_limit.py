"""Fail a run in which more tests skip than ``CALICO_MAX_SKIPPED`` (A09-3).

A skip is the right answer on a laptop without Chrome, so there is no limit unless CI sets
one: each workflow states how many tests it expects to skip on its machine, and losing
PyTorch or Chrome there fails the run instead of quietly testing less.
"""

from __future__ import annotations

import os

import pytest

MAX_SKIPPED_ENV = "CALICO_MAX_SKIPPED"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    limit = os.environ.get(MAX_SKIPPED_ENV)
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if limit is None or reporter is None or exitstatus != pytest.ExitCode.OK:
        return
    skipped = len(reporter.stats.get("skipped", []))
    if skipped > int(limit):
        reporter.write_line("")  # off the progress line
        reporter.write_sep(
            "=",
            f"{skipped} tests skipped, more than {MAX_SKIPPED_ENV}={limit}: the SKIPPED lines "
            "say why; if a new skip is meant, raise the limit in the workflow that sets it",
            red=True,
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
