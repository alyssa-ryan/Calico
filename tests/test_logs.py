from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest

from calico.logs import configure_logging

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _restore_logger() -> Iterator[None]:
    logger = logging.getLogger("calico")
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def emit_all() -> None:
    log = logging.getLogger("calico.test")
    log.debug("d-msg")
    log.info("i-msg")
    log.warning("w-msg")


def test_default_level_shows_warnings_only(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(0, json_lines=False)
    emit_all()
    err = capsys.readouterr().err
    assert "w-msg" in err
    assert "i-msg" not in err
    assert capsys.readouterr().out == ""


def test_single_v_adds_info(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(1, json_lines=False)
    emit_all()
    err = capsys.readouterr().err
    assert "i-msg" in err
    assert "d-msg" not in err


def test_double_v_adds_debug(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(2, json_lines=False)
    emit_all()
    assert "d-msg" in capsys.readouterr().err


def test_json_lines_are_one_object_per_line_with_extras(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(1, json_lines=True)
    logging.getLogger("calico.test").info("indexed %d files", 3, extra={"folder": "/a", "n": 3})
    (line,) = capsys.readouterr().err.splitlines()
    record = json.loads(line)
    assert record["msg"] == "indexed 3 files"
    assert record["level"] == "INFO"
    assert record["logger"] == "calico.test"
    assert record["folder"] == "/a"
    assert record["n"] == 3
    assert "ts" in record


def test_json_includes_exception_text(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(0, json_lines=True)
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("calico.test").exception("failed")
    record = json.loads(capsys.readouterr().err.splitlines()[0])
    assert "ValueError: boom" in record["exc"]


def test_json_survives_unserialisable_extras(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(0, json_lines=True)
    logging.getLogger("calico.test").warning("odd", extra={"thing": object()})
    assert json.loads(capsys.readouterr().err)["msg"] == "odd"


def test_reconfiguring_does_not_duplicate_output(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(0, json_lines=False)
    configure_logging(0, json_lines=False)
    logging.getLogger("calico.test").warning("once")
    assert capsys.readouterr().err.count("once") == 1


def test_a_library_configuring_the_root_logger_does_not_duplicate_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Some ML libraries call ``logging.basicConfig()``, which gives the root logger a handler."""
    root = logging.getLogger()
    stray = logging.StreamHandler()  # writes to stderr, like basicConfig's handler
    root.addHandler(stray)
    try:
        configure_logging(1, json_lines=False)
        logging.getLogger("calico.test").info("once")
    finally:
        root.removeHandler(stray)
    assert capsys.readouterr().err.count("once") == 1


def test_noisy_libraries_are_quiet_unless_very_verbose() -> None:
    configure_logging(1, json_lines=False)
    assert logging.getLogger("huggingface_hub").level == logging.ERROR
    assert logging.getLogger("urllib3").level == logging.ERROR
    configure_logging(2, json_lines=False)
    assert logging.getLogger("huggingface_hub").level == logging.NOTSET
    assert logging.getLogger("urllib3").level == logging.NOTSET
