from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import factories as f

TOOLS = Path(__file__).resolve().parents[1] / "claude" / "improvement-loop" / "tools"
sys.path.insert(0, str(TOOLS))

import guard  # noqa: E402
import ledger  # noqa: E402
import list_answers  # noqa: E402
import loop  # noqa: E402
import splits  # noqa: E402
import verdict  # noqa: E402

TRACK = "strongest-yours"


@pytest.fixture
def books(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name, file in (
        ("REGISTRATIONS", "registrations.jsonl"),
        ("LEDGER", "ledger.jsonl"),
        ("LOOKS", "looks.jsonl"),
        ("BARS", "bars.jsonl"),
    ):
        monkeypatch.setattr(ledger, name, tmp_path / file)
    monkeypatch.setattr(ledger, "SPLITS", tmp_path / "splits")
    return tmp_path


def registration(**over: Any) -> dict[str, Any]:
    row = {
        "track": TRACK,
        "hypothesis": "h",
        "change": "c",
        "quick_test": "q",
        "worth_building_if": "w",
        "validation_test": "v",
        "pass_rule": "p",
    }
    return {**row, **over}


def outcome(attempt: str, **over: Any) -> dict[str, Any]:
    row = {
        "id": attempt,
        "quick_result": {"x": 1},
        "built": True,
        "validation": {"report": "r"},
        "guardrails": {"ok": True},
        "decision": "pass",
        "reason": "because",
    }
    return {**row, **over}


def by_hand(attempt: str, **over: Any) -> dict[str, Any]:
    row = {
        "id": attempt,
        "quick_result": {"x": 1},
        "built": True,
        "decision": "fail",
        "reason": "the quick test missed its line",
    }
    return {**row, **over}


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../outside.py",
        "claude/../CLAUDE.md",
        "CLAUDE.md",
        "uv.lock",
        "curation_data/feedback.csv",
        ".claude/settings.json",
        ".loop/home/catalog.sqlite",
        "out/report.json",
        "frontend/src/api/schema.ts",
        "README.md",
        "frontend/src/pages/Home.tsx",
    ],
)
def test_commit_refuses_paths_the_loop_must_not_commit(path: str) -> None:
    with pytest.raises(guard.Refused):
        guard.check_commit_paths([path], set())


def test_commit_refuses_the_owners_entries() -> None:
    with pytest.raises(guard.Refused, match="owner"):
        guard.check_commit_paths(["docs/roadmap.md"], {"docs/roadmap.md"})


PRIVATE_DATA = (
    "claude/improvement-loop/reports/strongest-yours/dev/r.json",
    "claude/improvement-loop/ledger/ledger.jsonl",
    "claude/improvement-loop/ledger/looks.jsonl",
    "docs/decisions/0021-x.md",
)
PUBLIC_CODE = (
    "claude/improvement-loop/ledger/bars.jsonl",
    "claude/improvement-loop/ledger/registrations.jsonl",
    "claude/improvement-loop/tools/loop.py",
    "claude/improvement-loop/tracks/story-yours/champion.toml",
    "docs/architecture.md",
    "src/calico/engine.py",
)


def test_the_loops_data_is_private_and_its_code_public() -> None:
    assert guard.ignored([*PRIVATE_DATA, *PUBLIC_CODE]) == set(PRIVATE_DATA)


@pytest.fixture
def overlay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    private = tmp_path / ".private.git"

    def run(*args: str) -> None:
        subprocess.run(  # noqa: S603 - git, fixed arguments
            ["git", *args],  # noqa: S607
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    run("init", "-q", "-b", "main")
    run("init", "-q", "--bare", "-b", "main", str(private))
    run(f"--git-dir={private}", "config", "core.bare", "false")
    for prefix in ((), (f"--git-dir={private}", f"--work-tree={tmp_path}")):
        run(*prefix, "config", "user.name", "loop")
        run(*prefix, "config", "user.email", "loop@users.noreply.github.com")
    (tmp_path / ".gitignore").write_text(".private.git/\nreports/\n__pycache__/\n")
    monkeypatch.setattr(guard, "REPO", tmp_path)
    monkeypatch.setattr(guard, "PRIVATE_GIT", private)
    monkeypatch.setattr(guard, "PRIVATE_SCOPE", (".",))
    monkeypatch.setattr(loop, "REPO", tmp_path)
    monkeypatch.setenv(guard.COAUTHOR, "Claude Test <noreply@anthropic.com>")
    return tmp_path


def test_private_data_commits_to_the_overlay_and_code_to_the_public_repo(overlay: Path) -> None:
    (overlay / "reports").mkdir()
    (overlay / "reports" / "r.json").write_text("{}\n")
    (overlay / "code.py").write_text("x = 1\n")
    public, private = loop.split_private(["code.py", "reports/r.json"])
    assert (public, private) == (["code.py"], ["reports/r.json"])
    loop.commit_in(private, "Loop: data", private=True)
    loop.commit_in(public, "Loop: code", private=False)
    assert guard.private_git("ls-files").split() == ["reports/r.json"]
    assert guard.git("ls-files").split() == ["code.py"]


def test_new_private_data_counts_as_changed_and_caches_do_not(overlay: Path) -> None:
    (overlay / "reports").mkdir()
    (overlay / "reports" / "r.json").write_text("{}\n")
    (overlay / "tools" / "__pycache__").mkdir(parents=True)
    (overlay / "tools" / "__pycache__" / "t.pyc").write_bytes(b"\0")
    changed = loop.changed_files()
    assert "reports/r.json" in changed
    assert not any("__pycache__" in f for f in changed)


def test_private_data_without_the_overlay_is_refused(
    overlay: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard, "PRIVATE_GIT", overlay / "missing.git")
    (overlay / "reports").mkdir()
    (overlay / "reports" / "r.json").write_text("{}\n")
    with pytest.raises(guard.Refused, match="private"):
        loop.split_private(["reports/r.json"])


@pytest.mark.parametrize(
    ("paths", "scope"),
    [
        (["claude/improvement-loop/ledger/ledger.jsonl"], "data"),
        (["claude/improvement-loop/02-verification.md"], "data"),
        (["claude/improvement-loop/tools/guard.py"], "loop"),
        (["claude/improvement-loop/tools/guard.py", "tests/test_improvement_loop.py"], "loop"),
        (["tests/test_improvement_loop_harness.py"], "loop"),
        (["claude/improvement-loop/tracks/story-yours/x.toml"], "loop"),
        (["claude/improvement-loop/ledger/ledger.jsonl", "src/calico/tasks/best.py"], "full"),
        (["docs/roadmap.md"], "full"),
    ],
)
def test_checks_scale_with_what_changed(paths: list[str], scope: str) -> None:
    assert guard.check_commit_paths(paths, set()).scope == scope


def test_commit_message_carries_the_sessions_attribution_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The line names the model that did the work, so it is given by the session, not fixed."""
    line = "Co-Authored-By: Some Model <noreply@example.com>"
    monkeypatch.setenv(guard.COAUTHOR, "Some Model <noreply@example.com>")
    once = guard.commit_message("Loop: x")
    assert once.count(line) == 1
    assert guard.commit_message(once).count(line) == 1
    with pytest.raises(guard.Refused):
        guard.commit_message("  ")


def test_a_commit_without_a_named_author_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(guard.COAUTHOR, raising=False)
    with pytest.raises(guard.Refused, match=guard.COAUTHOR):
        guard.commit_message("Loop: x")


def test_sandbox_profile_denies_every_path_and_the_network() -> None:
    rules = guard.Rules(deny=(Path("/Volumes/A B/X"), Path("/Users/me/home")))
    text = guard.sandbox_profile(rules)
    assert '(deny file-read* file-write* (subpath "/Volumes/A B/X"))' in text
    assert '(deny file-read* file-write* (subpath "/Users/me/home"))' in text
    assert "network-outbound" in text
    with pytest.raises(guard.Refused):
        guard.sandbox_profile(guard.Rules(deny=(Path('/a"b'),)))


def test_sandbox_profile_orders_an_exception_after_its_denial() -> None:
    """macOS takes the last rule that matches, so the one readable file comes last."""
    rules = guard.Rules(deny=(Path("/s/validation"),), readable=(Path("/s/validation/a.json"),))
    text = guard.sandbox_profile(rules)
    assert text.index('(subpath "/s/validation")') < text.index('(literal "/s/validation/a.json")')
    assert '(deny file-write* (subpath "/w"))' in guard.sandbox_profile(
        guard.Rules(read_only=(Path("/w"),))
    )


def test_sandbox_profile_hides_the_files_of_a_folder_the_engine_must_still_open() -> None:
    """Calico looks for a marker in the ratings folder when it opens, so the folder stays
    and the journals in it are what a script cannot read."""
    text = guard.sandbox_profile(guard.Rules(deny_files=((Path("/h/ratings"), ".jsonl"),)))
    assert '(deny file-read* file-write* (regex #"^/h/ratings/.*\\.jsonl$"))' in text


HOME = Path("/copy/home")
SPLITS = guard.LOOP / "splits"


def denied(mode: str) -> set[Path]:
    return set(guard.tier_rules(mode, HOME, Path("/cache")).deny)


def test_an_ordinary_script_reads_no_rating_and_no_tier_but_dev() -> None:
    hidden = denied("dev")
    files = set(guard.tier_rules("dev", HOME, Path("/cache")).deny_files)
    assert files == {(HOME / "ratings", ".jsonl"), (HOME / "labels", ".json")}
    assert Path("/cache/labels.json") in hidden
    assert {SPLITS / "validation", SPLITS / "locked", guard.REPO / "curation_data"} <= hidden
    assert guard.REPO / ".git" in hidden
    assert guard.REPO / ".private.git" in hidden
    assert guard.tier_rules("dev", HOME, Path("/cache")).readable == ()


def test_the_harness_on_dev_reads_validations_photo_ratings_to_fit_yours_and_nothing_else() -> None:
    rules = guard.tier_rules("train", HOME, Path("/cache"))
    assert SPLITS / "validation" in rules.deny
    assert rules.readable == (SPLITS / "validation" / "strongest-yours.json",)
    assert SPLITS / "locked" in rules.deny


def test_a_look_opens_its_own_tier_only() -> None:
    assert SPLITS / "validation" not in denied("validation")
    assert SPLITS / "locked" in denied("validation")
    assert not {SPLITS / "validation", SPLITS / "locked"} & denied("locked")
    assert guard.tier_rules("locked", HOME, Path("/cache")).deny_files


def test_only_the_freeze_reads_raw_ratings_and_writes_the_splits() -> None:
    freeze = guard.tier_rules("freeze", HOME, Path("/cache"))
    assert freeze.deny_files == ()
    assert SPLITS not in freeze.read_only
    for mode in ("dev", "train", "validation", "locked"):
        rules = guard.tier_rules(mode, HOME, Path("/cache"))
        assert {SPLITS, guard.LOOP / "ledger", guard.LOOP / "tools"} <= set(rules.read_only)
    with pytest.raises(guard.Refused):
        guard.tier_rules("anything", HOME, Path("/cache"))


@pytest.mark.parametrize(
    ("script", "args", "mode"),
    [
        ("tools/harness.py", ["strongest-yours"], "train"),
        ("tools/harness.py", ["strongest-yours", "--tier", "dev"], "train"),
        ("tools/splits.py", ["--dry"], "freeze"),
        ("tools/list_answers.py", ["arms.json"], "freeze"),
        ("tools/summarize.py", ["r.json"], "dev"),
        ("tools/calibrate_repeats.py", [], "dev"),
    ],
)
def test_the_script_decides_what_the_sandbox_opens(script: str, args: list[str], mode: str) -> None:
    assert guard.script_mode(guard.LOOP / script, args) == mode


def test_the_repetition_line_is_calibrated_on_dev_and_validation_never_locked() -> None:
    """`loop calibrate` is the one script, other than a look, that opens validation."""
    script, mode = guard.FIXED["calibrate"]
    assert (guard.LOOP / script).exists()
    assert mode == "validation"
    assert SPLITS / "locked" in denied(mode)
    assert guard.script_mode(guard.LOOP / script, []) == "dev"
    assert "locked" not in (guard.LOOP / script).read_text()


@pytest.mark.parametrize("tier", ["validation", "locked"])
def test_no_script_names_its_way_into_a_tier(tier: str) -> None:
    with pytest.raises(guard.Refused, match="loop look"):
        guard.script_mode(guard.LOOP / "tools" / "harness.py", ["x", "--tier", tier])
    with pytest.raises(guard.Refused, match="loop look"):
        guard.script_mode(guard.LOOP / "tools" / "harness.py", ["x", f"--tier={tier}"])


@pytest.mark.parametrize(
    "verb", ["doctor", "index", "serve", "backup", "sandbox", "holdout", "eval", "train"]
)
def test_only_reading_and_measuring_calico_verbs_run(verb: str) -> None:
    with pytest.raises(guard.Refused):
        guard.calico_args([verb])


def test_verbs_that_read_no_ratings_run() -> None:
    assert guard.calico_args(["select", "best"]) == ["select", "best"]


def test_scripts_run_only_from_the_loop_folder() -> None:
    with pytest.raises(guard.Refused):
        guard.loop_script("src/calico/cli.py")
    with pytest.raises(guard.Refused):
        guard.loop_script("claude/improvement-loop/missing.py")
    assert guard.loop_script("claude/improvement-loop/tools/guard.py").name == "guard.py"


@pytest.mark.parametrize(
    "path",
    [
        "claude/improvement-loop/tracks/strongest-yours/champion.toml",
        "claude/improvement-loop/tracks/story-yours/champion.toml",
        "claude/improvement-loop/tracks/strongest-yours/missing.toml",
        "pyproject.toml",
    ],
)
def test_a_look_takes_a_candidate_of_its_own_track(path: str) -> None:
    """The champion is always in a look, so it is never named; nor is another track's."""
    with pytest.raises(guard.Refused):
        guard.track_config(TRACK, path)


def test_drift_names_the_options_a_champion_and_its_shipped_track_differ_on() -> None:
    assert ledger.drift({"spend_alike": 0.871}, {"spend_alike": 0.871}) == []
    assert ledger.drift({"spend_alike": 0.871}, {}) == ["spend_alike"]
    assert ledger.drift({}, {"diversity": 0.2, "spend_alike": 0.871}) == [
        "diversity",
        "spend_alike",
    ]


@pytest.mark.parametrize("track", guard.TRACKS)
def test_shipped_drift_reads_both_files_of_every_track(track: str) -> None:
    assert isinstance(ledger.shipped_drift(track), list)


def test_registration_numbers_attempts_per_track(books: Path) -> None:
    first = ledger.register(registration(), head="abc")
    second = ledger.register(registration(), head="abc")
    other = ledger.register(registration(track="story-yours"), head="abc")
    assert (first["id"], second["id"], other["id"]) == (
        "strongest-yours-001",
        "strongest-yours-002",
        "story-yours-001",
    )


def test_registration_needs_every_field_and_nothing_else(books: Path) -> None:
    with pytest.raises(guard.Refused, match="missing"):
        ledger.register(registration(pass_rule=""), head="abc")
    with pytest.raises(guard.Refused, match="unknown"):
        ledger.register({**registration(), "extra": 1}, head="abc")
    with pytest.raises(guard.Refused, match="track"):
        ledger.register(registration(track="nope"), head="abc")


def test_a_record_needs_a_registration_and_keeps_its_rule(books: Path) -> None:
    with pytest.raises(guard.Refused, match="never registered"):
        ledger.record(outcome("strongest-yours-001"), head="h", registered_in="r", decided=True)
    rule = "beat by 0.02"
    reg = ledger.register(registration(pass_rule=rule), head="abc")
    row = ledger.record(outcome(reg["id"]), head="h", registered_in="r", decided=True)
    assert row["pass_rule"] == rule
    assert row["registered_in"] == "r"
    with pytest.raises(guard.Refused, match="unknown"):
        ledger.record({**outcome(reg["id"]), "pass_rule": "anything"}, head="h", registered_in="r")


def test_an_attempt_is_recorded_once(books: Path) -> None:
    reg = ledger.register(registration(), head="abc")
    ledger.record(by_hand(reg["id"]), head="h", registered_in="r")
    with pytest.raises(guard.Refused, match="already"):
        ledger.record(by_hand(reg["id"]), head="h", registered_in="r")


@pytest.mark.parametrize("decision", ["pass", "partial"])
def test_a_session_cannot_write_a_pass_by_hand(books: Path, decision: str) -> None:
    """Whether a change passed is computed from its look (`loop decide`), never typed."""
    reg = ledger.register(registration(), head="abc")
    with pytest.raises(guard.Refused, match="loop decide"):
        ledger.record(outcome(reg["id"], decision=decision), head="h", registered_in="r")


def test_a_looked_at_attempt_is_not_closed_by_hand(books: Path) -> None:
    reg = ledger.register(registration(), head="abc")
    with pytest.raises(guard.Refused, match="loop decide"):
        ledger.record(outcome(reg["id"], decision="fail"), head="h", registered_in="r")


def test_a_quick_test_that_fails_is_recorded_without_a_look(books: Path) -> None:
    reg = ledger.register(registration(), head="abc")
    row = ledger.record(
        by_hand(reg["id"], decision="not-built", built=False), head="h", registered_in="r"
    )
    assert row["validation"] is None
    other = ledger.register(registration(), head="abc")
    with pytest.raises(guard.Refused, match="not-built"):
        ledger.record(by_hand(other["id"], decision="not-built"), head="h", registered_in="r")


def with_ladder() -> None:
    ledger.bar({"track": TRACK, "name": "ladder", "value": [{"sigma": 0.1}], "reason": "r"})


def test_validation_looks_stop_at_the_budget(books: Path) -> None:
    ledger.bar({"track": TRACK, "name": "validation_looks", "value": 2, "reason": "r"})
    with_ladder()
    with pytest.raises(guard.Refused, match="registered"):
        ledger.may_look(TRACK, "validation", None)
    for _ in range(2):
        reg = ledger.register(registration(), head="abc")
        n = ledger.may_look(TRACK, "validation", reg["id"])
        ledger.log_look(
            {"track": TRACK, "tier": "validation", "epoch": 1, "n": n, "attempt": reg["id"]}
        )
    third = ledger.register(registration(), head="abc")
    with pytest.raises(guard.Refused, match="budget"):
        ledger.may_look(TRACK, "validation", third["id"])
    ledger.bar({"track": TRACK, "name": "validation_epoch", "value": 2, "reason": "refresh"})
    assert ledger.may_look(TRACK, "validation", third["id"]) == 1


def test_an_attempt_looks_once_and_only_while_open(books: Path) -> None:
    ledger.bar({"track": TRACK, "name": "validation_looks", "value": 9, "reason": "r"})
    with_ladder()
    reg = ledger.register(registration(), head="abc")
    ledger.log_look({"track": TRACK, "tier": "validation", "attempt": reg["id"]})
    with pytest.raises(guard.Refused, match="new registration"):
        ledger.may_look(TRACK, "validation", reg["id"])
    closed = ledger.register(registration(), head="abc")
    ledger.record(by_hand(closed["id"]), head="h", registered_in="r")
    with pytest.raises(guard.Refused, match="no ledger row"):
        ledger.may_look(TRACK, "validation", closed["id"])
    other = ledger.register(registration(track="story-yours"), head="abc")
    with pytest.raises(guard.Refused, match="this track"):
        ledger.may_look(TRACK, "validation", other["id"])


def test_no_look_before_the_track_has_its_nulls(books: Path) -> None:
    ledger.bar({"track": TRACK, "name": "validation_looks", "value": 9, "reason": "r"})
    reg = ledger.register(registration(), head="abc")
    with pytest.raises(guard.Refused, match="loop null"):
        ledger.may_look(TRACK, "validation", reg["id"])


def test_the_champions_null_runs_spend_no_budget(books: Path) -> None:
    ledger.bar({"track": TRACK, "name": "validation_looks", "value": 1, "reason": "r"})
    ledger.log_look({"track": TRACK, "tier": "validation", "kind": "null", "epoch": 1})
    assert ledger.budget(TRACK) == (0, 1)


def test_dev_is_never_counted(books: Path) -> None:
    with pytest.raises(guard.Refused, match="dev"):
        ledger.may_look(TRACK, "dev", None)


def test_the_locked_test_opens_after_enough_accepted_changes(books: Path) -> None:
    ledger.bar({"track": TRACK, "name": "locked_every", "value": 2, "reason": "r"})
    with pytest.raises(guard.Refused, match="locked"):
        ledger.may_look(TRACK, "locked", None)
    for _ in range(2):
        reg = ledger.register(registration(), head="abc")
        ledger.record(outcome(reg["id"]), head="h", registered_in="r", decided=True)
    assert ledger.may_look(TRACK, "locked", None) == 1
    ledger.log_look({"track": TRACK, "tier": "locked"})
    with pytest.raises(guard.Refused, match="locked"):
        ledger.may_look(TRACK, "locked", None)


def test_the_final_locked_look_is_taken_once(books: Path) -> None:
    assert ledger.may_look(TRACK, "locked", None, final=True) == 1
    ledger.log_look({"track": TRACK, "tier": "locked", "final": True})
    with pytest.raises(guard.Refused, match="final"):
        ledger.may_look(TRACK, "locked", None, final=True)


def test_changed_splits_are_found(books: Path) -> None:
    splits = books / "splits"
    splits.mkdir()
    (splits / "a.json").write_text(json.dumps({"x": 1}))
    digest = hashlib.sha256((splits / "a.json").read_bytes()).hexdigest()
    (splits / "SHA256SUMS").write_text(f"{digest}  a.json\n")
    assert ledger.verify_splits() == []
    (splits / "a.json").write_text(json.dumps({"x": 2}))
    assert ledger.verify_splits() == ["a.json"]


def test_the_answers_to_a_list_are_read_for_each_arm_with_an_interval_over_places() -> None:
    """Rate this list: the photos only one of two lists showed, rated blind. The reader
    joins the answers to the arms the loop kept and compares liked ÷ rated."""
    arms, answers, place_of = {}, {}, {}
    for n in range(6):
        for arm, liked in (("champion", [0, 0, 2, 0]), ("candidate", [2, 2, 2, 0])):
            for m, label in enumerate(liked):
                key = f"{arm}-{n}-{m}"
                arms[key], answers[key], place_of[key] = arm, label, f"place {n}"
    arms["unsure"], answers["unsure"], place_of["unsure"] = "candidate", 1, "place 0"
    arms["unrated"], place_of["unrated"] = "champion", "place 1"
    got = list_answers.read_answers(arms, answers, place_of)
    assert got["champion"] == {"liked": 6, "rated": 24, "share": 0.25}
    assert got["candidate"] == {"liked": 18, "rated": 24, "share": 0.75}
    assert got["difference"] == pytest.approx(0.5)
    assert got["interval_90"] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert (got["listed"], got["rated"], got["not_sure"], got["places"]) == (50, 48, 1, 6)


def test_a_list_written_with_config_names_is_read_by_naming_the_champion() -> None:
    arms = {"a": "y0", "b": "y-q0", "c": "y0"}
    assert list_answers.as_roles(arms, "y0") == {
        "a": "champion",
        "b": "candidate",
        "c": "champion",
    }
    assert list_answers.as_roles({"a": "champion"}, None) == {"a": "champion"}


def test_a_list_nobody_rated_yet_reports_nothing_to_compare() -> None:
    got = list_answers.read_answers({"a": "champion", "b": "candidate"}, {}, {})
    assert got["difference"] is None
    assert "interval_90" not in got
    assert got["rated"] == 0


def frozen_and_drawn(root: Path) -> tuple[Path, Path]:
    frozen, drawn = root / "frozen", root / "drawn"
    for folder in (frozen, drawn):
        (folder / "dev").mkdir(parents=True)
        (folder / "places.json").write_text('{"places":{}}\n')
        (folder / "dev" / "story-yours.json").write_text('{"stories":[]}\n')
    sums = [
        f"{hashlib.sha256((frozen / n).read_bytes()).hexdigest()}  {n}"
        for n in ("dev/story-yours.json", "places.json")
    ]
    (frozen / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (drawn / "SHA256SUMS").write_text("ignored\n")
    (drawn / "dev" / "same-view.json").write_text('{"same_view":[1]}\n')
    return frozen, drawn


def test_a_new_label_file_joins_the_frozen_splits_without_touching_them(
    books: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same-view labels were counted in the draw but never written. A second draw may add
    their files only if it reproduces every frozen file byte for byte."""
    frozen, drawn = frozen_and_drawn(books)
    before = (frozen / "SHA256SUMS").read_text()
    assert splits.add_new(drawn, frozen) == ["dev/same-view.json"]
    assert (frozen / "dev" / "same-view.json").read_text() == '{"same_view":[1]}\n'
    after = (frozen / "SHA256SUMS").read_text()
    assert set(before.splitlines()) < set(after.splitlines())
    assert len(after.splitlines()) == 3
    monkeypatch.setattr(ledger, "SPLITS", frozen)
    assert ledger.verify_splits() == []
    assert splits.add_new(drawn, frozen) == []


def test_a_draw_that_differs_from_the_frozen_splits_adds_nothing(books: Path) -> None:
    frozen, drawn = frozen_and_drawn(books)
    (drawn / "places.json").write_text('{"places":{"x":"dev"}}\n')
    before = (frozen / "SHA256SUMS").read_text()
    with pytest.raises(SystemExit, match=r"places\.json"):
        splits.add_new(drawn, frozen)
    assert not (frozen / "dev" / "same-view.json").exists()
    assert (frozen / "SHA256SUMS").read_text() == before


def test_a_draw_that_lacks_a_frozen_file_adds_nothing(books: Path) -> None:
    frozen, drawn = frozen_and_drawn(books)
    (drawn / "dev" / "story-yours.json").unlink()
    with pytest.raises(SystemExit, match="story-yours"):
        splits.add_new(drawn, frozen)
    assert not (frozen / "dev" / "same-view.json").exists()


def test_frozen_splits_are_checked_file_by_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    splits_dir = f.loop_splits(tmp_path / "splits")
    monkeypatch.setattr(ledger, "SPLITS", splits_dir)
    assert ledger.verify_splits() == []
    changed = splits_dir / "validation" / "strongest-yours.json"
    changed.write_text(changed.read_text().replace("true", "false", 1))
    assert ledger.verify_splits() == ["validation/strongest-yours.json"]
    (splits_dir / "SHA256SUMS").unlink()
    assert ledger.verify_splits() == ["no frozen splits"]


def test_made_up_splits_have_every_tier_and_track_file_of_the_real_ones(tmp_path: Path) -> None:
    splits_dir = f.loop_splits(tmp_path / "splits")
    places = json.loads((splits_dir / "places.json").read_text())["places"]
    assert set(places.values()) == set(guard.TIERS)
    for tier in guard.TIERS:
        names = {p.stem for p in (splits_dir / tier).glob("*.json")}
        assert names == {"legacy", "same-view", "story-general", "story-yours", *guard.TRACKS}
        ratings = json.loads((splits_dir / tier / "strongest-yours.json").read_text())["ratings"]
        assert {r["place"] for r in ratings} <= {p for p, t in places.items() if t == tier}


LADDER = [
    {"sigma": 0.1, "overlap": 0.9, "sd": {"precision": 0.005, "repeat_excess": 0.01}},
    {"sigma": 0.5, "overlap": 0.6, "sd": {"precision": 0.02, "repeat_excess": 0.02}},
]


def measured(difference: float, low: float, high: float) -> dict[str, Any]:
    return {"difference": difference, "interval_90": [low, high], "interval_99": [low, high]}


def strongest(precision: dict[str, Any], **over: Any) -> dict[str, Any]:
    paired = {
        "precision": precision,
        "repeat_excess": measured(0.0, -0.01, 0.01),
        "seconds": {"champion": 10.0, "candidate": 10.0},
        "overlap": 0.62,
    }
    return {**paired, **over}


def test_a_gain_passes_when_it_clears_the_null_of_its_own_size_and_zero() -> None:
    """The bar is twice what a meaningless change moving as many picks does: 2 x 0.02."""
    got = verdict.decide(TRACK, strongest(measured(0.05, 0.01, 0.09)), LADDER)
    assert got["decision"] == "pass"
    assert got["rung"]["sigma"] == 0.5
    assert got["checks"][0]["bar"] == pytest.approx(0.04)


def test_the_same_gain_fails_the_bar_of_a_change_that_moves_more_picks() -> None:
    small = strongest(measured(0.015, 0.005, 0.03), overlap=0.92)
    assert verdict.decide(TRACK, small, LADDER)["decision"] == "pass"
    assert verdict.decide(TRACK, {**small, "overlap": 0.6}, LADDER)["decision"] == "partial"


def test_a_gain_whose_interval_reaches_zero_is_partial() -> None:
    got = verdict.decide(TRACK, strongest(measured(0.05, -0.01, 0.11)), LADDER)
    assert got["decision"] == "partial"


@pytest.mark.parametrize(
    "over",
    [
        {"repeat_excess": measured(0.05, 0.03, 0.07)},
        {"seconds": {"champion": 10.0, "candidate": 16.0}},
        {"precision:place@12": measured(-0.2, -0.3, -0.1)},
    ],
)
def test_a_gain_with_a_guardrail_broken_fails(over: dict[str, Any]) -> None:
    got = verdict.decide(TRACK, strongest(measured(0.05, 0.01, 0.09), **over), LADDER)
    assert got["decision"] == "fail"


def test_no_change_is_a_fail() -> None:
    got = verdict.decide(TRACK, strongest(measured(0.0, 0.0, 0.0), overlap=1.0), LADDER)
    assert got["decision"] == "fail"


GENERAL_LADDER = [
    {
        "sigma": 0.3,
        "overlap": 0.6,
        "sd": {
            "kept_auc": 0.01,
            "repeat_excess": 0.01,
            "precision": 0.01,
            "kept_lift": 0.1,
            "laion_pct": 0.01,
        },
    }
]


def general(laion: dict[str, Any] | None) -> dict[str, Any]:
    still = measured(0.0, -0.01, 0.01)
    paired = {
        "kept_auc": measured(0.05, 0.02, 0.08),
        "repeat_excess": still,
        "precision": still,
        "kept_lift": still,
        "seconds": {"champion": 10.0, "candidate": 10.0},
        "overlap": 0.6,
    }
    return paired if laion is None else {**paired, "laion_pct": laion}


def test_general_may_not_pick_what_laion_ranks_lower() -> None:
    """LAION is General's second judge: it can stop a change, never pass one."""
    held = verdict.decide("strongest-general", general(measured(0.0, -0.01, 0.01)), GENERAL_LADDER)
    assert held["decision"] == "pass"
    fell = verdict.decide(
        "strongest-general", general(measured(-0.05, -0.07, -0.03)), GENERAL_LADDER
    )
    assert fell["decision"] == "fail"
    assert "laion_pct" not in [c["measure"] for c in fell["checks"] if c["check"] == "gain"]


def test_general_is_not_decided_without_laion() -> None:
    with pytest.raises(verdict.NoBarError, match="laion_pct"):
        verdict.decide("strongest-general", general(None), GENERAL_LADDER)


STORY_LADDER = [
    {
        "sigma": 0.3,
        "overlap": 0.5,
        "sd": {"rated_auc": 0.01, "set_repetition": 0.01, "within": 0.01, "filled": 0.0},
    }
]


def story(auc: dict[str, Any], repetition: dict[str, Any], filled: float = 0.0) -> dict[str, Any]:
    return {
        "filled": measured(filled, filled, filled),
        "rated_auc": auc,
        "set_repetition": repetition,
        "within": measured(0.0, -0.01, 0.01),
        "seconds": {"champion": 2.0, "candidate": 2.0},
        "overlap": 0.5,
    }


def test_a_story_change_passes_by_repeating_less_when_the_verdicts_hold() -> None:
    less = story(measured(0.0, -0.015, 0.015), measured(-0.06, -0.09, -0.03))
    assert verdict.decide("story-yours", less, STORY_LADDER)["decision"] == "pass"


def test_a_story_change_that_repeats_less_but_reads_verdicts_worse_fails() -> None:
    worse = story(measured(-0.05, -0.08, -0.02), measured(-0.06, -0.09, -0.03))
    assert verdict.decide("story-yours", worse, STORY_LADDER)["decision"] == "fail"


def test_a_story_change_that_reads_verdicts_better_may_not_repeat_more() -> None:
    better = story(measured(0.05, 0.02, 0.08), measured(0.0, -0.01, 0.01))
    assert verdict.decide("story-yours", better, STORY_LADDER)["decision"] == "pass"
    more = story(measured(0.05, 0.02, 0.08), measured(0.05, 0.03, 0.07))
    assert verdict.decide("story-yours", more, STORY_LADDER)["decision"] == "fail"


def test_a_story_change_may_not_make_fewer_stories_than_were_asked_for() -> None:
    """Attempt 001 (14 November 2026) removed repetition by making 4 stories of 12. The
    owner made ``filled`` a guard the same day; nulls always fill, so any shortfall fails."""
    less = measured(-0.06, -0.09, -0.03)
    still = measured(0.0, -0.015, 0.015)
    assert verdict.decide("story-yours", story(still, less), STORY_LADDER)["decision"] == "pass"
    spread = {"rated_auc": 0.01, "set_repetition": 0.01, "within": 0.01, "filled": 0.0}
    ladder = [
        {"sigma": 0.3, "overlap": 0.5, "sd": {**spread, "unfit_located": 0.01, "siglip": 0.01}}
    ]
    for track in ("story-yours", "story-general"):
        short = {**story(still, less, filled=-0.02), "unfit_located": still, "siglip": still}
        assert verdict.decide(track, short, ladder)["decision"] == "fail"


def test_a_story_ladder_made_before_filled_was_a_guard_gives_no_verdict() -> None:
    spread = {"rated_auc": 0.01, "set_repetition": 0.01, "within": 0.01}
    old = [{"sigma": 0.3, "overlap": 0.5, "sd": spread}]
    less = story(measured(0.0, -0.015, 0.015), measured(-0.06, -0.09, -0.03))
    with pytest.raises(verdict.NoBarError, match="filled"):
        verdict.decide("story-yours", less, old)


def test_no_verdict_without_a_null_ladder() -> None:
    with pytest.raises(verdict.NoBarError, match="loop null"):
        verdict.decide(TRACK, strongest(measured(0.05, 0.01, 0.09)), [])


def test_a_record_names_the_dev_reports_its_quick_result_rests_on() -> None:
    """`loop record` commits them with the ledger row, as `loop look` does validation's."""
    text = (
        "dev, reports/strongest-yours/dev/2026-11-15-2123-y0-y-q10.json.gz: +0.008; "
        "also reports/strongest-yours/dev/2026-11-15-2127-y0-y-scale.json.gz, "
        "and reports/strongest-yours/dev/2026-11-15-2123-y0-y-q10.json.gz again"
    )
    assert ledger.dev_reports({"quick_result": text}) == [
        "claude/improvement-loop/reports/strongest-yours/dev/2026-11-15-2123-y0-y-q10.json.gz",
        "claude/improvement-loop/reports/strongest-yours/dev/2026-11-15-2127-y0-y-scale.json.gz",
    ]
    assert ledger.dev_reports({"quick_result": {"x": 1}}) == []
    assert ledger.dev_reports({"quick_result": "closed by hand, no report"}) == []


def test_each_tier_has_its_own_ladder(books: Path) -> None:
    """Dev's nulls give a quick test a measured rung; validation's decide a look."""
    ledger.bar({"track": TRACK, "name": "ladder", "value": [{"sigma": 0.1}], "reason": "r"})
    ledger.bar({"track": TRACK, "name": "dev_ladder", "value": [{"sigma": 0.3}], "reason": "r"})
    assert ledger.ladder(TRACK) == [{"sigma": 0.1}]
    assert ledger.ladder(TRACK, "validation") == [{"sigma": 0.1}]
    assert ledger.ladder(TRACK, "dev") == [{"sigma": 0.3}]
    assert ledger.ladder_name("dev") == "dev_ladder"
    with pytest.raises(guard.Refused, match="locked"):
        ledger.ladder(TRACK, "locked")


def test_a_dev_look_opens_what_the_harness_opens_on_dev() -> None:
    """Dev nulls fit Yours from validation's photo ratings, as `loop py harness.py` does."""
    assert guard.look_mode("dev") == guard.script_mode(
        guard.LOOP / guard.HARNESS, ["--tier", "dev"]
    )
    assert guard.look_mode("validation") == "validation"
    assert guard.look_mode("locked") == "locked"


def test_a_closing_row_commits_the_dev_reports_it_names_that_are_not_committed(
    tmp_path: Path,
) -> None:
    """`loop record` and `loop decide` both carry them, so a closed attempt leaves none behind."""
    dev = "claude/improvement-loop/reports/story-yours/dev"
    for name in ("new.json.gz", "old.json.gz"):
        (tmp_path / dev).mkdir(parents=True, exist_ok=True)
        (tmp_path / dev / name).write_bytes(b"x")
    names = "reports/story-yours/dev/new.json.gz and reports/story-yours/dev/old.json.gz"
    row = {"quick_result": names}
    changed = {f"{dev}/new.json.gz", "somewhere/else.txt"}
    assert ledger.reports_to_commit(row, changed=changed, root=tmp_path) == [f"{dev}/new.json.gz"]
    gone = {"quick_result": "reports/story-yours/dev/gone.json.gz"}
    with pytest.raises(guard.Refused, match="do not exist"):
        ledger.reports_to_commit(gone, changed=changed, root=tmp_path)
