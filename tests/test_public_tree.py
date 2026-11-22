"""Nothing private reaches the public repository: the rules the git hooks apply
(``scripts/public_check.py``), and the same rules over this tree, so a clone whose hooks
were never switched on is caught by CI."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("public_check", ROOT / "scripts" / "public_check.py")
assert _spec
assert _spec.loader
check = importlib.util.module_from_spec(_spec)
sys.modules["public_check"] = check
_spec.loader.exec_module(check)


@pytest.mark.parametrize(
    "path",
    [
        ".private/denylist.txt",
        ".claude/settings.json",
        "docs/roadmap.md",
        "GLOSSARY.md",
        "curation_data/feedback.csv",
        "deploy/deploy.py",
        "tests/private/test_x.py",
        "claude/improvement-loop/RUNBOOK.md",
        "claude/improvement-loop/splits/dev/legacy.json",
        "claude/improvement-loop/reports/x/dev/r.json",
        "claude/improvement-loop/ledger/ledger.jsonl",
        "claude/improvement-loop/phase1/sameness.py",
        "claude/improvement-loop/reviews/story-notes.txt",
        "out/2026-11-20-report.json",
        "ratings.csv",
        "photos/IMG_0001.JPG",
        "frontend/src/assets/cat.webp",
        "catalog.sqlite",
        "vectors.npy",
        "taste.pkl",
        ".env",
    ],
)
def test_private_paths_and_data_files_are_refused(path: str) -> None:
    assert check.path_problems(path)


@pytest.mark.parametrize(
    "path",
    [
        "docs/architecture.md",
        "frontend/src/assets/header-960.webp",
        "claude/improvement-loop/ledger/bars.jsonl",
        "claude/improvement-loop/ledger/registrations.jsonl",
        "claude/improvement-loop/tools/loop.py",
        "src/calico/engine.py",
    ],
)
def test_public_paths_pass(path: str) -> None:
    assert check.path_problems(path) == []


def test_a_file_over_a_megabyte_is_refused() -> None:
    assert check.content_problems("src/big.py", b"x" * 1_000_001, None)
    assert check.content_problems("src/ok.py", b"x" * 1000, None) == []


@pytest.mark.parametrize(
    "text",
    [
        "root = '/Vol" + "umes/My Photos/2021'",
        "home = '/Us" + "ers/jane/Library'",
        "home = '/ho" + "me/jane/calico'",
        "key = '" + "0123456789abcdef" * 2 + "'",
        "mail me at someone@" + "gmail.com",
    ],
)
def test_real_paths_content_keys_and_personal_emails_are_refused(text: str) -> None:
    assert check.content_problems("src/x.py", text.encode(), None)


@pytest.mark.parametrize(
    "text",
    [
        'calico index "/Volumes/Archive/Photos"',
        "home = '/Users/me/home'",
        "Co-Authored-By: Claude <noreply@anthropic.com>",
        "id+name@users.noreply.github.com",
        "sha256 = '" + "a" * 64 + "'",
        "uses: actions/checkout@" + "b" * 40,
    ],
)
def test_invented_paths_hashes_and_noreply_addresses_pass(text: str) -> None:
    assert check.content_problems("src/x.py", text.encode(), None) == []


def test_a_key_in_a_lockfile_is_not_a_content_key() -> None:
    assert check.content_problems("uv.lock", b"hash = '" + b"c" * 32 + b"'", None) == []


def test_a_deny_list_word_is_refused_as_a_whole_word_in_any_case() -> None:
    deny = check.deny_pattern(["Old Quarry", "Tilden"])
    assert check.content_problems("src/x.py", b"photos at old quarry", deny)
    assert check.content_problems("src/x.py", b"TILDEN, 2019", deny)
    assert check.content_problems("src/x.py", b"Tildenville", deny) == []


def test_the_message_names_the_rule_but_never_echoes_a_deny_list_word() -> None:
    deny = check.deny_pattern(["Tilden"])
    (problem,) = check.content_problems("src/x.py", b"at Tilden\n", deny)
    assert "Tilden" not in problem
    assert "src/x.py:1" in problem


def test_this_tree_holds_nothing_private() -> None:
    files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    problems = [
        p
        for name in files
        if name and (ROOT / name).is_file()
        for p in check.path_problems(name)
        + check.content_problems(name, (ROOT / name).read_bytes(), None)
    ]
    assert problems == []


def test_the_hooks_are_what_make_setup_switches_on() -> None:
    makefile = (ROOT / "Makefile").read_text()
    assert re.search(r"git config core\.hooksPath scripts/git-hooks", makefile)
    for hook in ("pre-commit", "commit-msg", "pre-push"):
        assert (ROOT / "scripts" / "git-hooks" / hook).stat().st_mode & 0o111, hook


def _git(repo: Path, *args: str) -> str:
    who = ["-c", "user.name=t", "-c", "user.email=t@users.noreply.github.com"]
    who += ["-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"]
    done = subprocess.run(  # noqa: S603
        ["git", *who, *args],  # noqa: S607
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def _signed(name: str) -> str:
    return "Signed-off-by: " + name + " <" + name + "@" + "vendor.test>"


def test_a_new_branch_is_checked_for_what_it_adds_not_for_what_is_already_public(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, repo = tmp_path / "remote.git", tmp_path / "repo"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    _git(repo, "commit", "-q", "--allow-empty", "-m", "public already\n\n" + _signed("bot"))
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "switch", "-q", "-c", "work")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "new and clean")
    monkeypatch.setattr(check, "ROOT", repo)
    monkeypatch.setattr(check, "DENYLIST", tmp_path / "none.txt")
    assert check.push(_git(repo, "rev-parse", "HEAD"), check.ZERO, "origin") == []
    _git(repo, "commit", "-q", "--allow-empty", "-m", "new\n\n" + _signed("me"))
    found = check.push(_git(repo, "rev-parse", "HEAD"), check.ZERO, "origin")
    assert len(found) == 1
    assert "no-reply" in found[0]


def test_a_commit_already_on_another_remote_is_still_checked_for_this_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public, fork, repo = tmp_path / "public.git", tmp_path / "fork.git", tmp_path / "repo"
    for bare in (public, fork):
        _git(tmp_path, "init", "-q", "--bare", str(bare))
    _git(tmp_path, "init", "-q", "-b", "main", str(repo))
    _git(repo, "commit", "-q", "--allow-empty", "-m", "public already")
    _git(repo, "remote", "add", "origin", str(public))
    _git(repo, "remote", "add", "fork", str(fork))
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "switch", "-q", "-c", "work")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "only on the fork\n\n" + _signed("me"))
    _git(repo, "push", "-q", "fork", "work")
    monkeypatch.setattr(check, "ROOT", repo)
    monkeypatch.setattr(check, "DENYLIST", tmp_path / "none.txt")
    head = _git(repo, "rev-parse", "HEAD")
    assert check.push(head, check.ZERO, "fork") == []
    found = check.push(head, check.ZERO, "origin")
    assert len(found) == 1
    assert "no-reply" in found[0]
