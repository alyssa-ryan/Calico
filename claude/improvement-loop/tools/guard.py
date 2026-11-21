from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[3]
LOOP = REPO / "claude" / "improvement-loop"
STATE = REPO / ".loop"
WORK_BRANCH = "feat/improvement-loop"
REMOTE = "origin"
REAL_HOME = Path.home() / "Library" / "Application Support" / "calico"
TRACKS = ("strongest-general", "strongest-yours", "story-general", "story-yours")
TIERS = ("dev", "validation", "locked")
COAUTHOR = "LOOP_COAUTHOR"
PRIVATE_GIT = REPO / ".private.git"
PRIVATE_REMOTE = "calico-private"
PRIVATE_BRANCH = "main"
PRIVATE_SCOPE = ("claude/improvement-loop/", "docs/")
JUNK = ("__pycache__/", ".DS_Store", ".pyc")

COMMIT_PREFIXES = (
    "claude/improvement-loop/",
    "src/calico/",
    "tests/",
    "docs/",
    ".gitignore",
    "pyproject.toml",
)
NEVER_COMMIT = (
    "CLAUDE.md",
    "uv.lock",
    "curation_data/",
    ".claude/",
    "frontend/openapi.json",
    "frontend/src/api/",
    ".loop/",
    "out/",
)
DATA_SUFFIXES = (".md", ".json", ".json.gz", ".jsonl", ".csv", ".txt", ".sha256")
CALICO_VERBS = frozenset({"select", "runs", "locations", "tasks", "models"})


class Refused(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(f"loop: refused: {message}")


LOOP_TESTS = ("tests/test_improvement_loop.py", "tests/test_improvement_loop_harness.py")
CHECKS = {
    "data": (("uv", "run", "pytest", "-q", "tests/test_docs.py", *LOOP_TESTS),),
    "loop": (
        ("uv", "run", "ruff", "format", "--check", "."),
        ("uv", "run", "ruff", "check", "."),
        ("uv", "run", "mypy"),
        ("uv", "run", "pytest", "-q", "tests/test_docs.py", *LOOP_TESTS),
    ),
    "full": (("make", "check"),),
}


@dataclass(frozen=True)
class CommitPlan:
    paths: tuple[str, ...]
    scope: str


def check_commit_paths(paths: list[str], owner_entries: set[str]) -> CommitPlan:
    if not paths:
        raise Refused("name the files to commit")
    clean: list[str] = []
    for raw in paths:
        p = PurePosixPath(raw)
        if p.is_absolute() or ".." in p.parts or raw.strip() != raw:
            raise Refused(f"{raw!r}: give a path relative to the repository, without '..'")
        text = p.as_posix()
        if any(text == n.rstrip("/") or text.startswith(n) for n in NEVER_COMMIT):
            raise Refused(f"{text}: never committed by the loop")
        if not any(text == c or text.startswith(c) for c in COMMIT_PREFIXES):
            raise Refused(f"{text}: outside the folders the loop may commit {COMMIT_PREFIXES}")
        if text in owner_entries:
            raise Refused(f"{text}: was changed before the loop started; it is the owner's")
        clean.append(text)
    return CommitPlan(tuple(sorted(set(clean))), check_scope(clean))


def check_scope(paths: list[str]) -> str:
    loop = "claude/improvement-loop/"
    if all(p.startswith(loop) and p.endswith(DATA_SUFFIXES) for p in paths):
        return "data"
    if all(p.startswith(loop) or p in LOOP_TESTS for p in paths):
        return "loop"
    return "full"


def commit_message(message: str) -> str:
    body = message.strip()
    if not body:
        raise Refused("the commit message is empty")
    who = os.environ.get(COAUTHOR, "").strip()
    if not who:
        raise Refused(
            f"set {COAUTHOR} to the model doing the work, "
            f"as in {COAUTHOR}='Claude X <noreply@anthropic.com>'"
        )
    line = f"Co-Authored-By: {who}"
    if line in body:
        return body + "\n"
    return f"{body}\n\n{line}\n"


def collection_roots(home: Path) -> list[Path]:
    settings = home / "calico.toml"
    if not settings.exists():
        raise Refused(f"no settings at {settings}")
    data = tomllib.loads(settings.read_text())
    return sorted({Path(c["root"]) for c in data.get("collections", [])})


MODES = ("dev", "train", "validation", "locked", "freeze")
FREEZE_SCRIPTS = frozenset({"tools/splits.py", "tools/list_answers.py"})
HARNESS = "tools/harness.py"
FIXED = {"calibrate": ("tools/calibrate_repeats.py", "validation")}
TRAINING_FILE = "strongest-yours.json"


@dataclass(frozen=True)
class Rules:
    deny: tuple[Path, ...] = ()
    read_only: tuple[Path, ...] = ()
    readable: tuple[Path, ...] = ()
    deny_files: tuple[tuple[Path, str], ...] = ()


def tier_rules(mode: str, home: Path, cache: Path) -> Rules:
    if mode not in MODES:
        raise Refused(f"sandbox mode {mode!r} is not one of {MODES}")
    splits = LOOP / "splits"
    deny = [REPO / ".git", PRIVATE_GIT]
    readable: list[Path] = []
    files: list[tuple[Path, str]] = []
    if mode != "freeze":
        files += [(home / "ratings", ".jsonl"), (home / "labels", ".json")]
        deny += [cache / "labels.json", REPO / "curation_data"]
        if mode in ("dev", "train"):
            deny.append(splits / "validation")
        if mode != "locked":
            deny.append(splits / "locked")
        if mode == "train":
            readable.append(splits / "validation" / TRAINING_FILE)
    read_only = [
        LOOP / "tools",
        LOOP / "tracks",
        LOOP / "ledger",
        REPO / "src",
        REPO / "tests",
        REPO / "docs",
    ]
    if mode != "freeze":
        read_only.append(splits)
    return Rules(tuple(deny), tuple(read_only), tuple(readable), tuple(files))


def script_mode(script: Path, args: list[str]) -> str:
    name = script.relative_to(LOOP).as_posix()
    if name in FREEZE_SCRIPTS:
        return "freeze"
    if name != HARNESS:
        return "dev"
    tier = "dev"
    for n, arg in enumerate(args):
        if arg == "--tier" and n + 1 < len(args):
            tier = args[n + 1]
        elif arg.startswith("--tier="):
            tier = arg.split("=", 1)[1]
    if tier != "dev":
        raise Refused(f"the {tier} tier opens only through `loop look` / `loop locked`")
    return "train"


def look_mode(tier: str) -> str:
    """The sandbox mode a look or null run on a tier needs: dev's is the harness's own."""
    return "train" if tier == "dev" else tier


def quoted(path: Path) -> str:
    text = str(path)
    if '"' in text or "\\" in text:
        raise Refused(f"cannot quote {text!r} in a sandbox profile")
    return f'"{text}"'


def sandbox_profile(rules: Rules) -> str:
    lines = ["(version 1)", "(allow default)", '(deny network-outbound (remote ip "*:*"))']
    lines += [f"(deny file-read* file-write* (subpath {quoted(p)}))" for p in rules.deny]
    for folder, suffix in rules.deny_files:
        quoted(folder)
        pattern = f"^{re.escape(str(folder))}/.*{re.escape(suffix)}$"
        lines.append(f'(deny file-read* file-write* (regex #"{pattern}"))')
    lines += [f"(deny file-write* (subpath {quoted(p)}))" for p in rules.read_only]
    lines += [f"(allow file-read* (literal {quoted(p)}))" for p in rules.readable]
    return "\n".join(lines) + "\n"


def calico_args(args: list[str]) -> list[str]:
    if not args or args[0] not in CALICO_VERBS:
        raise Refused(f"calico {args[:1]}: allowed verbs are {sorted(CALICO_VERBS)}")
    if args[0] == "train" and "--home" in args:
        raise Refused("train runs on the copy only")
    return args


def loop_script(path: str) -> Path:
    target = (REPO / path).resolve()
    if not target.is_relative_to(LOOP.resolve()) or target.suffix != ".py":
        raise Refused(f"{path}: only Python scripts under {LOOP.relative_to(REPO)} run here")
    if not target.exists():
        raise Refused(f"{path}: no such script")
    return target


def track_config(track: str, path: str) -> Path:
    target = (REPO / path).resolve()
    folder = (LOOP / "tracks" / track).resolve()
    if target.parent != folder or target.suffix != ".toml" or not target.exists():
        raise Refused(f"{path}: a look takes a config in {folder.relative_to(REPO)}")
    if target.name == "champion.toml":
        raise Refused("the champion is always in a look; name the candidate's config")
    return target


def git(*args: str, check: bool = True) -> str:
    done = subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and done.returncode != 0:
        raise Refused(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def private_args() -> list[str]:
    return [f"--git-dir={PRIVATE_GIT}", f"--work-tree={REPO}"]


def private_git(*args: str, check: bool = True) -> str:
    return git(*private_args(), *args, check=check)


def has_private() -> bool:
    return PRIVATE_GIT.is_dir()


def ignored(paths: list[str]) -> set[str]:
    """The paths the public repository ignores: the private material's."""
    if not paths:
        return set()
    done = subprocess.run(
        ["git", "check-ignore", "--no-index", "-z", "--stdin"],
        cwd=REPO,
        input="\0".join(paths) + "\0",
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode not in (0, 1):
        raise Refused(f"git check-ignore failed: {done.stderr.strip()}")
    return {p for p in done.stdout.split("\0") if p}


def owner_entries() -> set[str]:
    path = STATE / "owner-entries.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()))


def require_branch() -> None:
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    if branch != WORK_BRANCH:
        raise Refused(f"on {branch}; the loop works on {WORK_BRANCH} only")
