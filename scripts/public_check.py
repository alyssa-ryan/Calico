"""What may enter the public repository, checked by its git hooks and by CI.

    public_check.py staged             the files a commit is about to record
    public_check.py message FILE       a commit message
    public_check.py push LOCAL REMOTE NAME
                                       every commit a push to the remote NAME sends: its tree,
                                       message and emails

Refuses private paths, photos and data files, files over 1 MB, real volume and home
paths, content keys, emails that are not no-reply addresses and, when the private material
is present, any word of ``.private/denylist.txt`` (which this file never names).
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DENYLIST = ROOT / ".private" / "denylist.txt"
MAX_BYTES = 1_000_000
ZERO = "0" * 40

PRIVATE = (
    ".private/*",
    ".private.git/*",
    ".claude/*",
    ".vscode/*",
    "docs/*",
    "GLOSSARY.md",
    "curation_data/*",
    "deploy/*",
    "tests/private/*",
    "claude/improvement-loop/*.md",
    "claude/improvement-loop/proposals/*",
    "claude/improvement-loop/splits/*",
    "claude/improvement-loop/reports/*",
    "claude/improvement-loop/ledger/*",
    "claude/improvement-loop/phase*",
    "claude/improvement-loop/reviews/*",
    ".github/workflows/private.yml",
    ".loop/*",
    "out/*",
    ".env",
    ".env.*",
)
PUBLIC = (
    "docs/architecture.md",
    "claude/improvement-loop/ledger/bars.jsonl",
    "claude/improvement-loop/ledger/registrations.jsonl",
)
DATA = re.compile(
    r"\.(jpe?g|png|webp|heic|heif|gif|tiff?|bmp|dng|cr[23]|nef|arw|orf|rw2|raf|mp4|mov|m4v"
    r"|avi|mkv|3gp|sqlite3?|db|npz|npy|pt|pth|pkl|pickle|csv|tsv|jsonl)$",
    re.IGNORECASE,
)
BANNERS = frozenset(
    f"frontend/src/assets/{name}.webp"
    for name in ("header-960", "header-1942", "footer-960", "footer-1944")
)
LOCKFILES = frozenset({"uv.lock", "frontend/package-lock.json"})

INVENTED_VOLUMES = "Archive|Backup|Spare|Photos|Travel|No|A|Drive|Disk|External"
INVENTED_USERS = "me|someone|you|user|name|runner|catalog"
REAL_PATH = re.compile(
    rf"/Volumes/(?!(?:{INVENTED_VOLUMES})\b)[A-Z]"
    rf"|/Users/(?!(?:{INVENTED_USERS})\b)[a-z]"
    rf"|/home/(?!(?:{INVENTED_USERS})\b)[a-z]"
)
CONTENT_KEY = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{32}(?![0-9a-fA-F])")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
NO_REPLY = re.compile(r"noreply|@example\.(?:com|org|net|invalid)$", re.IGNORECASE)


def path_problems(path: str) -> list[str]:
    if path in PUBLIC or path in BANNERS:
        return []
    for pattern in PRIVATE:
        if fnmatch.fnmatchcase(path, pattern) or path == pattern.rstrip("/*"):
            return [f"{path}: private material ({pattern}) never goes in the public repository"]
    if DATA.search(path):
        return [f"{path}: photos and data files never go in the public repository"]
    return []


def deny_pattern(words: list[str]) -> re.Pattern[str] | None:
    words = [w.strip() for w in words if w.strip() and not w.startswith("#")]
    if not words:
        return None
    alternatives = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


def load_deny() -> re.Pattern[str] | None:
    if not DENYLIST.exists():
        return None
    return deny_pattern(DENYLIST.read_text(encoding="utf-8").splitlines())


def text_problems(where: str, text: str, deny: re.Pattern[str] | None, *, keys: bool) -> list[str]:
    found: list[str] = []
    for n, line in enumerate(text.splitlines(), 1):
        at = f"{where}:{n}"
        if REAL_PATH.search(line):
            found.append(f"{at}: a real volume or home path (use <archive volume> or <home>)")
        if keys and CONTENT_KEY.search(line):
            found.append(f"{at}: a 32-hex string, which may be a photo's content key")
        if any(not NO_REPLY.search(m.group()) for m in EMAIL.finditer(line)):
            found.append(f"{at}: an email address that is not a no-reply address")
        if deny is not None and deny.search(line):
            found.append(f"{at}: a word on the private deny list")
    return found


def content_problems(path: str, data: bytes, deny: re.Pattern[str] | None) -> list[str]:
    if len(data) > MAX_BYTES:
        return [f"{path}: {len(data):,} bytes, over the 1 MB limit"]
    if path in BANNERS or b"\0" in data[:8000]:
        return []
    text = data.decode("utf-8", errors="replace")
    return text_problems(path, text, deny, keys=path not in LOCKFILES)


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout  # noqa: S603, S607


def staged() -> list[str]:
    deny = load_deny()
    names = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode()
    problems: list[str] = []
    for name in filter(None, names.split("\0")):
        problems += path_problems(name)
        problems += content_problems(name, git("show", f":{name}"), deny)
    return problems


def message(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return text_problems("commit message", text, load_deny(), keys=True)


def published(remote_name: str) -> list[str]:
    own = f"refs/remotes/{remote_name}/"
    keys = subprocess.run(
        ["git", "config", "--name-only", "--get-regexp", r"^remote\..+\.url$"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        check=False,
    ).stdout.decode()
    names = [key[len("remote.") : -len(".url")] for key in keys.split()]
    longer = [f"refs/remotes/{other}/" for other in names if other.startswith(f"{remote_name}/")]
    refs = git("for-each-ref", "--format=%(refname)", own).decode().split()
    return [ref for ref in refs if not any(ref.startswith(o) for o in longer)]


def files(rev: str, *, changed: bool) -> list[str]:
    listing: tuple[str, ...] = ("ls-tree", "-r")
    if changed:
        listing = ("diff-tree", "-r", "-c", "--root", "--no-commit-id", "--diff-filter=ACMRT")
    return list(filter(None, git(*listing, "-z", "--name-only", rev).decode().split("\0")))


def push(local: str, remote: str, remote_name: str) -> list[str]:
    deny = load_deny()
    span = [local, "--not", *published(remote_name)] if remote == ZERO else [f"{remote}..{local}"]
    outgoing = git("rev-list", *span).decode().split()
    problems: list[str] = []
    for sha in outgoing:
        short = sha[:7]
        info = git("show", "-s", "--format=%ae%n%ce%n%B", sha).decode()
        author, committer, body = info.split("\n", 2)
        problems += [
            f"commit {short}: {email!r} is not a no-reply address"
            for email in sorted({author, committer})
            if not NO_REPLY.search(email)
        ]
        problems += text_problems(f"commit {short} message", body, deny, keys=True)
    sent = [(local, files(local, changed=False))]
    sent += [(sha, files(sha, changed=True)) for sha in outgoing]
    for rev, names in sent:
        for name in names:
            problems += path_problems(name)
            problems += content_problems(name, git("show", f"{rev}:{name}"), deny)
    return list(dict.fromkeys(problems))


def main(argv: list[str]) -> int:
    if argv[:1] == ["staged"]:
        problems = staged()
    elif argv[:1] == ["message"] and len(argv) == 2:
        problems = message(argv[1])
    elif argv[:1] == ["push"] and len(argv) == 4:
        problems = push(argv[1], argv[2], argv[3])
    else:
        sys.stderr.write(__doc__ or "")
        return 2
    for problem in problems:
        sys.stderr.write(f"refused: {problem}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
