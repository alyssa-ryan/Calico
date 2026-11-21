from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import tomllib
from typing import TYPE_CHECKING, Any

from guard import LOOP, REPO, TIERS, TRACKS, Refused

if TYPE_CHECKING:
    from pathlib import Path

BOOKS = LOOP / "ledger"
REGISTRATIONS = BOOKS / "registrations.jsonl"
LEDGER = BOOKS / "ledger.jsonl"
LOOKS = BOOKS / "looks.jsonl"
BARS = BOOKS / "bars.jsonl"
SPLITS = LOOP / "splits"
SHIPPED = REPO / "src" / "calico" / "tracks"
DEV_REPORT = re.compile(rf"reports/(?:{'|'.join(TRACKS)})/dev/[\w.-]+\.json\.gz")

REGISTER_FIELDS = (
    "track",
    "hypothesis",
    "change",
    "quick_test",
    "worth_building_if",
    "validation_test",
    "pass_rule",
)
RECORD_FIELDS = ("id", "quick_result", "built", "validation", "guardrails", "decision", "reason")
OPTIONAL = ("validation", "guardrails")
BY_HAND = ("fail", "not-built")
DECISIONS = ("pass", "partial", "fail", "not-built")
BAR_FIELDS = ("track", "name", "value", "reason")


def read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def stamp() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def need(row: dict[str, Any], fields: tuple[str, ...], optional: tuple[str, ...] = ()) -> None:
    missing = [f for f in fields if f not in optional and (f not in row or row[f] in ("", None))]
    if missing:
        raise Refused(f"missing fields {missing}")
    extra = sorted(set(row) - set(fields))
    if extra:
        raise Refused(f"unknown fields {extra}; allowed {list(fields)}")


def register(row: dict[str, Any], *, head: str) -> dict[str, Any]:
    need(row, REGISTER_FIELDS)
    if row["track"] not in TRACKS:
        raise Refused(f"track {row['track']!r} is not one of {TRACKS}")
    done = read(REGISTRATIONS)
    n = 1 + sum(1 for r in done if r["track"] == row["track"])
    entry = {**row, "id": f"{row['track']}-{n:03d}", "at": stamp(), "head": head}
    append(REGISTRATIONS, entry)
    return entry


def record(
    row: dict[str, Any], *, head: str, registered_in: str, decided: bool = False
) -> dict[str, Any]:
    need(row, RECORD_FIELDS, OPTIONAL)
    row = {**dict.fromkeys(OPTIONAL), **row}
    if row["decision"] not in DECISIONS:
        raise Refused(f"decision {row['decision']!r} is not one of {DECISIONS}")
    if not decided and row["decision"] not in BY_HAND:
        raise Refused(
            f"{row['decision']!r} is computed from the validation look by `loop decide`; "
            f"`loop record` writes only {BY_HAND}"
        )
    if not decided and row["validation"]:
        raise Refused("an attempt that had its validation look is closed by `loop decide`")
    if row["decision"] == "not-built" and row["built"]:
        raise Refused("a change that was built cannot be 'not-built'")
    if row["decision"] in ("pass", "partial") and not row["validation"]:
        raise Refused("pass and partial need the validation look's report")
    reg = {r["id"]: r for r in read(REGISTRATIONS)}.get(row["id"])
    if reg is None:
        raise Refused(f"{row['id']} was never registered: register before testing")
    if any(r["id"] == row["id"] for r in read(LEDGER)):
        raise Refused(f"{row['id']} already has its ledger row; a tweak is a new registration")
    entry = {
        **row,
        "track": reg["track"],
        "hypothesis": reg["hypothesis"],
        "validation_test": reg["validation_test"],
        "pass_rule": reg["pass_rule"],
        "registered_at": reg["at"],
        "registered_in": registered_in,
        "at": stamp(),
        "commit": head,
    }
    append(LEDGER, entry)
    return entry


def bar(row: dict[str, Any]) -> dict[str, Any]:
    need(row, BAR_FIELDS)
    if row["track"] not in TRACKS:
        raise Refused(f"track {row['track']!r} is not one of {TRACKS}")
    entry = {**row, "at": stamp()}
    append(BARS, entry)
    return entry


def bars() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {t: {} for t in TRACKS}
    for r in read(BARS):
        out[r["track"]][r["name"]] = r["value"]
    return out


def budget(track: str) -> tuple[int, int]:
    limit = int(bars()[track].get("validation_looks", 0))
    epoch = int(bars()[track].get("validation_epoch", 1))
    used = sum(
        1
        for r in read(LOOKS)
        if r["track"] == track
        and r["tier"] == "validation"
        and r.get("kind", "look") == "look"
        and r.get("epoch", 1) == epoch
    )
    return used, limit


def accepted_since_locked(track: str) -> int:
    looks = [r for r in read(LOOKS) if r["track"] == track and r["tier"] == "locked"]
    since = looks[-1]["at"] if looks else ""
    return sum(
        1
        for r in read(LEDGER)
        if r["track"] == track and r["decision"] == "pass" and r["at"] > since
    )


def open_attempts(track: str) -> list[str]:
    closed = {r["id"] for r in read(LEDGER)}
    return [r["id"] for r in read(REGISTRATIONS) if r["track"] == track and r["id"] not in closed]


def look_of(attempt: str) -> dict[str, Any]:
    mine = [r for r in read(LOOKS) if r.get("attempt") == attempt and r["tier"] == "validation"]
    if not mine:
        raise Refused(f"{attempt} has had no validation look: run `loop look` first")
    return mine[-1]


LADDERS = {"validation": "ladder", "dev": "dev_ladder"}


def ladder_name(tier: str) -> str:
    """The bar that holds a tier's null ladder: validation's decides looks, dev's reads
    quick tests. The locked tier has none."""
    if tier not in LADDERS:
        raise Refused(f"a null ladder is for {sorted(LADDERS)}, not {tier!r}")
    return LADDERS[tier]


def ladder(track: str, tier: str = "validation") -> list[dict[str, Any]]:
    return list(bars()[track].get(ladder_name(tier), []))


def may_look(track: str, tier: str, attempt: str | None, *, final: bool = False) -> int:
    if track not in TRACKS:
        raise Refused(f"track {track!r} is not one of {TRACKS}")
    if tier not in TIERS or tier == "dev":
        raise Refused("only validation and locked looks are counted; dev is free")
    if tier == "validation":
        if not attempt or attempt not in open_attempts(track):
            raise Refused(
                "a validation look belongs to a registered attempt of this track that has "
                "no ledger row yet (--attempt ID)"
            )
        if any(r.get("attempt") == attempt for r in read(LOOKS)):
            raise Refused(f"{attempt} has had its look; a tweak is a new registration")
        if not ladder(track):
            raise Refused(f"{track} has no null ladder to judge by: run `loop null {track}`")
        used, limit = budget(track)
        if used >= limit:
            raise Refused(
                f"{track}: validation budget spent ({used}/{limit}); refresh it as "
                "02-verification.md says, or stop"
            )
        return used + 1
    every = int(bars()[track].get("locked_every", 5))
    ready = accepted_since_locked(track)
    if ready < every and not final:
        raise Refused(
            f"{track}: the locked test opens after {every} accepted changes ({ready} so far), "
            "or once with --final when the loop stops"
        )
    if final and any(
        r["track"] == track and r["tier"] == "locked" and r.get("final") for r in read(LOOKS)
    ):
        raise Refused(f"{track}: the final locked look was already taken")
    return 1 + sum(1 for r in read(LOOKS) if r["track"] == track and r["tier"] == "locked")


def log_look(row: dict[str, Any]) -> dict[str, Any]:
    entry = {**row, "at": stamp()}
    append(LOOKS, entry)
    return entry


def verify_splits() -> list[str]:
    sums = SPLITS / "SHA256SUMS"
    if not sums.exists():
        return ["no frozen splits"]
    bad = []
    for line in sums.read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = SPLITS / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    return bad


def dev_reports(row: dict[str, Any]) -> list[str]:
    """The dev reports a record's quick result names, as paths from the repo root, in
    order of first mention."""
    text = row.get("quick_result")
    if not isinstance(text, str):
        return []
    return [f"claude/improvement-loop/{m}" for m in dict.fromkeys(DEV_REPORT.findall(text))]


def reports_to_commit(row: dict[str, Any], *, changed: set[str], root: Path) -> list[str]:
    """The dev reports a closing row names that are not committed yet; refuses a named report
    that does not exist."""
    reports = dev_reports(row)
    missing = [r for r in reports if not (root / r).exists()]
    if missing:
        raise Refused(f"the quick result names reports that do not exist: {missing}")
    return [r for r in reports if r in changed]


def drift(champion: dict[str, Any], shipped: dict[str, Any]) -> list[str]:
    """The options on which a champion and the track file Calico ships differ."""
    return sorted(k for k in {*champion, *shipped} if champion.get(k) != shipped.get(k))


def shipped_drift(track: str) -> list[str]:
    champion = tomllib.loads((LOOP / "tracks" / track / "champion.toml").read_text())
    shipped = tomllib.loads((SHIPPED / f"{track}.toml").read_text())
    return drift(champion.get("options", {}), shipped.get("options", {}))
