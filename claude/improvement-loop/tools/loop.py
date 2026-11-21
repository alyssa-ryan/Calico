from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import guard
import ledger
from guard import REAL_HOME, REPO, STATE, Refused, git

HOMES = STATE / "homes"
CURRENT = STATE / "home"
LOGS = STATE / "logs"


def say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def stamp() -> str:
    return now().strftime("%Y%m%d-%H%M%S")


def fingerprint(home: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in ("ratings", "labels"):
        digest = hashlib.sha256()
        count = 0
        for f in sorted((home / name).rglob("*")):
            if f.is_file():
                digest.update(f.relative_to(home).as_posix().encode())
                digest.update(hashlib.sha256(f.read_bytes()).digest())
                count += 1
        out[name] = {"files": count, "sha256": digest.hexdigest()}
    return out


def current_home() -> Path:
    if not CURRENT.exists():
        raise Refused("no copy of the home yet: run `loop home`")
    return CURRENT.resolve()


def snapshot_meta(home: Path) -> dict[str, object]:
    meta = home.parent / f"{home.name}.json"
    return dict(json.loads(meta.read_text())) if meta.exists() else {}


def make_copy() -> Path:
    if not REAL_HOME.is_dir():
        raise Refused(f"no calico home at {REAL_HOME}")
    HOMES.mkdir(parents=True, exist_ok=True)
    (STATE / ".metadata_never_index").touch()
    target = HOMES / stamp()
    say(f"cloning {REAL_HOME} -> {target} (APFS clone, no extra disk)")
    subprocess.run(["cp", "-cR", str(REAL_HOME), str(target)], check=True)
    meta = {
        "taken_at": now().isoformat(timespec="seconds"),
        "source": str(REAL_HOME),
        **fingerprint(target),
    }
    (HOMES / f"{target.name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    if CURRENT.is_symlink():
        CURRENT.unlink()
    CURRENT.symlink_to(target.relative_to(STATE))
    return target


def cache_of(home: Path) -> Path:
    return STATE / "cache" / home.name


def write_profile(home: Path, mode: str = "dev") -> Path:
    tiers = guard.tier_rules(mode, home, cache_of(home))
    rules = guard.Rules(
        deny=(*guard.collection_roots(home), REAL_HOME, *tiers.deny),
        read_only=tiers.read_only,
        readable=tiers.readable,
        deny_files=tiers.deny_files,
    )
    text = guard.sandbox_profile(rules)
    path = STATE / f"sandbox-{mode}.sb"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return path


WATCHED = ("src/", "claude/improvement-loop/tools/", "claude/improvement-loop/tracks/")


def uncommitted() -> list[str]:
    return sorted(f for f in changed_files() if f.startswith(WATCHED))


def tools_sha256() -> str:
    digest = hashlib.sha256()
    tools = guard.LOOP / "tools"
    for f in sorted([*tools.glob("*.py"), *tools.glob("*.toml")]):
        digest.update(f.name.encode())
        digest.update(hashlib.sha256(f.read_bytes()).digest())
    return digest.hexdigest()


def sandbox_env(home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("LOOP_SPLITS", None)
    env.pop("LOOP_CACHE", None)
    env.update(
        LOOP_HEAD=git("rev-parse", "HEAD").strip(),
        LOOP_DIRTY=" ".join(uncommitted()),
        LOOP_TOOLS_SHA256=tools_sha256(),
        CALICO_HOME=str(home),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        LOOP_SNAPSHOT=home.name,
        LOOP_REPO=str(REPO),
        PYTHONPATH=str(guard.LOOP / "tools"),
    )
    return env


def run_sandboxed(
    argv: list[str], extra_env: dict[str, str] | None = None, *, mode: str = "dev"
) -> int:
    home = current_home()
    profile = write_profile(home, mode)
    env = sandbox_env(home)
    if extra_env:
        env.update(extra_env)
    command = ["sandbox-exec", "-f", str(profile), "uv", "run", "--no-sync", *argv]
    return subprocess.run(command, cwd=REPO, env=env, check=False).returncode


def look(
    track: str,
    tier: str,
    attempt: str | None,
    configs: list[str],
    *,
    final: bool = False,
    nulls: str = "",
    seeds: int = 2,
) -> Path:
    guard.require_branch()
    bad = ledger.verify_splits()
    if bad:
        raise Refused(f"the frozen splits changed ({bad}); no look until they are restored")
    changed = uncommitted()
    if changed:
        raise Refused(
            f"commit first: a look is tied to a commit, and these differ from it: {changed}"
        )
    kind = "null" if nulls else "look"
    n = 0 if nulls else ledger.may_look(track, tier, attempt, final=final)
    epoch = int(ledger.bars()[track].get("validation_epoch", 1))
    name = "nulls" if nulls else f"look{n}"
    out = guard.LOOP / "reports" / track / tier / f"{stamp()}-{name}.json.gz"
    argv = ["python", str(guard.LOOP / "tools" / "harness.py"), track, "--tier", tier]
    for c in configs:
        argv += ["--config", str(guard.track_config(track, c))]
    if nulls:
        argv += ["--nulls", nulls, "--null-seeds", str(seeds)]
    argv += ["--out", str(out)]
    code = run_sandboxed(argv, mode=guard.look_mode(tier))
    if code != 0 or not out.exists():
        raise Refused(f"the {tier} evaluation failed (exit {code}); no look was logged")
    ledger.log_look(
        {
            "track": track,
            "tier": tier,
            "kind": kind,
            "epoch": epoch,
            "n": n,
            "attempt": attempt,
            "final": final,
            "configs": configs,
            "head": head(),
            "report": rel(out),
            "report_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        }
    )
    return out


def cmd_look(args: argparse.Namespace) -> int:
    out = look(args.track, "validation", args.attempt, args.config)
    commit([rel(out), rel(ledger.LOOKS)], f"Loop: {args.track} validation look for {args.attempt}")
    say(rel(out))
    return 0


def cmd_locked(args: argparse.Namespace) -> int:
    out = look(args.track, "locked", None, args.config, final=args.final)
    commit([rel(out), rel(ledger.LOOKS)], f"Loop: {args.track} locked look")
    say(rel(out))
    return 0


def cmd_null(args: argparse.Namespace) -> int:
    import summarize

    name = ledger.ladder_name(args.tier)
    out = look(args.track, args.tier, None, [], nulls=args.sigmas, seeds=args.seeds)
    rungs = summarize.ladder(summarize.read_report(out))
    ledger.bar(
        {
            "track": args.track,
            "name": name,
            "value": rungs,
            "reason": f"nulls of the champion on {args.tier}: {rel(out)}",
        }
    )
    commit(
        [rel(out), rel(ledger.LOOKS), rel(ledger.BARS)],
        f"Loop: {args.track} {args.tier} null ladder ({args.sigmas})",
    )
    say(json.dumps(rungs, indent=1))
    return 0


def cmd_quick(args: argparse.Namespace) -> int:
    """A dev report read as a look would be, against dev's own ladder. Logs nothing."""
    import summarize
    import verdict

    report = summarize.read_report(REPO / args.report)
    if report["tier"] != "dev":
        raise Refused(f"{args.report} is a {report['tier']} report; `loop quick` reads dev's")
    if args.candidate not in report["configs"] or args.candidate == report["champion"]:
        raise Refused(f"{args.candidate} is not a candidate in {args.report}")
    paired = summarize.paired(report, report["champion"], args.candidate)
    try:
        found = verdict.decide(report["track"], paired, ledger.ladder(report["track"], "dev"))
    except verdict.NoBarError as exc:
        raise Refused(f"{exc} (for dev: `loop null {report['track']} --tier dev`)") from None
    say(json.dumps({"candidate": args.candidate, "tier": "dev", **found}, indent=1))
    say("read on dev: nothing logged; a verdict comes from `loop look` and `loop decide`")
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    import summarize
    import verdict

    seen = ledger.look_of(args.attempt)
    path = REPO / seen["report"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != seen["report_sha256"]:
        raise Refused(f"{seen['report']} is not the report the look wrote")
    report = summarize.read_report(path)
    candidates = [n for n, c in report["configs"].items() if n != report["champion"]]
    if len(candidates) != 1:
        raise Refused(f"the look holds {candidates}; a verdict is on one candidate")
    paired = summarize.paired(report, report["champion"], candidates[0])
    try:
        found = verdict.decide(report["track"], paired, ledger.ladder(report["track"]))
    except verdict.NoBarError as exc:
        raise Refused(str(exc)) from None
    reports = ledger.reports_to_commit(
        {"quick_result": args.quick_result}, changed=changed_files(), root=REPO
    )
    entry = ledger.record(
        {
            "id": args.attempt,
            "quick_result": args.quick_result,
            "built": True,
            "validation": {
                "report": seen["report"],
                "head": report["head"],
                "tools_sha256": report["tools_sha256"],
                "champion": report["champion"],
                "candidate": candidates[0],
                "overlap": found["overlap"],
                "rung": found["rung"],
            },
            "guardrails": {"checks": found["checks"], "flags": found["flags"]},
            "decision": found["decision"],
            "reason": args.reason,
        },
        head=head(),
        registered_in=registered_in(args.attempt),
        decided=True,
    )
    commit([rel(ledger.LEDGER), *reports], f"Loop: {entry['id']} {entry['decision']}")
    say(json.dumps(found, indent=1, ensure_ascii=False))
    return 0


def cmd_bars(args: argparse.Namespace) -> int:
    rows = json.loads(Path(args.file).read_text())
    if not isinstance(rows, list) or not rows:
        raise Refused(f"{args.file}: expected a list of {{track, name, value}}")
    for row in rows:
        ledger.bar({**row, "reason": args.reason})
    commit([rel(ledger.BARS)], f"Loop: {len(rows)} bars: {args.reason[:60]}")
    return 0


def sandbox_denies(path: Path, mode: str = "dev") -> bool:
    home = current_home()
    profile = write_profile(home, mode)
    reader = "/bin/cat" if path.is_file() else "/bin/ls"
    done = subprocess.run(
        ["sandbox-exec", "-f", str(profile), reader, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode != 0 and "Operation not permitted" in done.stderr


def cmd_home(args: argparse.Namespace) -> int:
    if args.refresh or not CURRENT.exists():
        make_copy()
    home = current_home()
    say(f"copy: {home}")
    say(json.dumps(snapshot_meta(home), indent=1))
    return 0


def cmd_calico(args: argparse.Namespace) -> int:
    return run_sandboxed(["calico", *guard.calico_args(args.rest)])


def cmd_py(args: argparse.Namespace) -> int:
    script = guard.loop_script(args.script)
    mode = guard.script_mode(script, args.rest)
    return run_sandboxed(["python", str(script), *args.rest], mode=mode)


def cmd_calibrate(_: argparse.Namespace) -> int:
    script, mode = guard.FIXED["calibrate"]
    out = guard.LOOP / "phase2" / "reports" / "calibrate_repeats_by_tier.json"
    return run_sandboxed(["python", str(guard.LOOP / script), str(out)], mode=mode)


def cmd_baseline(_: argparse.Namespace) -> int:
    path = STATE / "owner-entries.json"
    if path.exists():
        raise Refused(f"{path} already records the owner's entries; it is never rewritten")
    entries = sorted(line[3:] for line in git("status", "--porcelain=v1").splitlines())
    STATE.mkdir(exist_ok=True)
    path.write_text(json.dumps(entries, indent=1) + "\n")
    say(f"recorded {len(entries)} owner entries")
    return 0


def porcelain(out: str) -> set[str]:
    files: set[str] = set()
    parts = out.split("\0")
    i = 0
    while i < len(parts):
        entry = parts[i]
        if len(entry) > 3:
            files.add(entry[3:])
            if entry[0] in "RC":
                i += 1
        i += 1
    return files


def changed_files() -> set[str]:
    files = porcelain(git("status", "--porcelain=v1", "-uall", "-z"))
    if guard.has_private():
        out = guard.private_git(
            "status", "--porcelain=v1", "-uall", "--ignored=traditional", "-z", "--",
            *guard.PRIVATE_SCOPE,
        )  # fmt: skip
        theirs = sorted(f for f in porcelain(out) if not any(j in f for j in guard.JUNK))
        files |= guard.ignored(theirs)
    return files


def split_private(paths: list[str]) -> tuple[list[str], list[str]]:
    private = guard.ignored(paths)
    if private and not guard.has_private():
        raise Refused(
            f"{sorted(private)} are private material, and this checkout has no "
            f"{guard.PRIVATE_GIT.name} to keep them in"
        )
    return [p for p in paths if p not in private], sorted(private)


def run_checks(scope: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"check-{stamp()}.log"
    say(f"checks for a '{scope}' change (log: {log.relative_to(REPO)})")
    with log.open("w") as handle:
        for command in guard.CHECKS[scope]:
            handle.write(f"$ {' '.join(command)}\n")
            handle.flush()
            done = subprocess.run(
                command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=False
            )
            if done.returncode != 0:
                break
    if done.returncode != 0:
        say("\n".join(log.read_text().splitlines()[-40:]))
        raise Refused(f"checks failed; nothing committed (full log {log.relative_to(REPO)})")


def commit(paths: list[str], message: str, *, push_after: bool = True) -> str:
    guard.require_branch()
    if not (STATE / "owner-entries.json").exists():
        raise Refused("record the owner's entries first: `loop baseline` on a clean start")
    if git("diff", "--cached", "--name-only").strip():
        raise Refused("something is already staged; the loop stages only its own files")
    named = guard.check_commit_paths(paths, guard.owner_entries())
    changed = changed_files()
    chosen = sorted(
        f for f in changed if any(f == p or f.startswith(p.rstrip("/") + "/") for p in named.paths)
    )
    missing = [p for p in named.paths if not any(c == p or c.startswith(p + "/") for c in chosen)]
    if missing:
        raise Refused(f"no change to commit in {missing}")
    plan = guard.check_commit_paths(chosen, guard.owner_entries())
    public, private = split_private(chosen)
    before = digests(chosen)
    run_checks(plan.scope)
    if digests(chosen) != before:
        raise Refused("files changed while the checks ran; nothing committed, run it again")
    heads = [
        commit_in(p, message, private=is_private)
        for p, is_private in ((public, False), (private, True))
        if p
    ]
    if push_after:
        if public:
            push()
        if private:
            push_private()
    return heads[0]


def commit_in(paths: list[str], message: str, *, private: bool) -> str:
    run = guard.private_git if private else git
    prefix = guard.private_args() if private else []
    if run("diff", "--cached", "--name-only").strip():
        raise Refused("something is already staged; the loop stages only its own files")
    run("add", *(["--force"] if private else []), "--", *paths)
    staged = sorted(
        s for s in run("diff", "--cached", "--name-only", "--no-renames").split("\n") if s
    )
    if staged != sorted(paths):
        run("restore", "--staged", "--", *staged)
        raise Refused(f"staged {staged} but meant {paths}; unstaged again")
    done = subprocess.run(
        ["git", *prefix, "commit", "-F", "-"],
        cwd=REPO,
        input=guard.commit_message(message),
        text=True,
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        run("restore", "--staged", "--", *paths)
        raise Refused(
            f"git commit failed, unstaged again: {done.stderr.strip() or done.stdout.strip()}"
        )
    head = run("rev-parse", "--short", "HEAD").strip()
    say(f"committed {head}{' (private)' if private else ''}: {len(paths)} file(s)")
    return head


def digests(paths: list[str]) -> dict[str, str]:
    return {
        p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() if (REPO / p).exists() else ""
        for p in paths
    }


def cmd_commit(args: argparse.Namespace) -> int:
    commit(args.paths, args.message, push_after=not args.no_push)
    return 0


def head() -> str:
    return git("rev-parse", "--short", "HEAD").strip()


def read_json(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise Refused(f"{path}: expected one JSON object")
    return data


def cmd_register(args: argparse.Namespace) -> int:
    entry = ledger.register(read_json(args.file), head=head())
    commit([rel(ledger.REGISTRATIONS)], f"Loop: register {entry['id']}: {entry['hypothesis'][:60]}")
    say(json.dumps(entry, indent=1, ensure_ascii=False))
    return 0


def registered_in(attempt: str) -> str:
    out = git("log", "--format=%h", "-S", f'"id": "{attempt}"', "--", rel(ledger.REGISTRATIONS))
    found = out.split()
    if not found:
        raise Refused(f"{attempt}: its registration was never committed")
    return found[-1]


def cmd_record(args: argparse.Namespace) -> int:
    row = read_json(args.file)
    reports = ledger.reports_to_commit(row, changed=changed_files(), root=REPO)
    entry = ledger.record(row, head=head(), registered_in=registered_in(str(row.get("id"))))
    commit([rel(ledger.LEDGER), *reports], f"Loop: {entry['id']} {entry['decision']}")
    say(json.dumps(entry, indent=1, ensure_ascii=False))
    return 0


def cmd_bar(args: argparse.Namespace) -> int:
    value: Any = json.loads(args.value) if args.value[:1] in "[{0123456789-.tfn" else args.value
    entry = ledger.bar(
        {"track": args.track, "name": args.name, "value": value, "reason": args.reason}
    )
    commit([rel(ledger.BARS)], f"Loop: {args.track} bar {args.name} = {args.value}")
    say(json.dumps(entry, ensure_ascii=False))
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    bad = ledger.verify_splits()
    say(f"splits: {'intact' if not bad else 'CHANGED: ' + ', '.join(bad)}")
    rows = ledger.read(ledger.LEDGER)
    drifted = {t: d for t in guard.TRACKS if (d := ledger.shipped_drift(t))}
    for track, names in drifted.items():
        say(f"{track}: champion.toml and src/calico/tracks/{track}.toml differ on {names}")
    for track in guard.TRACKS:
        used, limit = ledger.budget(track)
        mine = [r for r in rows if r["track"] == track]
        open_ = ledger.open_attempts(track)
        rungs = [(r["sigma"], round(r["overlap"], 2)) for r in ledger.ladder(track)]
        dev = [(r["sigma"], round(r["overlap"], 2)) for r in ledger.ladder(track, "dev")]
        say(
            f"{track}: {len(mine)} attempts ({sum(r['decision'] == 'pass' for r in mine)} passed), "
            f"validation looks {used}/{limit}, accepted since locked "
            f"{ledger.accepted_since_locked(track)}, open {open_}, "
            f"null rungs (sigma, overlap) {rungs or 'none: run loop null'}, "
            f"dev rungs {dev or 'none: run loop null --tier dev'}"
        )
    return 1 if bad or drifted else 0


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def push() -> None:
    guard.require_branch()
    target = f"HEAD:refs/heads/{guard.WORK_BRANCH}"
    done = subprocess.run(
        ["git", "push", "--set-upstream", guard.REMOTE, target],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise Refused(f"push failed (nothing forced): {done.stderr.strip()}")
    say(f"pushed to {guard.REMOTE}/{guard.WORK_BRANCH}")


def push_private() -> None:
    target = f"HEAD:refs/heads/{guard.PRIVATE_BRANCH}"
    done = subprocess.run(
        ["git", *guard.private_args(), "push", guard.PRIVATE_REMOTE, target],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise Refused(f"private push failed (nothing forced): {done.stderr.strip()}")
    say(f"pushed to {guard.PRIVATE_REMOTE}/{guard.PRIVATE_BRANCH}")


def cmd_push(_: argparse.Namespace) -> int:
    push()
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    problems = 0

    def line(ok: bool, text: str) -> None:
        nonlocal problems
        problems += not ok
        say(f"{'ok ' if ok else 'BAD'} {text}")

    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    line(branch == guard.WORK_BRANCH, f"branch {branch}")
    line((STATE / "owner-entries.json").exists(), "owner entries recorded")
    line(not git("diff", "--cached", "--name-only").strip(), "nothing staged")
    if CURRENT.exists():
        home = current_home()
        meta = snapshot_meta(home)
        line(True, f"copy {home.name}, taken {meta.get('taken_at')}")
        for root in guard.collection_roots(home):
            line(sandbox_denies(root), f"sandbox refuses {root}")
        line(sandbox_denies(REAL_HOME), f"sandbox refuses the real home {REAL_HOME}")
        splits = guard.LOOP / "splits"
        journal = sorted((home / "ratings").glob("*.jsonl"))[:1]
        for hidden in (*journal, splits / "validation", splits / "locked"):
            if hidden.exists():
                line(sandbox_denies(hidden), f"a script cannot read {hidden.relative_to(REPO)}")
        ratings = splits / "validation" / guard.TRAINING_FILE
        if ratings.exists():
            line(
                not sandbox_denies(ratings, "train"),
                "the harness on dev reads validation's photo ratings to fit Yours",
            )
            line(
                sandbox_denies(splits / "validation" / "story-yours.json", "train"),
                "and no other validation label",
            )
    else:
        line(False, "no copy of the home: run `loop home`")
    ahead = git("rev-list", "--count", f"{guard.REMOTE}/{guard.WORK_BRANCH}..HEAD", check=False)
    line(True, f"commits not pushed: {ahead.strip() or 'branch not on the remote yet'}")
    return 1 if problems else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loop")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor").set_defaults(func=cmd_doctor)
    h = sub.add_parser("home")
    h.add_argument("--refresh", action="store_true")
    h.set_defaults(func=cmd_home)
    sub.add_parser("baseline").set_defaults(func=cmd_baseline)
    sub.add_parser("calibrate").set_defaults(func=cmd_calibrate)
    c = sub.add_parser("calico")
    c.add_argument("rest", nargs=argparse.REMAINDER)
    c.set_defaults(func=cmd_calico)
    s = sub.add_parser("py")
    s.add_argument("script")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(func=cmd_py)
    m = sub.add_parser("commit")
    m.add_argument("-m", "--message", required=True)
    m.add_argument("--no-push", action="store_true")
    m.add_argument("paths", nargs="+")
    m.set_defaults(func=cmd_commit)
    r = sub.add_parser("register")
    r.add_argument("file")
    r.set_defaults(func=cmd_register)
    d = sub.add_parser("record")
    d.add_argument("file")
    d.set_defaults(func=cmd_record)
    b = sub.add_parser("bar")
    b.add_argument("track", choices=guard.TRACKS)
    b.add_argument("name")
    b.add_argument("value")
    b.add_argument("--reason", required=True)
    b.set_defaults(func=cmd_bar)
    sub.add_parser("status").set_defaults(func=cmd_status)
    lk = sub.add_parser("look")
    lk.add_argument("track", choices=guard.TRACKS)
    lk.add_argument("--attempt", required=True)
    lk.add_argument("--config", action="append", required=True)
    lk.set_defaults(func=cmd_look)
    nl = sub.add_parser("null")
    nl.add_argument("track", choices=guard.TRACKS)
    nl.add_argument("--sigmas", default="0.1,0.3,1.0")
    nl.add_argument("--seeds", type=int, default=2)
    nl.add_argument("--tier", choices=sorted(ledger.LADDERS), default="validation")
    nl.set_defaults(func=cmd_null)
    qk = sub.add_parser("quick")
    qk.add_argument("report")
    qk.add_argument("--candidate", required=True)
    qk.set_defaults(func=cmd_quick)
    de = sub.add_parser("decide")
    de.add_argument("attempt")
    de.add_argument("--quick-result", required=True)
    de.add_argument("--reason", required=True)
    de.set_defaults(func=cmd_decide)
    bs = sub.add_parser("bars")
    bs.add_argument("file")
    bs.add_argument("--reason", required=True)
    bs.set_defaults(func=cmd_bars)
    lo = sub.add_parser("locked")
    lo.add_argument("track", choices=guard.TRACKS)
    lo.add_argument("--config", action="append", default=[])
    lo.add_argument("--final", action="store_true")
    lo.set_defaults(func=cmd_locked)
    sub.add_parser("push").set_defaults(func=cmd_push)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
