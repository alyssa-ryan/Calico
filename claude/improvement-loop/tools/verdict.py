from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SPEED_RATIO = 1.5
SPEED_SLACK = 1.0
MARGIN_SDS = 2.0


@dataclass(frozen=True)
class Rule:
    gains: tuple[tuple[str, int], ...]
    guards: tuple[tuple[str, int], ...]
    flags: tuple[tuple[str, int], ...]
    cells: str | None = None


KINDS = (("auc_close_up", 1), ("auc_portrait", 1))
RULES = {
    "strongest-yours": Rule((("precision", 1),), (("repeat_excess", -1),), KINDS, "precision"),
    "strongest-general": Rule(
        (("precision", 1),),
        (("kept_auc", 1), ("repeat_excess", -1), ("kept_lift", 1), ("laion_pct", 1)),
        KINDS,
        "kept_lift",
    ),
    "story-yours": Rule(
        (("rated_auc", 1), ("set_repetition", -1)),
        (("within", -1), ("filled", 1)),
        (("finished_beats_swaps", 1),),
    ),
    "story-general": Rule(
        (("unfit_located", 1), ("set_repetition", -1)),
        (("within", -1), ("siglip", 1), ("filled", 1)),
        (("finished_beats_swaps", 1),),
    ),
}


class NoBarError(ValueError):
    pass


def rung_for(ladder: list[dict[str, Any]], overlap: float | None) -> dict[str, Any]:
    if not ladder:
        raise NoBarError("no null ladder for this track: run `loop null TRACK` first")
    if overlap is None:
        return min(ladder, key=lambda r: float(r["overlap"]))
    return min(ladder, key=lambda r: abs(float(r["overlap"]) - overlap))


def margin(rung: dict[str, Any], name: str) -> float:
    sd = rung["sd"].get(name)
    if sd is None:
        raise NoBarError(f"the null ladder has no spread for {name!r}")
    return MARGIN_SDS * float(sd)


def moved(paired: dict[str, Any], name: str, sign: int) -> tuple[float, float] | None:
    got = paired.get(name)
    if not got or got.get("difference") is None or "interval_90" not in got:
        return None
    low, high = got["interval_90"]
    return sign * float(got["difference"]), float(low if sign > 0 else -high)


def decide(track: str, paired: dict[str, Any], ladder: list[dict[str, Any]]) -> dict[str, Any]:
    rule = RULES[track]
    rung = rung_for(ladder, paired.get("overlap"))
    checks: list[dict[str, Any]] = []
    gained = False
    promising = False
    sound = True
    for name, sign in rule.gains:
        got = moved(paired, name, sign)
        if got is None:
            raise NoBarError(f"the report does not measure {name!r}")
        gain, low = got
        bar = margin(rung, name)
        ok = gain >= bar and low > 0
        gained |= ok
        promising |= gain > 0
        checks.append(
            {"check": "gain", "measure": name, "gain": gain, "low_90": low, "bar": bar, "ok": ok}
        )
    for name, sign in (*rule.gains, *rule.guards):
        got = moved(paired, name, sign)
        if got is None:
            raise NoBarError(f"the report does not measure {name!r}")
        bar = margin(rung, name)
        ok = got[0] >= -bar
        sound &= ok
        checks.append(
            {"check": "not worse", "measure": name, "gain": got[0], "bar": -bar, "ok": ok}
        )
    before = float(paired["seconds"]["champion"])
    after = float(paired["seconds"]["candidate"])
    quick = after <= max(SPEED_RATIO * before, before + SPEED_SLACK)
    sound &= quick
    checks.append({"check": "speed", "champion": before, "candidate": after, "ok": quick})
    if rule.cells:
        worse = sorted(
            name.split(":", 1)[1]
            for name, got in paired.items()
            if name.startswith(f"{rule.cells}:")
            and "interval_99" in got
            and got["interval_99"][1] < 0
        )
        sound &= not worse
        checks.append({"check": "no cell shown worse", "worse": worse, "ok": not worse})
    flags = []
    for name, sign in rule.flags:
        got = moved(paired, name, sign)
        sd = rung["sd"].get(name)
        if got is not None and sd is not None and got[0] < -MARGIN_SDS * float(sd):
            flags.append({"measure": name, "gain": got[0], "bar": -MARGIN_SDS * float(sd)})
    decision = "fail"
    if sound and gained:
        decision = "pass"
    elif sound and promising:
        decision = "partial"
    return {
        "decision": decision,
        "overlap": paired.get("overlap"),
        "rung": {"sigma": rung["sigma"], "overlap": rung["overlap"]},
        "checks": checks,
        "flags": flags,
    }
