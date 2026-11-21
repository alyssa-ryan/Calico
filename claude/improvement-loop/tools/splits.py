from __future__ import annotations

import collections
import hashlib
import json
import random
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import catalogue
import labels
import numpy as np

SEED = 20261113
DRAW = {"large": 0.6, "small": 0.2, "locked": 0.2}
ROLE = {"large": "validation", "small": "dev", "locked": "locked"}
TARGET = {ROLE[d]: share for d, share in DRAW.items()}
NEAR_DUP = 0.95
SAME_IMAGE = 0.99
SAME_VISIT_SECONDS = 3600.0
ORIGINAL_SECONDS = 2.0
ORIGINAL_CLIP = 0.80
FROZEN = catalogue.LOOP / "splits"
WEIGHTS = ("taste", "story", "unfit", "kept", "anchor", "same_view", "finished", "legacy")
RULES = {
    "unit": "a place (its visits stay together)",
    "look_alikes": (
        "when a labelled photo and a photo at a place in another tier are the same image "
        "(CLIP >= 0.99) or near-duplicates (CLIP >= 0.95) taken within an hour, both are "
        "excluded from every tier and pool"
    ),
    "reserved": (
        "decision 0016's eight places hold its 454 photos and are in no tier and no pool; a photo "
        "elsewhere that is the same image as, or a near-duplicate within an hour of, a photo there "
        "is excluded by itself"
    ),
    "locked": "every unit holding a hold-out place (generation 2)",
    "rest": "seeded shuffle, then each unit, largest first, to the tier furthest below its share",
    "roles": (
        "the large draw (60%) is validation, judged out of fold; the small draw (20%) is dev, "
        "for quick tests; validation's photo ratings also fit the Yours that dev is judged by"
    ),
    "stories": (
        "a story is in the tier of its three photos' places; one that straddles tiers is in none"
    ),
    "holdout_story_sessions": (
        "their stories are locked when all three photos are, otherwise in none"
    ),
}


def originals(cat: catalogue.Catalogue, home: Path) -> dict[str, dict[str, Any]]:
    con = sqlite3.connect(f"file:{home}/catalog.sqlite?mode=ro", uri=True)
    plain: set[str] = set()
    for path, key in con.execute("select path, content_key from photos"):
        if not any(catalogue.CURATION.match(f) for f in Path(path).parts[:-1]):
            plain.add(key)
    by_place = {
        p: np.flatnonzero((cat.location == p) & ~cat.curated)
        for p in sorted(set(cat.location.tolist()))
    }
    out: dict[str, dict[str, Any]] = {}
    for i in np.flatnonzero(cat.curated):
        key = str(cat.keys[i])
        if key in plain:
            out[key] = {"original": key, "how": "same file"}
            continue
        cands = by_place[str(cat.location[i])]
        if len(cands) == 0:
            out[key] = {"original": None, "how": "no plain photo at the place"}
            continue
        sims = cat.clip[cands] @ cat.clip[i]
        dt = np.abs(cat.captured[cands] - cat.captured[i])
        timed = np.isfinite(dt) & (dt <= ORIGINAL_SECONDS)
        if timed.any():
            j = cands[timed][int(np.argmax(sims[timed]))]
            out[key] = {"original": str(cat.keys[j]), "how": "same time"}
            continue
        best = int(np.argmax(sims))
        if sims[best] >= ORIGINAL_CLIP and not np.isfinite(cat.captured[i]):
            out[key] = {"original": str(cat.keys[cands[best]]), "how": "look, no time"}
        else:
            out[key] = {"original": None, "how": "no match"}
    return out


def near_links(cat: catalogue.Catalogue, keys: set[str]) -> list[tuple[int, int]]:
    rows = np.array(sorted(cat.row[k] for k in keys if k in cat.row))
    links: list[tuple[int, int]] = []
    for start in range(0, len(rows), 1024):
        block = rows[start : start + 1024]
        sims = cat.clip[block] @ cat.clip.T
        for bi, j in zip(*np.nonzero(sims >= NEAR_DUP), strict=True):
            i = int(block[bi])
            if cat.location[i] == cat.location[j]:
                continue
            dt = abs(cat.captured[i] - cat.captured[j])
            if sims[bi, j] >= SAME_IMAGE or (np.isfinite(dt) and dt <= SAME_VISIT_SECONDS):
                links.append((i, int(j)))
    return links


def main() -> None:
    dry = "--dry" in sys.argv
    add = "--add" in sys.argv
    out_dir = FROZEN
    if dry or add:
        out_dir = catalogue.cache_dir() / ("splits-dry" if dry else "splits-add")
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "places.json").exists() and not (dry or add):
        raise SystemExit(
            "the splits are frozen and never re-drawn (02-verification.md); "
            "`--add` writes a label file the freeze left out"
        )
    eng = catalogue.engine()
    cat = catalogue.load(eng)
    lab = labels.load(eng)
    home = eng.settings.home
    hold = catalogue.holdout(home)
    reserved_keys = catalogue.reserved_keys()

    def place(key: str) -> str | None:
        i = cat.row.get(key)
        return None if i is None else str(cat.location[i])

    reserved_places = {p for k in reserved_keys if (p := place(k))}
    kept = originals(cat, home)
    labelled = {r.key for r in lab.taste} | {k for s in lab.stories for k in s.keys}
    labelled |= {str(v["original"]) for v in kept.values() if v["original"]}
    labelled |= {k for f in lab.finished for k in f if k}
    labelled |= {a.key for a in lab.anchors} | {g.key for g in lab.legacy}
    reserved_rows = {i for i, p in enumerate(cat.location) if p in reserved_places}
    places = sorted(set(cat.location.tolist()) - reserved_places)
    excluded: set[str] = set()
    cross: list[tuple[int, int]] = []
    for i, j in near_links(cat, labelled | {str(cat.keys[r]) for r in reserved_rows}):
        a, b = str(cat.location[i]), str(cat.location[j])
        if a in reserved_places and b not in reserved_places:
            excluded.add(str(cat.keys[j]))
        elif b in reserved_places and a not in reserved_places:
            excluded.add(str(cat.keys[i]))
        elif a not in reserved_places:
            cross.append((i, j))
    units: dict[str, list[str]] = {p: [p] for p in places}

    weight: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)

    def count(key: str, what: str, n: int = 1) -> None:
        if key not in excluded and (p := place(key)) and p not in reserved_places:
            weight[p][what] += n

    for rated in lab.taste:
        count(rated.key, "taste")
    for s in lab.stories:
        count(s.keys[1], "story")
        if s.unfit:
            count(s.keys[1], "unfit")
    for v in kept.values():
        if v["original"]:
            count(str(v["original"]), "kept")
    for anchor in lab.anchors:
        count(anchor.key, "anchor")
    for sv in lab.same_view:
        count(sv.key, "same_view")
    for f in lab.finished:
        for k in f:
            if k:
                count(k, "finished")
    for g in lab.legacy:
        if g.stars is not None:
            count(g.key, "legacy")

    hold_places = set(hold.get("places", []))
    tier: dict[str, str] = {r: "locked" for r, m in units.items() if hold_places & set(m)}
    total = {w: sum(weight[p][w] for p in places) for w in WEIGHTS}
    have: dict[str, collections.Counter[str]] = {t: collections.Counter() for t in DRAW}
    for r, t in tier.items():
        for p in units[r]:
            have[t].update(weight[p])
    rest = sorted(r for r in units if r not in tier)
    random.Random(SEED).shuffle(rest)  # noqa: S311
    mass = {r: sum((weight[p] for p in units[r]), collections.Counter()) for r in rest}
    rest.sort(key=lambda r: -sum(mass[r][w] / max(total[w], 1) for w in WEIGHTS))
    places_in = collections.Counter(tier.values())
    for r in rest:
        m = mass[r]
        if any(m[w] for w in WEIGHTS):
            gaps = {
                t: sum(
                    (DRAW[t] * total[w] - have[t][w]) / max(total[w], 1) for w in WEIGHTS if m[w]
                )
                for t in DRAW
            }
        else:
            gaps = {t: DRAW[t] * len(units) - places_in[t] for t in DRAW}
        chosen = max(DRAW, key=lambda t: gaps[t])
        tier[r] = chosen
        have[chosen].update(m)
        places_in[chosen] += 1

    tier = {r: ROLE[t] for r, t in tier.items()}
    have = {ROLE[t]: c for t, c in have.items()}
    by_place = {p: tier[p] for p in places}
    straddling = 0
    for i, j in cross:
        if by_place[str(cat.location[i])] != by_place[str(cat.location[j])]:
            excluded.update((str(cat.keys[i]), str(cat.keys[j])))
            straddling += 1
    by_place.update(dict.fromkeys(sorted(reserved_places), "reserved"))
    share = {t: {w: round(have[t][w] / max(total[w], 1), 3) for w in WEIGHTS} for t in TARGET}
    tracks = track_manifests(
        cat, lab, kept, by_place, excluded, set(hold.get("story_sessions", []))
    )
    derived = sorted(
        {
            r["edit"]
            for t in TARGET
            for r in tracks[t]["strongest-general"]["kept"]
            if r["edit"] != r["original"]
        }
    )
    manifest = {
        "snapshot": cat.snapshot,
        "seed": SEED,
        "target": TARGET,
        "rules": RULES,
        "places": by_place,
        "look_alike_pairs_across_tiers": straddling,
        "excluded_photos": sorted(excluded),
        "derived_photos": derived,
        "share_by_tier": share,
        "counts_by_tier": {t: dict(have[t]) for t in TARGET},
    }
    written = [out_dir / "places.json"]
    write(written[0], manifest)
    for t, files in tracks.items():
        (out_dir / t).mkdir(exist_ok=True)
        for name, data in files.items():
            written.append(out_dir / t / f"{name}.json")
            write(written[-1], {"snapshot": cat.snapshot, "tier": t, **data})
    sums = [
        f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out_dir).as_posix()}"
        for f in sorted(written)
    ]
    (out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    sys.stdout.write(json.dumps({"share_by_tier": share, "units": len(units)}, indent=1) + "\n")
    if add:
        sys.stdout.write(json.dumps({"added": add_new(out_dir, FROZEN)}) + "\n")


def add_new(drawn: Path, frozen: Path) -> list[str]:
    def files(folder: Path) -> dict[str, Path]:
        return {
            f.relative_to(folder).as_posix(): f
            for f in sorted(folder.rglob("*.json"))
            if f.is_file()
        }

    was, now = files(frozen), files(drawn)
    differ = sorted(
        n for n, f in was.items() if n not in now or now[n].read_bytes() != f.read_bytes()
    )
    if differ:
        raise SystemExit(
            f"this draw does not reproduce the frozen splits ({', '.join(differ)}): nothing "
            "added. Draw from the copy the splits were frozen from (places.json, `snapshot`)"
        )
    new = sorted(set(now) - set(was))
    for name in new:
        (frozen / name).parent.mkdir(parents=True, exist_ok=True)
        (frozen / name).write_bytes(now[name].read_bytes())
    if new:
        sums = frozen / "SHA256SUMS"
        lines = sums.read_text().splitlines() + [
            f"{hashlib.sha256(now[n].read_bytes()).hexdigest()}  {n}" for n in new
        ]
        sums.write_text("\n".join(lines) + "\n")
    return new


def write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n")


def track_manifests(
    cat: catalogue.Catalogue,
    lab: labels.Labels,
    kept: dict[str, dict[str, Any]],
    by_place: dict[str, str],
    excluded: set[str],
    hold_sessions: set[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    def place(key: str) -> str | None:
        i = cat.row.get(key)
        return None if i is None else str(cat.location[i])

    def tier_of(*keys: str) -> str | None:
        found = set()
        for k in keys:
            p = place(k)
            if p is None or k in excluded:
                return None
            found.add(by_place.get(p))
        t = found.pop() if len(found) == 1 else None
        return t if t in TARGET else None

    def bucket() -> dict[str, list[Any]]:
        return {t: [] for t in TARGET}

    ratings, legacy, kept_rows, stories, unfit, finished, anchors = (bucket() for _ in range(7))
    views = bucket()
    for sv in lab.same_view:
        if t := tier_of(sv.key):
            views[t].append({**asdict(sv), "place": place(sv.key)})
    for r in lab.taste:
        if t := tier_of(r.key):
            ratings[t].append({**asdict(r), "place": place(r.key)})
    for g in lab.legacy:
        if g.stars is not None and (t := tier_of(g.key)):
            legacy[t].append([g.key, g.stars, g.run, g.position, g.blind, place(g.key)])
    for edit, v in sorted(kept.items()):
        if v["original"] and (t := tier_of(str(v["original"]))):
            kept_rows[t].append(
                {"original": v["original"], "edit": edit, "how": v["how"],
                 "place": place(str(v["original"]))}
            )  # fmt: skip
    for s in lab.stories:
        t = tier_of(*s.keys)
        if s.session in hold_sessions and t != "locked":
            continue
        if t:
            row = {**asdict(s), "place": place(s.keys[1])}
            stories[t].append(row)
            if s.unfit and len(s.unfit) < 3:
                unfit[t].append(row)
    for f in lab.finished:
        if all(f) and (t := tier_of(*f)):
            finished[t].append(list(f))
    for a in lab.anchors:
        if t := tier_of(a.key):
            anchors[t].append(asdict(a))
    return {
        tier: {
            "strongest-yours": {"ratings": ratings[tier]},
            "legacy": {
                "columns": ["key", "stars", "run", "position", "blind", "place"],
                "legacy_stars": legacy[tier],
            },
            "strongest-general": {"kept": kept_rows[tier]},
            "story-yours": {
                "stories": stories[tier],
                "finished": finished[tier],
                "anchors": anchors[tier],
            },
            "story-general": {"unfit_marks": unfit[tier]},
            "same-view": {"same_view": views[tier]},
        }
        for tier in TARGET
    }


if __name__ == "__main__":
    main()
