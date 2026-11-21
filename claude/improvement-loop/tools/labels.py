from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from catalogue import REPO, all_paths, cache_dir, nfc

if TYPE_CHECKING:
    from calico.engine import Engine

FINISHED = REPO / "curation_data" / "finished_stories.json"
LEGACY = REPO / "curation_data" / "v0_2_0" / "feedback.csv"


@dataclass(frozen=True)
class PhotoLabel:
    key: str
    liked: bool
    use: str
    session: str | None


@dataclass(frozen=True)
class StoryLabel:
    session: str
    keys: tuple[str, str, str]
    works: bool
    unfit: tuple[str, ...]
    templates: tuple[str, ...]
    ways: tuple[str, ...]


@dataclass(frozen=True)
class ViewLabel:
    reference: str
    key: str
    value: int


@dataclass(frozen=True)
class AnchorLabel:
    key: str
    could: bool
    session: str


@dataclass(frozen=True)
class LegacyLabel:
    key: str
    run: str
    stars: int | None
    thumb: int | None
    position: int | None
    blind: bool
    mode: str
    repeat: str
    variation: str
    duplicate_of: str | None


@dataclass
class Labels:
    taste: list[PhotoLabel] = field(default_factory=list)
    compare_ranks: dict[str, dict[str, int]] = field(default_factory=dict)
    stories: list[StoryLabel] = field(default_factory=list)
    anchors: list[AnchorLabel] = field(default_factory=list)
    same_view: list[ViewLabel] = field(default_factory=list)
    finished: list[tuple[str, ...]] = field(default_factory=list)
    legacy: list[LegacyLabel] = field(default_factory=list)


def gather(eng: Engine) -> Labels:
    from calico.reviews import labels_of, story_label, taste_labels, unfit

    current = eng.ratings.current()
    sets = eng.labels.all()
    out = Labels()
    tl = taste_labels(current)
    uses: dict[str, str] = {}
    for answer in current.values():
        uses.setdefault(answer.subject[0], answer.use)
    for key in sorted(tl.liked):
        out.taste.append(PhotoLabel(key, tl.liked[key], uses.get(key, ""), tl.session.get(key)))
    for s in sets:
        if s.kind == "compare":
            for it in s.items:
                for sb in it.shown_by:
                    out.compare_ranks.setdefault(it.content_key, {})[sb.template] = sb.rank
        elif s.kind == "stories":
            for st in s.stories:
                lab = story_label(st, current)
                if lab not in (0, 2):
                    continue
                keys = tuple(p.content_key for p in st.photos)
                if len(keys) != 3:
                    continue
                out.stories.append(
                    StoryLabel(
                        session=s.set_id,
                        keys=(keys[0], keys[1], keys[2]),
                        works=lab == 2,
                        unfit=tuple(sorted(unfit(st, current))),
                        templates=tuple(sorted({a.template for a in st.arms})),
                        ways=tuple(sorted({a.way for a in st.arms})),
                    )
                )
        elif s.kind == "anchor":
            for key, lab in labels_of(s, current, controls=False).items():
                if lab in (0, 2):
                    out.anchors.append(AnchorLabel(key, lab == 2, s.set_id))
        elif s.kind == "same_view":
            ref = str(s.reference)
            for key, lab in labels_of(s, current, controls=False).items():
                out.same_view.append(ViewLabel(ref, key, int(lab)))
    out.finished = finished_stories(eng)
    out.legacy = legacy(eng)
    return out


def finished_stories(eng: Engine) -> list[tuple[str, ...]]:
    if not FINISHED.exists():
        return []
    data = json.loads(FINISHED.read_text())
    paths = all_paths(eng.settings.home)
    out = []
    for name in sorted(data["stories"]):
        st = data["stories"][name]
        out.append(tuple(paths.get(nfc(str(f)), "") for f in st["paths"]))
    return out


def legacy(eng: Engine) -> list[LegacyLabel]:
    paths = all_paths(eng.settings.home)
    out = []
    with LEGACY.open() as fh:
        for row in csv.DictReader(fh):
            key = paths.get(nfc(row["image_path"]))
            if not key:
                continue
            dup = paths.get(nfc(row["duplicate_of"])) if row.get("duplicate_of") else None
            out.append(
                LegacyLabel(
                    key=key,
                    run=row["run_id"],
                    stars=int(row["stars"]) if row["stars"] else None,
                    thumb=int(row["vote"]) if row["vote"] in ("1", "-1") else None,
                    position=int(row["position_in_gallery"])
                    if row["position_in_gallery"]
                    else None,
                    blind=row["blind"].lower() in ("true", "1", "yes"),
                    mode=row["match_mode"],
                    repeat=row["is_repeat_shot"],
                    variation=row["repeat_variation"],
                    duplicate_of=dup,
                )
            )
    return out


def load(eng: Engine) -> Labels:
    path = cache_dir() / "labels.json"
    if path.exists():
        raw = json.loads(path.read_text())
        return Labels(
            taste=[PhotoLabel(**r) for r in raw["taste"]],
            compare_ranks=raw["compare_ranks"],
            stories=[
                StoryLabel(
                    **{
                        **r,
                        "keys": tuple(r["keys"]),
                        "unfit": tuple(r["unfit"]),
                        "templates": tuple(r["templates"]),
                        "ways": tuple(r["ways"]),
                    }
                )
                for r in raw["stories"]
            ],
            anchors=[AnchorLabel(**r) for r in raw["anchors"]],
            same_view=[ViewLabel(**r) for r in raw["same_view"]],
            finished=[tuple(f) for f in raw["finished"]],
            legacy=[LegacyLabel(**r) for r in raw["legacy"]],
        )
    labels = gather(eng)
    path.write_text(json.dumps(as_json(labels)))
    return labels


def as_json(labels: Labels) -> dict[str, Any]:
    return {
        "taste": [asdict(r) for r in labels.taste],
        "compare_ranks": labels.compare_ranks,
        "stories": [asdict(r) for r in labels.stories],
        "anchors": [asdict(r) for r in labels.anchors],
        "same_view": [asdict(r) for r in labels.same_view],
        "finished": [list(f) for f in labels.finished],
        "legacy": [asdict(r) for r in labels.legacy],
    }
