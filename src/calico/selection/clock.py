"""Capture times on one clock, so photos from cameras set to different zones compare.

Abroad, a phone moves to local time while a camera keeps the zone it was set to: at Can
Bros Church a Canon (-04:00) and an iPhone (+01:00) five hours apart by their clocks were
together in UTC. Both wrote their zone, so a photo with one is put on UTC. A photo
without one (an export that dropped it, an older camera) borrows the zone its camera used
in the same folder, or else the folder's, if every photo there that has one agrees;
otherwise it keeps its own clock. Pure: photos in, times out.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from calico.domain import Photo


def on_one_clock(photos: Sequence[Photo]) -> list[datetime | None]:
    """Each photo's capture time in UTC where its zone is known or borrowed, else as its
    camera recorded it (naive either way); ``None`` when undated."""
    by_camera: dict[tuple[Path, str | None], Counter[int]] = {}
    by_folder: dict[Path, set[int]] = {}
    for p in photos:
        if p.utc_offset is not None:
            by_camera.setdefault((p.path.parent, p.camera), Counter())[p.utc_offset] += 1
            by_folder.setdefault(p.path.parent, set()).add(p.utc_offset)
    times: list[datetime | None] = []
    for p in photos:
        if p.captured_at is None:
            times.append(None)
            continue
        offset = p.utc_offset
        if offset is None and (p.path.parent, p.camera) in by_camera:
            ((offset, _),) = by_camera[p.path.parent, p.camera].most_common(1)
        if offset is None and len(by_folder.get(p.path.parent, ())) == 1:
            (offset,) = by_folder[p.path.parent]
        local = p.captured_at.replace(tzinfo=None)
        times.append(local if offset is None else _to_utc(local, offset))
    return times


def _to_utc(local: datetime, offset: int) -> datetime:
    """``local`` moved to UTC, or kept on its own clock when that would leave the calendar
    (a damaged date at year 1 or 9999)."""
    try:
        return local - timedelta(minutes=offset)
    except OverflowError:
        return local
