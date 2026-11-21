from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Any

import catalogue
import summarize

SEED = 20261113
OUT = catalogue.REPO / "out"
ARMS = catalogue.REPO / ".loop" / "disagreement"


def only_one_list(report: dict[str, Any], candidate: str, cell: str) -> list[dict[str, Any]]:
    champion = report["champion"]
    theirs = {p["unit"]: p for p in report["results"][candidate].get(cell, [])}
    rows = []
    for pool in report["results"][champion].get(cell, []):
        other = theirs.get(pool["unit"])
        if other is None:
            continue
        for arm, mine, yours in (("champion", pool, other), ("candidate", other, pool)):
            seen = set(yours["pick_rows"])
            rows += [
                {"place": pool["unit"], "row": row, "arm": arm}
                for row, rated in zip(mine["pick_rows"], mine["pick_rated"], strict=True)
                if row not in seen and not rated
            ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--cell", default="place@50")
    args = parser.parse_args()
    report = summarize.read_report(Path(args.report))
    rows = only_one_list(report, args.candidate, args.cell)
    cat = catalogue.load()
    paths = {
        key: path for path, key in catalogue.all_paths(catalogue.engine().settings.home).items()
    }
    random.Random(SEED).shuffle(rows)  # noqa: S311
    rows.sort(key=lambda r: r["place"])
    name = f"{dt.datetime.now(dt.UTC):%Y-%m-%d}-disagreement-{report['track']}-{args.candidate}"
    OUT.mkdir(exist_ok=True)
    ARMS.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}.csv").open("w", newline="") as handle:
        sheet = csv.writer(handle)
        sheet.writerow(["place", "photo"])
        for r in rows:
            sheet.writerow([r["place"], paths.get(str(cat.keys[r["row"]]), "")])
    (ARMS / f"{name}.json").write_text(
        json.dumps({str(cat.keys[r["row"]]): r["arm"] for r in rows}, indent=1) + "\n"
    )
    places = len({r["place"] for r in rows})
    sys.stdout.write(f"{OUT / f'{name}.csv'}: {len(rows)} unrated photos at {places} places\n")


if __name__ == "__main__":
    main()
