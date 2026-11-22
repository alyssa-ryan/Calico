"""The command line. Parses arguments, calls the engine, renders the answer.

No selection logic lives here: ``select`` builds a ``SelectionRequest`` and hands
it to ``Engine.select``; everything else reads the catalog, run store or
settings and prints them. Errors a user can fix are ``CalicoError``s, shown as
``error: <message>`` with exit code 1; anything else is a bug and propagates.
"""

from __future__ import annotations

import base64
import csv
import dataclasses
import errno
import html
import importlib
import io
import ipaddress
import itertools
import json
import math
import shlex
import shutil
import socket
import statistics
import sys
import threading
import urllib.request
import webbrowser
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import tomli_w
import typer
from pydantic import JsonValue, ValidationError
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column, Table
from rich.text import Text
from typer.core import TyperGroup

from calico import __version__
from calico.backup import Noted, Part, backup, last_backup, verify
from calico.catalog.indexer import Indexer, IndexReport, thumbnail_path
from calico.catalog.locations import location_for
from calico.catalog.place_names import (
    BRACKETS,
    SamePlace,
    WordSpellings,
    has_stray_spaces,
    similar_places,
    word_spellings,
)
from calico.catalog.scanner import scan
from calico.catalog.store import Catalog
from calico.controls import read_controls
from calico.domain import SelectionRequest, SelectionResult
from calico.engine import Engine
from calico.errors import (
    CalicoError,
    CapabilityError,
    ConfigError,
    FolderNotFoundError,
    InvalidRequestError,
)
from calico.evaluation import (
    BAR_AT,
    COMPARE_PICKS,
    CURVE_STEP,
    DISAGREEMENT_COUNT,
    DISAGREEMENT_GAP,
    EXAMPLE_WEIGHTS,
    LEARNED,
    LEARNED_ALONE,
    LEARNED_SWAPS,
    RUNNING_AFTER,
    RUNNING_LINE,
    STORY_L2,
    STORY_TOO_ALIKE,
    STORY_VARIETY_WEIGHTS,
    TODAY,
    WIN_PLACES,
    Disagreement,
    Disagreements,
    FinishedReport,
    FirstPicks,
    LearnedStories,
    Signals,
    StoryReport,
    TasteReport,
    criteria_intervals,
    example_intervals,
    pair_weighted_auc,
    view_differences,
)
from calico.evaluation.controls import CONTROLS_PER_SESSION, Retest, Share
from calico.experiments import Tried, record
from calico.features import build_extractors, extractor_registry
from calico.features.base import Embedder, feature_key
from calico.holdout import read_holdout
from calico.imaging import RAW_HINT, raw_support
from calico.labels import LabelStore
from calico.logs import configure_logging
from calico.models import calibration as calibration_module
from calico.models.base import LabelModel
from calico.models.taste import TASTE_L2
from calico.patch_store import folder_size, owner, running, scratch_folders, size_text
from calico.paths import missing_folder, normalise
from calico.plugins import TASKS_GROUP, Registry
from calico.ratings import read_legacy
from calico.ratings_migration import MigrationReport, migrate, open_journal
from calico.ratings_store import RatingsJournal
from calico.reviews import answered
from calico.runs import RunStore, left_out_because
from calico.sandbox import build_sandbox, refuse_to_replace
from calico.sessions import Sessions, StoryVariant
from calico.settings import (
    CollectionConfig,
    Settings,
    default_home,
    load_settings,
    save_settings,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from calico.catalog.indexer import ProgressFn, ProgressStep
    from calico.domain import Evidence
    from calico.evaluation import (
        ClearCalls,
        CompareReport,
        ExampleEvaluated,
        Threshold,
        ViewReport,
    )
    from calico.evaluation.holdout import HoldOut
    from calico.evaluation.stats import Compared, Interval
    from calico.features.base import ImageExtractor
    from calico.sessions import PlaceProgressFn, TasteTraining

MIN_PYTHON = (3, 12)
BACKUP_STALE_DAYS = 30
"""Days after which calico doctor says to back up again."""
BACKUP_STALE_ANSWERS = 300
"""Ratings made since the last backup after which calico doctor says to back up again, as
well: about an evening's rating, which cannot be recomputed (10 November 2026 audit, DT-1;
a backup 12 days old had missed 39.5% of the ratings)."""
MAX_EXPLAINED = 10
MAX_FAILURES_SHOWN = 10
COMPARING = "Comparing details"


class _CalicoGroup(TyperGroup):
    """Turns expected failures into ``error: ...`` and exit code 1, without a traceback."""

    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except CalicoError as exc:
            Console(stderr=True).print(Text(f"error: {exc}", style="red"), soft_wrap=True)
            raise typer.Exit(1) from None


app = typer.Typer(
    cls=_CalicoGroup,
    name="calico",
    help="Pick photographs from an archive, and explain every pick.",
    no_args_is_help=True,
    add_completion=False,
)
collections_app = typer.Typer(
    help="Folders of photos Calico may read.", no_args_is_help=True, cls=_CalicoGroup
)
config_app = typer.Typer(help="Inspect and change settings.", no_args_is_help=True)
runs_app = typer.Typer(help="Saved selections.", no_args_is_help=True)
feedback_app = typer.Typer(help="Ratings recorded against runs.", no_args_is_help=True)
ratings_app = typer.Typer(
    help="Your ratings: every answer you gave, kept in the ratings journal.", no_args_is_help=True
)
eval_app = typer.Typer(
    help="Label searches and measure how well 'Describe it' ranks photos.", no_args_is_help=True
)
app.add_typer(collections_app, name="collections")
app.add_typer(config_app, name="config")
app.add_typer(runs_app, name="runs")
app.add_typer(feedback_app, name="feedback")
app.add_typer(ratings_app, name="ratings")
app.add_typer(eval_app, name="eval")
train_app = typer.Typer(help="Learn from your ratings.", no_args_is_help=True)
app.add_typer(train_app, name="train")
holdout_app = typer.Typer(
    help="Ratings no tuning may see: left out of every calico eval, kept for a last check.",
    no_args_is_help=True,
)
app.add_typer(holdout_app, name="holdout")
controls_app = typer.Typer(
    help="Photos you rated before, asked again inside Compare Strongest sessions without saying "
    "so, to measure how consistent you are.",
    no_args_is_help=True,
)
app.add_typer(controls_app, name="controls")


@eval_app.callback()
@train_app.callback()
def _say_which_label_sets_are_left_out() -> None:
    # Every measurement and every training reads the labels: one that cannot be read is
    # left out of all of them, so say so before any of them runs.
    unreadable = LabelStore(load_settings().labels_dir).unreadable()
    if unreadable:
        count = len(unreadable)
        _err().print(
            f"{_plural(count, 'labelling set')} could not be read and "
            f"{'is' if count == 1 else 'are'} left out (calico doctor says why): "
            + ", ".join(path.name for path, _ in unreadable),
            markup=False,
            soft_wrap=True,
        )


def _out() -> Console:
    return Console(highlight=False)


def _err() -> Console:
    return Console(stderr=True, highlight=False)


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            metavar="",
            show_default=False,
            help="Log decisions and timings to stderr (-v info, -vv debug).",
        ),
    ] = 0,
    log_json: Annotated[
        bool, typer.Option("--log-json", help="Log one JSON object per line on stderr.")
    ] = False,
) -> None:
    configure_logging(verbose, log_json)


# -- collections ---------------------------------------------------------------------------


@collections_app.command("add")
def collections_add(
    name: Annotated[str, typer.Argument(help="Short name, e.g. 'explore'.")],
    root: Annotated[Path, typer.Argument(help="Folder holding the photos (must exist).")],
    *,
    location_depth: Annotated[
        int,
        typer.Option(
            min=1, help="Folder level below ROOT whose name is a photo's location (1 = top)."
        ),
    ] = 1,
    location_pattern: Annotated[
        str | None,
        typer.Option(
            help="Regex applied to that folder name; its (?P<location>...) group is the location."
        ),
    ] = None,
    location_ignore: Annotated[
        list[str] | None,
        typer.Option(
            help="Regex removed from the folder name first, e.g. '\\([^)]*\\)' for notes "
            "in brackets; repeatable."
        ),
    ] = None,
    location_deepest: Annotated[
        bool,
        typer.Option(
            help="Name the place by the deepest folder at or below that level that names "
            "one (a city inside a trip folder)."
        ),
    ] = False,
    not_place: Annotated[
        list[str] | None,
        typer.Option(
            help="Regex for a folder name that organises rather than names a place, e.g. "
            "'Originals' or 'Lightroom.*' (matched whole, any case); repeatable."
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(help="Glob of files/folders to skip; repeatable. Replaces the defaults."),
    ] = None,
    exclude_screenshots: Annotated[
        bool,
        typer.Option(
            help="Leave PNG files with no camera recorded (screenshots, saved images) out of "
            "searches. Leave off for scans, which are often camera-less PNGs too."
        ),
    ] = False,
) -> None:
    """Add (or replace) a collection and save it to calico.toml."""
    resolved = normalise(root)
    if not resolved.is_dir():
        raise FolderNotFoundError(missing_folder(resolved))
    fields: dict[str, Any] = {
        "name": name,
        "root": resolved,
        "location_depth": location_depth,
        "location_pattern": location_pattern,
        "location_deepest": location_deepest,
        "exclude_screenshots": exclude_screenshots,
    }
    if not_place:
        fields["location_not_places"] = tuple(not_place)
    if exclude:
        fields["exclude"] = tuple(exclude)
    if location_ignore:
        fields["location_ignore"] = tuple(location_ignore)
    try:
        collection = CollectionConfig(**fields)
    except ValidationError as exc:
        raise ConfigError(f"Invalid collection:\n{exc}") from exc
    settings = load_settings()
    verb = "Updated" if any(c.name == name for c in settings.collections) else "Added"
    saved = settings.with_collection(collection)
    path = save_settings(saved)
    console = _out()
    console.print(f"{verb} collection {name!r} in {path}", markup=False, soft_wrap=True)
    console.print(_collection_summary(collection), markup=False, soft_wrap=True)
    _say_how_it_nests(saved, collection)
    same = [c for c in saved.collections if c.root == collection.root and c.name != name]
    if same:
        used = saved.collection_for(collection.root)
        _err().print(
            f"Warning: {same[0].name!r} already has that root. Only {used.name!r} is used for "
            "its photos, so the other's location rule never applies. Remove one of them "
            f"from {path}.",
            markup=False,
            soft_wrap=True,
        )


def _say_how_it_nests(settings: Settings, added: CollectionConfig) -> None:
    """Collections may nest (a photo belongs to the deepest), so say which one's rules a
    folder now follows."""
    for other in settings.collections:
        if other.name == added.name or other.root == added.root:
            continue
        if added.root.is_relative_to(other.root):
            line = (
                f"Note: {added.root} is inside {other.name!r} ({other.root}); its photos belong "
                f"to {added.name!r} now, with its own location rule and exclusions."
            )
        elif other.root.is_relative_to(added.root):
            line = (
                f"Note: {other.name!r} ({other.root}) is inside it; its photos stay "
                f"{other.name!r}'s."
            )
        else:
            continue
        _err().print(line, markup=False, soft_wrap=True)


@collections_app.command("remove")
def collections_remove(
    name: Annotated[str, typer.Argument(help="The collection to forget.")],
) -> None:
    """Forget a collection. Nothing in its folder is touched; its photos stay in the
    catalog but are no longer searched, until it is added again."""
    settings = load_settings()
    if not any(c.name == name for c in settings.collections):
        raise ConfigError(f"No collection {name!r}. See: calico collections list")
    kept = tuple(c for c in settings.collections if c.name != name)
    path = save_settings(settings.model_copy(update={"collections": kept}))
    _out().print(f"Removed collection {name!r} from {path}", markup=False, soft_wrap=True)


def _location_rule(collection: CollectionConfig) -> str:
    rule = f"depth {collection.location_depth}"
    if collection.location_deepest:
        rule += ", deepest place"
        if collection.location_not_places:
            rule += f", not {', '.join(collection.location_not_places)}"
    if collection.location_ignore:
        rule += f", ignore {' '.join(collection.location_ignore)}"
    return f"{rule}, {collection.location_pattern}" if collection.location_pattern else rule


def _collection_summary(collection: CollectionConfig) -> str:
    return (
        f"  root     {collection.root}\n"
        f"  location {_location_rule(collection)}\n"
        f"  exclude  {', '.join(collection.exclude)}"
        + (", screenshots (PNG with no camera)" if collection.exclude_screenshots else "")
    )


@collections_app.command("list")
def collections_list(
    as_json: Annotated[bool, typer.Option("--json", help="Print them as JSON.")] = False,
) -> None:
    """List configured collections and whether their folders are mounted."""
    settings = load_settings()
    console = _out()
    if as_json:
        found = [
            {
                "name": c.name,
                "root": str(c.root),
                "mounted": c.root.is_dir(),
                "location_rule": _location_rule(c),
            }
            for c in settings.collections
        ]
        print(json.dumps(found, indent=2))
        return
    if not settings.collections:
        console.print("No collections yet. Add one with: calico collections add NAME ROOT")
        return
    table = Table("name", "root", "mounted?", "location rule")
    for c in settings.collections:
        table.add_row(
            Text(c.name),
            Text(str(c.root), overflow="fold"),
            "yes" if c.root.is_dir() else "no",
            Text(_location_rule(c), overflow="fold"),
        )
    console.print(table)


# -- config --------------------------------------------------------------------------------


@config_app.command("show")
def config_show() -> None:
    """Print where calico.toml lives and the effective settings."""
    settings = load_settings()
    note = "" if settings.config_path.exists() else " (not written yet; showing defaults)"
    console = _out()
    console.print(f"{settings.config_path}{note}", markup=False, soft_wrap=True)
    data = settings.model_dump(mode="json", exclude={"home"}, exclude_none=True)
    console.print(tomli_w.dumps(data), markup=False, soft_wrap=True)


@config_app.command("set-embedder")
def config_set_embedder(
    name: Annotated[str, typer.Argument(help="Extractor name, e.g. 'clip', 'siglip' or 'tiny'.")],
    describe: Annotated[
        bool,
        typer.Option("--describe", help="Set the embedder that reads “Describe it” instead."),
    ] = False,
    view: Annotated[
        bool,
        typer.Option("--view", help="Set the image model that matches views for Same view."),
    ] = False,
) -> None:
    """Choose the embedder, checking first that it can be built here."""
    if describe and view:
        raise typer.BadParameter("Choose one: --describe or --view.")
    settings = load_settings()
    extractor_registry().get(name)  # unknown name -> error listing the known ones
    field = "describe_embedder" if describe else "view_embedder" if view else "embedder"
    candidate = settings.model_copy(update={field: name})
    extractors = build_extractors(candidate)  # CapabilityError / ConfigError: nothing is saved
    embedder = extractors[name]
    save_settings(candidate)
    reads_text = isinstance(embedder, Embedder) and embedder.supports_text
    what = "“Describe it” embedder" if describe else "Same view model" if view else "Embedder"
    suffix = (
        "" if reads_text or not describe else " (it cannot read text, so “Describe it” is ignored)"
    )
    _out().print(f"{what} set to {name!r}{suffix}.", markup=False, soft_wrap=True)
    if getattr(settings, field) != name:
        _out().print(
            "Features are cached per embedder; the next request re-indexes.",
            markup=False,
            soft_wrap=True,
        )


# -- index ---------------------------------------------------------------------------------


def _bar() -> Progress:
    """A progress bar on stderr; none off a terminal, where it left a blank line in what
    ``--log-json`` promises is JSON lines."""
    err = _err()
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err,
        transient=True,
        disable=not err.is_terminal,
    )


@contextmanager
def _place_progress() -> Iterator[PlaceProgressFn]:
    """A progress bar on stderr while a measurement compares a place's views part by part,
    driven by ``progress(location, done, total)``: naming the place, and that the wait is
    once per place, since a run that prints nothing for many minutes looks hung."""
    with _bar() as bar:
        task_id = bar.add_task("", total=None, visible=False)
        shown: list[str] = []

        def update(location: str, done: int, total: int) -> None:
            bar.update(
                task_id,
                completed=done,
                total=total,
                visible=True,
                description=f"Comparing views at {location} (once per place)",
            )
            if shown != [location]:  # a new place shows at once, not at the next tick
                shown[:] = [location]
                bar.refresh()

        yield update


@contextmanager
def _progress(label: str) -> Iterator[ProgressFn]:
    """A progress bar on stderr, driven by ``progress(done, total, step)`` (``_bar``)."""
    with _bar() as bar:
        task_id = bar.add_task(label, total=None)

        def update(done: int, total: int, step: ProgressStep) -> None:
            described = label if step == "reading" else COMPARING
            bar.update(task_id, completed=done, total=total, description=described)

        yield update


def _render_index_report(report: IndexReport) -> None:
    console = _out()
    console.print(
        f"Indexed {report.scanned} files in {report.seconds:.1f}s: "
        f"added {report.added}, updated {report.updated}, unchanged {report.unchanged}, "
        f"removed {report.removed}, failed {len(report.failed)}, "
        f"RAW skipped {report.raw_without_rendered}, links skipped "
        f"{report.links_skipped + len(report.linked_dirs)}",
        markup=False,
        soft_wrap=True,
    )
    if report.unsupported:
        kinds = ", ".join(f"{n} {s}" for s, n in sorted(report.unsupported.items()))
        console.print(f"  not read (not photos Calico decodes): {kinds}", markup=False)
    for path, message in report.failed[:MAX_FAILURES_SHOWN]:
        console.print(Text(f"  failed: {path}: {message}", style="yellow"), soft_wrap=True)
    hidden = len(report.failed) - MAX_FAILURES_SHOWN
    if hidden > 0:
        console.print(f"  ... and {hidden} more (see: calico doctor)", markup=False)


@app.command("index")
def index_command(
    folders: Annotated[list[Path], typer.Argument(help="Folders inside a collection.")],
    workers: Annotated[int, typer.Option(min=1, help="Parallel image decoders.")] = 4,
) -> None:
    """Bring the catalog up to date with FOLDER... (requests also do this on demand)."""
    settings = load_settings()
    resolved = [normalise(p) for p in folders]
    for folder in resolved:
        if not folder.is_dir():
            raise FolderNotFoundError(missing_folder(folder))
        settings.collection_for(folder)
    extractors = build_extractors(settings)
    with Catalog(settings.catalog_path) as catalog:
        indexer = Indexer(settings, catalog, extractors, workers=workers)
        with _progress("Indexing") as update:
            report = indexer.index(resolved, progress=update)
    _render_index_report(report)


@app.command("locations")
def locations_command(
    folders: Annotated[
        list[Path] | None,
        typer.Argument(help="Folders inside a collection (default: every mounted collection)."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Location names, and the ones that are probably one place spelt several ways.

    Lists files only (nothing is decoded, nothing is written). Suggests
    location_aliases for calico.toml; check each one before using it.
    """
    settings = load_settings()
    if folders:
        roots = [normalise(p) for p in folders]
    else:
        roots = [c.root for c in settings.collections if c.root.is_dir()]
        if not roots:
            raise CalicoError(
                "No mounted collections. Add one with: calico collections add NAME ROOT"
            )
    counts: Counter[str] = Counter()
    folder_names: set[str] = set()
    for folder in roots:
        if not folder.is_dir():
            raise FolderNotFoundError(missing_folder(folder))
        collection = settings.collection_for(folder)
        for path in scan(folder, collection.exclude).images:
            counts[location_for(collection, path, settings.location_aliases)] += 1
            folder_names.update(path.relative_to(collection.root).parts[:-1])
    same = similar_places(counts)
    words = word_spellings(sorted(counts))
    stray = sorted(n for n in folder_names if has_stray_spaces(n))
    if as_json:
        report = {
            "locations": dict(sorted(counts.items())),
            "same_place": [dataclasses.asdict(g) for g in same],
            "words": [dataclasses.asdict(w) for w in words],
            "stray_spaces": stray,
        }
        typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
        return
    _render_locations(counts, same, words, stray)


def _render_locations(
    counts: Counter[str], same: list[SamePlace], words: list[WordSpellings], stray: list[str]
) -> None:
    console = _out()
    console.print(
        f"{_plural(len(counts), 'location')} across {_plural(counts.total(), 'photo')}.",
        markup=False,
    )
    if same:
        table = Table(
            "photos", "name", "why", title="Probably the same place", title_justify="left"
        )
        for group in same:
            for i, (name, photos) in enumerate(zip(group.names, group.photos, strict=True)):
                why = "suggested name" if i == 0 else ", ".join(group.reasons)
                table.add_row(str(photos), Text(name), why, end_section=i == len(group.names) - 1)
        console.print(table)
    if words:
        console.print("\nWords spelt more than one way:", markup=False)
        for w in words:
            uses = ", ".join(f"{s} ({n})" for s, n in zip(w.spellings, w.places, strict=True))
            console.print(Text(f"  {uses}   e.g. {w.examples[-1]}"), soft_wrap=True)
    if stray:
        console.print(
            f"\n{_plural(len(stray), 'folder name')} with stray spaces (Calico ignores them; "
            "listed in case you rename folders):",
            markup=False,
        )
        for name in stray[:MAX_FAILURES_SHOWN]:
            console.print(Text(f"  “{name}”"), soft_wrap=True)
        if len(stray) > MAX_FAILURES_SHOWN:
            console.print(f"  ... and {len(stray) - MAX_FAILURES_SHOWN} more (see --json)")
    if not same:
        return
    aliases = {old: g.names[0] for g in same for old in g.names[1:]}
    console.print(
        "\nTo use the suggested names, check each group, then add to calico.toml:", markup=False
    )
    console.print(Text(tomli_w.dumps({"location_aliases": aliases}).rstrip()), soft_wrap=True)
    if any(BRACKETS in g.reasons for g in same):
        console.print(
            "Names that differ only by a note in brackets can instead be joined for a whole "
            "collection: calico collections add NAME ROOT --location-ignore '\\([^)]*\\)'",
            markup=False,
            soft_wrap=True,
        )


@app.command("sandbox")
def sandbox_command(
    target: Annotated[Path, typer.Argument(help="Directory to create (archive/ and home/).")],
    embedder: Annotated[str, typer.Option(help="tiny (no PyTorch) or clip.")] = "tiny",
    force: Annotated[bool, typer.Option("--force", help="Replace TARGET if it exists.")] = False,
) -> None:
    """Create a synthetic archive with its own calico home, for trying things safely."""
    target = target.expanduser().resolve()
    if target.exists():
        if not force:
            raise CalicoError(f"{target} already exists. Pass --force to replace it.")
        home = default_home()
        reason = refuse_to_replace(target, home, _collection_roots(home))
        if reason is not None:
            raise CalicoError(reason)
        shutil.rmtree(target)
    settings = build_sandbox(target, embedder)
    _err().print("Sandbox ready. Use it in this shell with:")
    print(f"export CALICO_HOME={settings.home}")
    print(f"uv run calico select best {settings.collections[0].root}")


def _collection_roots(home: Path) -> list[Path]:
    """The collection roots ``home`` configures, if its settings can be read."""
    try:
        return [c.root for c in load_settings(home).collections]
    except CalicoError:
        return []


# -- tasks & models ------------------------------------------------------------------------


@app.command("tasks")
def tasks_command() -> None:
    """List the things you can ask for, with every option and its default."""
    registry: Registry[Any] = Registry.from_entry_points(TASKS_GROUP, "task")
    console = _out()
    for name in registry.names():
        try:
            task = registry.get(name)
        except ImportError as exc:
            console.print(Text(f"{name}: unavailable: {exc}", style="red"), soft_wrap=True)
            continue
        console.print(Text(f"{task.name}  {task.title}", style="bold"))
        console.print(Text(task.description), soft_wrap=True)
        console.print(f"reference photo: {task.reference_use.value}", markup=False)
        console.print(
            f"description counts: {task.default_describe_weight:.0%} (--describe-weight)",
            markup=False,
        )
        table = Table("option", "type", "default", "description", box=None, pad_edge=False)
        for field_name, field in task.options_model.model_fields.items():
            default = "required" if field.is_required() else repr(field.default)
            table.add_row(
                Text(field_name),
                Text(_type_name(field.annotation)),
                Text(default),
                Text(field.description or "", overflow="fold"),
            )
        console.print(table)
        console.print()


def _type_name(annotation: object) -> str:
    name = getattr(annotation, "__name__", None)
    return name if isinstance(name, str) and "[" not in str(annotation) else str(annotation)


@app.command("models")
def models_command() -> None:
    """List models (judgements about photos) and whether they can run here."""
    settings = load_settings()
    table = Table("model", "kind", "version", "labels / notes")
    with Engine.open(settings) as engine:
        for name in engine.models.names():
            try:
                model = engine.models.get(name)
            except CapabilityError as exc:
                table.add_row(Text(name), "", "", Text(f"unavailable: {exc}", overflow="fold"))
                continue
            if isinstance(model, LabelModel):
                table.add_row(
                    Text(name),
                    "label",
                    Text(model.version),
                    Text(", ".join(model.labels), overflow="fold"),
                )
            else:
                table.add_row(Text(name), "score", Text(model.version), Text(model.description))
    _out().print(table)


# -- select & runs -------------------------------------------------------------------------


def parse_option(raw: str) -> tuple[str, JsonValue]:
    """``KEY=VALUE`` -> (key, value). VALUE is JSON if it parses, a list if it has commas."""
    key, sep, value = raw.partition("=")
    if not sep or not key.strip():
        raise InvalidRequestError(f"Option {raw!r} must look like KEY=VALUE, e.g. -o diversity=0.5")
    try:
        parsed: JsonValue = json.loads(value)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in value.split(",")] if "," in value else value
    return key.strip(), parsed


def _evidence_text(pick_evidence: Sequence[Evidence]) -> str:
    top = sorted(pick_evidence, key=lambda e: e.contribution, reverse=True)[:3]
    return "\n".join(f"{e.signal} {e.value:.2f}\u00d7{e.weight:.2f}" for e in top)


_POSITIONS = ("left", "anchor", "right")
_SHORT_DETAIL = 40


def _group_evidence_text(evidence: Sequence[Evidence]) -> str:
    def one(e: Evidence) -> str:
        text = f"{e.signal} {e.value:.2f}\u00d7{e.weight:.2f}"
        return f"{text} ({e.detail})" if e.detail and len(e.detail) <= _SHORT_DETAIL else text

    return " \u00b7 ".join(one(e) for e in evidence)


def render_result(result: SelectionResult, *, footer: bool) -> None:
    console = _out()
    if not result.groups:
        console.print("No picks.", markup=False)
    for number, group in enumerate(result.groups, start=1):
        positions = _POSITIONS if len(group.picks) == len(_POSITIONS) else None
        likely = any(pick.probability is not None for pick in group.picks)
        table = Table(
            "position" if positions else "rank",
            "file",
            "location",
            "date",
            "score",
            *(["likely"] if likely else []),
            "evidence",
            title=Text(f"{number}. {group.title}  (score {group.score:.2f})"),
            title_justify="left",
        )
        for rank, pick in enumerate(group.picks, start=1):
            if group.marker is not None and group.marker.before == rank - 1:
                below = f"less than {group.marker.probability:.0%} likely to be one you like"
                table.add_row(
                    "",
                    Text(f"\u2014 from here, {below} \u2014", style="yellow", overflow="fold"),
                    *([""] * (5 if likely else 4)),
                    end_section=True,
                )
            table.add_row(
                positions[rank - 1] if positions else str(rank),
                Text(pick.path.name, overflow="fold"),
                Text(pick.location),
                pick.captured_at.strftime("%Y-%m-%d") if pick.captured_at else "-",
                f"{pick.score:.2f}",
                *(
                    ["-" if pick.probability is None else f"{pick.probability:.0%}"]
                    if likely
                    else []
                ),
                Text(_evidence_text(pick.evidence)),
            )
        console.print(table)
        if group.marker is not None and group.marker.before >= len(group.picks):
            console.print(
                f"Every pick is at least {group.marker.probability:.0%} likely to be one you like.",
                markup=False,
            )
        if group.evidence:
            console.print(Text("  " + _group_evidence_text(group.evidence)), soft_wrap=True)
    for note in result.notes:
        console.print(Text(f"note: {note}", style="yellow"), soft_wrap=True)
    if footer:
        console.print(
            f"run {result.run_id} saved — calico runs show {result.run_id}",
            markup=False,
            soft_wrap=True,
        )


@app.command("select")
def select_command(  # noqa: PLR0917
    task: Annotated[str, typer.Argument(help="Task name; see: calico tasks")],
    folders: Annotated[list[Path], typer.Argument(help="Folders to choose from.")],
    criteria: Annotated[
        str, typer.Option(help="What to look for, in words (needs a text-capable embedder).")
    ] = "",
    reference: Annotated[
        list[Path] | None,
        typer.Option("--reference", "-r", help="Reference photo; repeatable."),
    ] = None,
    example: Annotated[
        list[Path] | None,
        typer.Option(
            "--example",
            "-e",
            help="A photo in your collections that shows what you mean; repeatable. "
            "Part of the description, left out of the results.",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(min=1, max=500, help="How many results (default: the task's own)."),
    ] = None,
    option: Annotated[
        list[str] | None,
        typer.Option(
            "--option",
            "-o",
            help="Task option KEY=VALUE (JSON, or comma list); repeatable. See: calico tasks",
        ),
    ] = None,
    only_clear_matches: Annotated[
        bool,
        typer.Option(
            "--only-clear-matches",
            help="Consider only photos that clearly match --criteria (all, if none do).",
        ),
    ] = False,
    describe_weight: Annotated[
        float | None,
        typer.Option(
            min=0,
            max=1,
            help="How much --criteria counts, 0-1: its share of each photo's score "
            "(default: the task's own).",
        ),
    ] = None,
    template: Annotated[
        str | None,
        typer.Option(
            help="Whose taste ranks photos: general (zero-shot) or yours (learned from your "
            "ratings; see calico train taste). Default: the template in calico.toml."
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full result as JSON and nothing else.")
    ] = False,
) -> None:
    """Run TASK over FOLDER... and show the picks with the reasons for each."""
    try:
        request = SelectionRequest(
            task=task,
            folders=tuple(folders),
            criteria=criteria,
            only_clear_matches=only_clear_matches,
            describe_weight=describe_weight,
            template=template,
            examples=tuple(example or ()),
            references=tuple(reference or ()),
            limit=limit,
            options=dict(parse_option(o) for o in option or ()),
        )
    except ValidationError as exc:
        raise InvalidRequestError(f"Invalid request:\n{exc}") from exc
    with Engine.open(load_settings()) as engine, _progress("Indexing") as update:
        result = engine.select(request, progress=update)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        render_result(result, footer=True)


@runs_app.command("list")
def runs_list(
    limit: Annotated[int, typer.Option(min=1, help="How many runs, newest first.")] = 20,
) -> None:
    """List saved runs, newest first."""
    store = RunStore(load_settings().runs_dir)
    runs = store.recent(limit)
    for _, why in store.unreadable(limit):
        _err().print(f"Left out: {why}", markup=False, soft_wrap=True)
    console = _out()
    if not runs:
        console.print("No runs yet. Try: calico select best FOLDER", markup=False)
        return
    table = Table(
        Column("run", no_wrap=True),
        "created",
        "task",
        Column("criteria", no_wrap=True, overflow="ellipsis", max_width=30),
        "groups",
        "picks",
    )
    for run in runs:
        table.add_row(
            run.run_id,
            run.created_at.strftime("%Y-%m-%d %H:%M"),
            run.task,
            Text(run.criteria, overflow="ellipsis", no_wrap=True),
            str(run.groups),
            str(run.picks),
        )
    console.print(table)


@runs_app.command("show")
def runs_show(
    run_id: Annotated[str, typer.Argument(help="Run id; see: calico runs list")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the stored result as JSON.")
    ] = False,
) -> None:
    """Show a saved run exactly as it was returned."""
    result = RunStore(load_settings().runs_dir).load(run_id)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        render_result(result, footer=False)


@runs_app.command("explain")
def runs_explain(
    run_id: Annotated[str, typer.Argument(help="Run id; see: calico runs list")],
    name: Annotated[str, typer.Argument(help="Part of a file path, e.g. IMG_1234")],
) -> None:
    """Every signal recorded for photos matching NAME, picked or not."""
    rows = RunStore(load_settings().runs_dir).explain(run_id, name)
    if not rows:
        raise InvalidRequestError(
            f"No photo in run {run_id} has {name!r} in its path. Check the spelling "
            f"(any part of the path works), or see what was considered: calico runs show {run_id}"
        )
    console = _out()
    for row in rows[:MAX_EXPLAINED]:
        placement = row.get("picked", "")
        if placement:
            group, _, rank = placement.partition(".")
            verdict = f"picked: group {group}, rank {rank}"
        else:
            reason = left_out_because(row)
            verdict = f"not picked: {reason}" if reason else "not picked"
        table = Table("signal", "value", title=Text(row["path"]), title_justify="left")
        table.add_row("result", verdict)
        for key, value in row.items():
            if key not in {"path", "picked"}:
                table.add_row(Text(key), Text(value))
        console.print(table)
    if len(rows) > MAX_EXPLAINED:
        console.print(
            f"... {len(rows) - MAX_EXPLAINED} more matches; give a longer NAME to narrow.",
            markup=False,
        )


# -- doctor --------------------------------------------------------------------------------

Status = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str
    hint: str = ""


def run_checks(settings: Settings) -> list[Check]:
    """Everything ``calico doctor`` looks at, as data."""
    checks = [_python_check(), Check("calico", "ok", __version__)]
    checks.extend(_home_checks(settings))
    checks.extend(_collection_check(c, settings) for c in settings.collections)
    embedder_check, extractors = _embedder_check(settings)
    checks.append(embedder_check)
    checks.append(_raw_check())
    checks.append(_catalog_check(settings, extractors))
    checks.append(_runs_check(settings))
    checks.append(_labels_check(settings))
    checks.append(_ratings_check(settings))
    checks.append(_scratch_check(settings))
    checks.append(_backup_check(settings))
    return checks


def _python_check() -> Check:
    version = ".".join(str(n) for n in sys.version_info[:3])
    if sys.version_info >= MIN_PYTHON:
        return Check("python", "ok", version)
    return Check("python", "fail", version, "Calico needs Python 3.12+: uv python install 3.12")


def _home_checks(settings: Settings) -> list[Check]:
    home = (
        Check("home", "ok", str(settings.home))
        if settings.home.is_dir()
        else Check(
            "home", "warn", f"{settings.home} does not exist yet", "It is created on first use."
        )
    )
    if settings.config_path.exists():
        return [home, Check("settings", "ok", str(settings.config_path))]
    detail = f"{settings.config_path} not written; using defaults"
    return [
        home,
        Check("settings", "warn", detail, "Add a collection: calico collections add NAME ROOT"),
    ]


def _collection_check(collection: CollectionConfig, settings: Settings) -> Check:
    name = f"collection {collection.name}"
    used = settings.collection_for(collection.root)
    if used.name != collection.name and used.root == collection.root:
        return Check(
            name,
            "warn",
            f"{used.name!r} has the same root ({collection.root}) and is used for its photos, "
            "so this collection's location rule never applies",
            f"Remove one of them from {settings.config_path}, or give this one its own folder.",
        )
    if collection.root.is_dir():
        return Check(name, "ok", str(collection.root))
    return Check(
        name,
        "warn",
        f"{collection.root} not mounted",
        "Mount the drive, or fix the path: calico collections add NAME ROOT",
    )


def _embedder_check(settings: Settings) -> tuple[Check, dict[str, ImageExtractor] | None]:
    try:
        extractors = build_extractors(settings)
    except CapabilityError as exc:
        return Check("embedder", "fail", f"{settings.embedder}: {exc}", _embedder_hint()), None
    except ConfigError as exc:
        return Check("embedder", "fail", str(exc), "calico config set-embedder tiny"), None
    device = getattr(extractors[settings.embedder], "device", None)
    where = f", device {device}" if device else ""
    describer = extractors[settings.describe_embedder]
    if isinstance(describer, Embedder) and describer.supports_text:
        detail = f"{settings.embedder}{where}; “Describe it” read by {settings.describe_embedder}"
        return Check("embedder", "ok", detail), extractors
    detail = (
        f"{settings.embedder}{where}; “Describe it” is ignored: "
        f"{settings.describe_embedder} cannot read text"
    )
    return Check("embedder", "warn", detail, _embedder_hint()), extractors


def _raw_check() -> Check:
    libraw = raw_support()
    if libraw is not None:
        return Check("raw", "ok", f"RAW files are read by {libraw}")
    return Check("raw", "warn", "RAW files without a JPEG beside them are skipped", RAW_HINT)


def _embedder_hint() -> str:
    return (
        "uv sync --extra ml, then: calico config set-embedder clip && "
        "calico config set-embedder siglip --describe"
    )


def _plural(count: int, noun: str, plural: str | None = None) -> str:
    return f"{count} {noun if count == 1 else plural or noun + 's'}"


def _catalog_check(settings: Settings, extractors: dict[str, ImageExtractor] | None) -> Check:
    if not settings.catalog_path.exists():
        return Check("catalog", "warn", "no catalog yet", "Index a folder: calico index FOLDER")
    with Catalog(settings.catalog_path) as catalog:
        stats = catalog.stats()
    if extractors is not None:
        keys = [feature_key(ex) for ex in extractors.values()]
    else:
        keys = sorted(stats.features)
    coverage = {
        k: (100 * stats.features.get(k, 0) // stats.photos if stats.photos else 0) for k in keys
    }
    per_collection = ", ".join(f"{k} {v}" for k, v in sorted(stats.by_collection.items()))
    parts = [
        f"{_plural(stats.photos, 'photo')}" + (f" ({per_collection})" if per_collection else ""),
        "features " + ", ".join(f"{k} {v}%" for k, v in coverage.items()),
        _plural(stats.failures, "failure"),
    ]
    incomplete = stats.photos == 0 or any(v < 100 for v in coverage.values())
    if stats.failures:
        hint = "Unreadable files are listed by: calico index FOLDER"
    elif incomplete:
        hint = "Run: calico index FOLDER"
    else:
        hint = ""
    return Check("catalog", "warn" if hint else "ok", "; ".join(parts), hint)


def _runs_check(settings: Settings) -> Check:
    count = 0
    if settings.runs_dir.is_dir():
        count = sum(1 for d in settings.runs_dir.iterdir() if (d / "result.json").exists())
    return Check("runs", "ok", f"{_plural(count, 'saved run')}")


def _labels_check(settings: Settings) -> Check:
    found, unreadable = LabelStore(settings.labels_dir).scan()
    detail = _plural(len(found), "labelling set")
    if not unreadable:
        return Check("labels", "ok", detail)
    names = "; ".join(f"{path.name} ({why})" for path, why in unreadable)
    return Check(
        "labels",
        "warn",
        f"{detail}; {_plural(len(unreadable), 'cannot', 'cannot')} be read: {names}",
        f"They were left untouched. Open them with the Calico that wrote them, or move them "
        f"out of {settings.labels_dir}.",
    )


def _ratings_check(settings: Settings) -> Check:
    """The ratings journal: how many answers, and any line this Calico cannot read. Only
    reads: it never migrates (Calico does that the first time it opens the journal)."""
    journal = RatingsJournal(settings.ratings_dir)
    if not journal.exists():
        return Check("ratings", "ok", "no ratings journal yet; made the first time Calico opens")
    answers = sum(1 for _ in journal.answers())
    detail = f"{_plural(answers, 'answer')} in {journal.folder}"
    if not journal.skipped:
        return Check("ratings", "ok", detail)
    verb = "is" if journal.skipped == 1 else "are"
    return Check(
        "ratings",
        "warn",
        f"{detail}; {_plural(journal.skipped, 'line')} cannot be read and {verb} left out",
        "A line cut short by a crash, or written by a newer Calico: the log names each. "
        "Nothing was changed.",
    )


def _scratch_check(settings: Settings) -> Check:
    """Pair comparisons' scratch space: GBs while Recurring views runs, and left behind by a
    Calico that was killed (``Engine.open`` removes those; an older Calico's, it cannot)."""
    folders = scratch_folders(settings.home)
    if not folders:
        return Check("scratch", "ok", "no scratch space in use")
    in_use: list[Path] = []
    left: list[Path] = []
    older: list[Path] = []
    for folder in folders:
        pid = owner(folder)
        (older if pid is None else in_use if running(pid) else left).append(folder)
    size = size_text(sum(folder_size(f) for f in folders))
    parts = [
        f"{len(found)} {why}"
        for found, why in (
            (in_use, "in use by a running Calico"),
            (left, "left by a stopped Calico"),
            (older, "left by an older Calico"),
        )
        if found
    ]
    detail = f"{_plural(len(folders), 'folder')}, {size}: {'; '.join(parts)}"
    if not left and not older:
        return Check("scratch", "ok", detail)
    hints = []
    if left:
        hints.append("Calico removes what a stopped Calico left the next time it starts.")
    if older:
        named = " ".join(shlex.quote(str(f)) for f in older)
        hints.append(f"Once no older Calico is running, delete the rest: rm -r {named}")
    return Check("scratch", "warn", detail, " ".join(hints))


def _backup_check(settings: Settings) -> Check:
    """Your ratings, labels and runs cannot be recomputed: say when they were last saved, and
    how many ratings are newer."""
    noted = last_backup(settings)
    hint = "Back them up, ideally to another disk: " + _backup_command(noted)
    if noted is None:
        return Check("backup", "warn", "no backup made yet", hint)
    days = (datetime.now(UTC) - noted.at).days
    newer = sum(1 for a in RatingsJournal(settings.ratings_dir).answers() if a.at > noted.at)
    detail = f"last backup {_plural(days, 'day')} ago: {noted.path}"
    if newer:
        detail += f"; {_plural(newer, 'rating')} since"
    if days > BACKUP_STALE_DAYS or newer > BACKUP_STALE_ANSWERS:
        return Check("backup", "warn", detail, hint)
    return Check("backup", "ok", detail)


def _backup_command(noted: Noted | None) -> str:
    """The command to back up with: into the folder the last backup went into, when there is
    one. Not a word to copy: someone typed the placeholder it used to show."""
    if noted is None:
        return "calico backup <folder>"
    return f"calico backup {shlex.quote(str(noted.path.parent))}"


_STATUS_STYLE = {"ok": "green", "warn": "yellow", "fail": "red"}


@app.command("backup")
def backup_command(
    dest: Annotated[
        Path | None,
        typer.Argument(
            help="The folder to write the backup into, outside the home and your collections: "
            "ideally another disk.",
            show_default=False,
        ),
    ] = None,
    with_catalog: Annotated[
        bool,
        typer.Option(
            "--with-catalog",
            help="Also keep the catalog (features, thumbnail index), which can otherwise be "
            "rebuilt by indexing again.",
        ),
    ] = False,
    check: Annotated[
        Path | None,
        typer.Option(
            "--verify",
            exists=True,
            dir_okay=False,
            help="Read a backup file and list what a restore would bring back.",
        ),
    ] = None,
) -> None:
    """Back up what cannot be recomputed: your ratings, labels, runs, trained taste and
    settings, in one calico-backup-<date>.tar.gz."""
    console = _out()
    if check is not None:
        checked = verify(check)
        when = f"made {checked.created_at:%Y-%m-%d %H:%M} " if checked.created_at else ""
        console.print(f"{check}: {when}by Calico {checked.calico or '?'}", markup=False)
        console.print(_parts_table(checked.parts))
        for problem in checked.problems:
            _err().print(f"Problem: {problem}", markup=False, soft_wrap=True)
        if not checked.ok:
            raise typer.Exit(1)
        console.print("Every file in it reads.", markup=False)
        return
    if dest is None:
        raise typer.BadParameter(
            "Give the folder to back up into: a real one, outside the home and your "
            "collections, ideally on another disk. For example:\n  "
            + _backup_command(last_backup(load_settings())),
            param_hint="DEST",
        )
    made = backup(load_settings(), dest, with_catalog=with_catalog)
    console.print(f"Backed up to {made.path}", markup=False, soft_wrap=True)
    console.print(_parts_table(made.parts))
    if not made.catalog:
        console.print(
            "The catalog is not in it: it can be rebuilt with calico index (or add "
            "--with-catalog).",
            markup=False,
        )
    if made.uploads_gone:
        console.print(
            f"{_plural(made.uploads_gone, 'upload')} saved runs point to were already pruned "
            "(uploads are kept a week), so they are not in it.",
            markup=False,
            soft_wrap=True,
        )
    console.print(
        f"Check it with: calico backup --verify {made.path}", markup=False, soft_wrap=True
    )


def _parts_table(parts: Sequence[Part]) -> Table:
    table = Table("part", "files", "size")
    for part in parts:
        table.add_row(part.name, str(part.files), size_text(part.bytes))
    return table


@app.command("doctor")
def doctor_command(
    as_json: Annotated[bool, typer.Option("--json", help="Print the checks as JSON.")] = False,
) -> None:
    """Check the environment, embedder, collections and catalog. Exit 1 if anything fails."""
    settings = load_settings()
    names = dict.fromkeys((settings.embedder, settings.describe_embedder, settings.view_embedder))
    loading = [n for n in names if n != "tiny"]
    if loading:
        noun = "embedder" if len(loading) == 1 else "embedders"
        _err().print(f"Loading the {' and '.join(loading)} {noun}\u2026", markup=False)
    checks = run_checks(settings)
    if as_json:
        typer.echo(json.dumps([dataclasses.asdict(c) for c in checks], indent=2))
    else:
        console = _out()
        for check in checks:
            line = Text()
            line.append(f"{check.status:<4}", style=_STATUS_STYLE[check.status])
            line.append(f"  {check.name}: {check.detail}")
            console.print(line, soft_wrap=True)
            if check.hint:
                console.print(Text(f"      fix: {check.hint}", style="dim"), soft_wrap=True)
    if any(c.status == "fail" for c in checks):
        raise typer.Exit(1)


# -- serve ---------------------------------------------------------------------------------


def open_browser_soon(url: str) -> None:
    """Open the browser shortly after the server starts listening."""
    timer = threading.Timer(1.0, webbrowser.open, [url])
    timer.daemon = True
    timer.start()


def _is_this_computer(host: str) -> bool:
    """Whether ``host`` is a loopback address, which no other computer can reach."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"
"""Where ``calico serve`` listens unless told otherwise; the label hints name it."""


def _port_taken(host: str, port: int) -> bool:
    """Whether a server already listens on ``port``. A second server cannot take the port
    from the first, which goes on answering the browser. SO_REUSEADDR, as uvicorn binds,
    so a server stopped a moment ago does not count."""
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            return exc.errno == errno.EADDRINUSE
    return False


def _serves_calico(url: str) -> bool:
    """Whether what holds the port answers Calico's health check."""
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=2) as response:  # noqa: S310 - our own loopback URL
            return "api" in json.loads(response.read())
    except (OSError, ValueError, TypeError):
        return False


@app.command("serve")
def serve_command(
    host: Annotated[
        str, typer.Option(help="This computer's address to listen on: 127.0.0.1, ::1 or localhost.")
    ] = DEFAULT_HOST,
    port: Annotated[int, typer.Option(min=1, max=65535, help="Port to listen on.")] = DEFAULT_PORT,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Do not open a browser.")
    ] = False,
) -> None:
    """Start the web interface, for this computer only."""
    import uvicorn  # noqa: PLC0415 - only `serve` needs the server

    if not _is_this_computer(host):
        raise CalicoError(
            f"Calico serves only this computer, and {host} can be reached from others: it has "
            "no login, so anyone on the network could browse your photos or delete your "
            "labels. Use --host 127.0.0.1 (the default), ::1 or localhost. To use Calico from "
            "another device, open a tunnel from that device, "
            f"ssh -L {port}:127.0.0.1:{port} <this computer>, then browse "
            f"http://127.0.0.1:{port}."
        )
    url = f"http://{f'[{host}]' if ':' in host else host}:{port}"
    if _port_taken(host, port) and _serves_calico(url):
        raise CalicoError(
            f"Calico is already served at {url}: open it there. If it came back after you "
            "stopped it, a service that restarts it is running an installed copy. To run this "
            f"checkout's code beside it, pass another port: --port {port + 1}."
        )
    if _port_taken(host, port):
        raise CalicoError(
            f"Port {port} is already in use, most likely by a Calico server that is still "
            "running and keeps answering with the code it started with. Stop it (Ctrl-C in "
            f"its terminal, or: kill $(lsof -ti tcp:{port})) and run calico serve again, or "
            "pass another --port."
        )
    create_app = importlib.import_module("calico.web.app").create_app
    web_app = create_app(load_settings())
    _out().print(f"Serving on {url} (Ctrl-C to stop)", markup=False)
    if not no_browser:
        open_browser_soon(url)
    uvicorn.run(web_app, host=host, port=port)


# -- eval ----------------------------------------------------------------------------------

FitChoice = Literal["centred", "offset", "leaning", "by_scale", "top_pairs"]

METRICS = ("precision_at_10", "ndcg_at_10", "auc")


@eval_app.command("new")
def eval_new(
    text: Annotated[str, typer.Argument(help="What to look for, as you would type it.")],
    folders: Annotated[list[Path], typer.Argument(help="Folders inside a collection.")],
) -> None:
    """Sample photos from FOLDER... to label against TEXT (labelling is in the web interface)."""
    with Engine.open() as engine, _progress("Indexing") as update:
        made = Sessions(engine).label_set(text, folders, progress=update)
    console = _out()
    console.print(
        f"Created {made.set_id}: {_plural(len(made.items), 'photo')} to label for “{made.text}”.",
        markup=False,
    )
    console.print(f"Label them at {DEFAULT_URL}/label/{made.set_id} (calico serve)", markup=False)


@eval_app.command("new-view")
def eval_new_view(
    reference: Annotated[Path, typer.Argument(help="The view: a photo to compare against.")],
    folders: Annotated[list[Path], typer.Argument(help="Folders inside a collection.")],
) -> None:
    """Sample photos from FOLDER... to judge as the same view as REFERENCE or not, so
    `calico eval same-view` can say where min_similarity belongs."""
    with Engine.open() as engine, _progress("Indexing") as update:
        made = Sessions(engine).label_same_view(reference, folders, progress=update)
    console = _out()
    console.print(
        f"Created {made.set_id}: {_plural(len(made.items), 'photo')} to label for “{made.text}”.",
        markup=False,
    )
    console.print(f"Label them at {DEFAULT_URL}/label/{made.set_id} (calico serve)", markup=False)


@eval_app.command("new-taste")
def eval_new_taste(
    folders: Annotated[list[Path], typer.Argument(help="Folders inside a collection.")],
) -> None:
    """Sample photos from FOLDER... to rate for your taste (like, not sure, don't like),
    to train and test the Yours template: mostly what General and Yours would show first,
    above all where they differ, and some at random from the whole place."""
    with Engine.open() as engine, _progress("Indexing") as update:
        made = Sessions(engine).label_taste(folders, progress=update)
    console = _out()
    console.print(
        f"Created {made.set_id}: {_plural(len(made.items), 'photo')} to rate for “{made.text}” "
        f"({made.scorer}).",
        markup=False,
    )
    console.print(f"Rate them at {DEFAULT_URL}/label/{made.set_id} (calico serve)", markup=False)


@eval_app.command("new-list")
def eval_new_list(
    file: Annotated[
        Path, typer.Argument(help="A CSV file with a column named photo: one path a row.")
    ],
    name: Annotated[
        str | None, typer.Option(help="What to call the session. Default: the file's name.")
    ] = None,
) -> None:
    """Make a session of the photos FILE lists, to rate for your taste (like, not sure,
    don't like). The photos are shown once each, shuffled; the session keeps only the
    photos, so a list of two versions' picks is rated without knowing whose each is. The
    ratings train Yours, as Compare Strongest's do."""
    try:
        with file.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise CalicoError(f"Cannot read {file}: {exc.strerror}. Check the path.") from exc
    if not rows or "photo" not in rows[0]:
        raise CalicoError(
            f"{file} needs a first line of column names with a column named photo, and the "
            "path of one photo on each line after it."
        )
    paths = [Path(row["photo"]) for row in rows if row["photo"]]
    with Engine.open() as engine:
        made = Sessions(engine).label_list(paths, name or file.stem)
    console = _out()
    console.print(
        f"Created {made.set_id}: {_plural(len(made.items), 'photo', 'photos')} to rate "
        f"for “{made.text}”.",
        markup=False,
    )
    console.print(f"Rate them at {DEFAULT_URL}/label/{made.set_id} (calico serve)", markup=False)


@eval_app.command("new-anchors")
def eval_new_anchors(
    folders: Annotated[list[Path], typer.Argument(help="Folders inside a collection.")],
) -> None:
    """Sample photos from FOLDER... to rate whether each could anchor a story (could, not
    sure, could not), each kind of photo in turn, spread over today's general strength."""
    with Engine.open() as engine, _progress("Indexing") as update:
        made = Sessions(engine).label_anchors(folders, progress=update)
    console = _out()
    console.print(
        f"Created {made.set_id}: {_plural(len(made.items), 'photo')} to rate for “{made.text}”.",
        markup=False,
        soft_wrap=True,
    )
    if "without" in made.scorer:
        console.print(f"Sampled {made.scorer}.", markup=False, soft_wrap=True)
    console.print(f"Rate them at {DEFAULT_URL}/label/{made.set_id} (calico serve)", markup=False)


@eval_app.command("anchors")
def eval_anchors(
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    holdout: Annotated[
        bool,
        typer.Option(
            "--holdout",
            help="Report on the hold-out alone: a last check of a final candidate, never for "
            "tuning (calico holdout show).",
        ),
    ] = False,
) -> None:
    """Does Calico already measure what makes an anchor? AUC: how often a photo you said
    could anchor a story outscores one you said could not, at the same place, by each
    signal. If none does well, anchors need a model of their own."""
    with Engine.open() as engine:
        sessions = Sessions(engine)
        report = sessions.evaluate_anchors(only_holdout=holdout)
        held = _held_out(sessions, holdout)
        tried = _record_tried(engine.settings, "anchors", ["defaults"], holdout)
    if as_json:
        out = {**dataclasses.asdict(report), "holdout": held, "tried": dataclasses.asdict(tried)}
        typer.echo(json.dumps(_nan_to_null(out), indent=2, allow_nan=False))
        return
    console = _out()
    console.print(held, markup=False, soft_wrap=True)
    _print_tried(console, "anchors", tried, holdout)
    console.print(
        f"{report.could} could anchor, {report.could_not} could not, at "
        f"{_plural(report.places, 'place')}.",
        markup=False,
        soft_wrap=True,
    )
    if report.skipped:
        console.print(
            "Left out, since their folders hold no catalogued photos now: "
            + "; ".join(report.skipped)
            + ". Mount the drive, or index the folders again (calico index FOLDER).",
            markup=False,
            soft_wrap=True,
        )
    measured = report.measured
    table = Table(
        "signal", "could over could not (AUC)", "average of places", "pairs", title="Anchors"
    )
    for s in report.signals:
        table.add_row(
            s.signal,
            _fmt(s.auc) if measured is None else _range(measured.auc[s.signal]),
            "" if measured is None else _fmt(measured.macro[s.signal]),
            str(s.pairs),
        )
    console.print(table)
    if measured is not None:
        _print_signals_note(console, measured)


@eval_app.command("new-stories")
def eval_new_stories(
    folders: Annotated[list[Path], typer.Argument(help="Folders inside a collection.")],
    variant: Annotated[
        str | None,
        typer.Option(
            help="Tell a named candidate's stories (its options, from -o) beside today's way, "
            "under --template alone, instead of the way under test. calico eval stories "
            "reports it as candidate:NAME.",
        ),
    ] = None,
    template: Annotated[
        str,
        typer.Option(help="With --variant: the template its stories are told under."),
    ] = "yours",
    option: Annotated[
        list[str] | None,
        typer.Option(
            "--option",
            "-o",
            help="With --variant: a story option KEY=VALUE (JSON, or comma list); repeatable.",
        ),
    ] = None,
) -> None:
    """Tell stories from FOLDER... to rate whole (works, not sure, doesn't), under each
    template, today's way and the way under test, or today's way and a candidate's
    (--variant). Rate one place at a time."""
    candidate = None
    if variant is not None:
        if template not in ("general", "yours"):
            raise InvalidRequestError(f"--template is general or yours, not {template!r}.")
        candidate = StoryVariant(
            name=variant,
            template="yours" if template == "yours" else "general",
            options=dict(parse_option(o) for o in option or ()),
        )
    elif option:
        raise InvalidRequestError(
            "-o sets a candidate's options: name the candidate with --variant NAME."
        )
    with Engine.open() as engine, _progress("Indexing") as update:
        made = Sessions(engine).label_stories(folders, progress=update, variant=candidate)
    console = _out()
    console.print(
        f"Created {made.set_id}: {_plural(len(made.stories), 'story', 'stories')} to rate "
        f"for “{made.text}”.",
        markup=False,
    )
    console.print(f"Rate them at {DEFAULT_URL}/label/{made.set_id} (calico serve)", markup=False)


@eval_app.command("new-compare")
def eval_new_compare(
    folders: Annotated[list[Path], typer.Argument(help="Folders inside a collection.")],
    picks: Annotated[
        int,
        typer.Option(
            min=2,
            help="How many of each list's first picks to show (the bar is read at 30). 60 "
            "also gives the liked share of the first 60, for about twice the photos to rate.",
        ),
    ] = COMPARE_PICKS,
    cross_fit: Annotated[
        bool,
        typer.Option(
            "--cross-fit",
            help="Show Yours fitted without these folders' places and the places next to them, "
            "not the saved Yours: a place Yours has already learned is then fresh to it. "
            "Reported apart by calico eval compare. Needs ratings at other places, not a "
            "trained Yours.",
        ),
    ] = False,
) -> None:
    """Show the first picks of General and of Yours at FOLDER..., merged and shuffled, to
    rate like, not sure or don't like without knowing whose they are. Says whose first
    picks you prefer (calico eval compare), and trains Yours. Rate one place at a time."""
    with Engine.open() as engine, _progress("Indexing") as update:
        made = Sessions(engine).label_compare(
            folders, progress=update, per_template=picks, cross_fit=cross_fit
        )
    console = _out()
    console.print(
        f"Created {made.set_id}: {_plural(len(made.items), 'photo', 'photos')} to rate "
        f"for “{made.text}”.",
        markup=False,
    )
    if cross_fit:
        console.print(
            "Yours here is fitted without these places and the places next to them, not the "
            "saved Yours: calico eval compare reports it apart.",
            markup=False,
            soft_wrap=True,
        )
    console.print(f"Rate them at {DEFAULT_URL}/label/{made.set_id} (calico serve)", markup=False)


_DISAGREEMENT_GROUPS = (
    ("general_missed", "General misses", "Liked, and Yours ranked it far above General"),
    (
        "general_wrong",
        "General's picks that are not portfolio worthy",
        "Not liked, and General ranked it far above Yours",
    ),
    ("yours_missed", "Yours misses", "Liked, and General ranked it far above Yours"),
    (
        "yours_wrong",
        "Yours's picks that are not portfolio worthy",
        "Not liked, and Yours ranked it far above General",
    ),
)
"""Each kind of disagreement (a ``Disagreements`` field), named, then described."""


@eval_app.command("disagreements")
def eval_disagreements(
    embedder: Annotated[
        str | None,
        typer.Option(help="The embedding Yours is learned on (default: the one Yours uses)."),
    ] = None,
    count: Annotated[
        int, typer.Option(min=1, help="How many of each kind of disagreement to show.")
    ] = DISAGREEMENT_COUNT,
    sheet: Annotated[
        Path | None,
        typer.Option(
            help="Also write a contact sheet of them to this HTML file (thumbnails inside, so "
            "it opens anywhere). Never inside a collection."
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Which rated photos General and Yours rank furthest apart, and which one you sided
    with: what Yours sees that General misses, and General's picks that are not portfolio
    worthy (audit plan S5). Each place is judged by a Yours that never learned from it.
    """
    with Engine.open() as engine:
        settings = engine.settings
        target = None if sheet is None else normalise(sheet)
        for collection in settings.collections:
            if target is not None and collection.contains(target):
                raise InvalidRequestError(
                    f"{sheet} is inside the {collection.name!r} collection ({collection.root}), "
                    "and Calico never writes into your archive. Choose a file outside it."
                )
        found, paths = Sessions(engine).taste_disagreements(embedder, count=count)
    shown = [d for field, _, _ in _DISAGREEMENT_GROUPS for d in getattr(found, field)]
    missing = 0
    if target is not None:
        missing = _write_sheet(target, found, paths, settings.thumbs_dir)
    if as_json:
        out: dict[str, object] = {
            field: [
                {**dataclasses.asdict(d), "path": str(paths.get(d.content_key, ""))}
                for d in getattr(found, field)
            ]
            for field, _, _ in _DISAGREEMENT_GROUPS
        }
        out |= {
            "counts": dict(zip((f for f, _, _ in _DISAGREEMENT_GROUPS), found.counts, strict=True)),
            "places": found.places,
            "rated": found.rated,
            "one_sided": list(found.one_sided),
            "gap": DISAGREEMENT_GAP,
            "sheet": None if target is None else str(target),
        }
        typer.echo(json.dumps(out, indent=2))
        return
    console = _out()
    if not found.places:
        console.print(
            "No taste ratings yet that can be judged: rate at least two places with Rate for "
            "your taste (the Label page) or: calico eval new-taste FOLDER",
            markup=False,
        )
        return
    console.print(
        f"At {_plural(found.places, 'place')} ({found.rated} rated photos), each judged by a "
        "Yours that never learned from it. Strength as a rank among the place's rated "
        f"photos, 0 the weakest and 1 the strongest; apart by {DISAGREEMENT_GAP:g} or more.",
        markup=False,
        soft_wrap=True,
    )
    for (field, name, described), total in zip(_DISAGREEMENT_GROUPS, found.counts, strict=True):
        group: tuple[Disagreement, ...] = getattr(found, field)
        if not group:
            console.print(f"{name}: none ({described}).", markup=False, soft_wrap=True)
            continue
        table = Table(
            "place",
            "photo",
            "General",
            "Yours",
            title=f"{name}: {described} ({len(group)} of {total})",
        )
        for d in group:
            path = paths.get(d.content_key)
            table.add_row(
                Text(d.location),
                Text(str(path) if path else d.content_key),
                _fmt(d.general),
                _fmt(d.yours),
            )
        console.print(table)
    if found.one_sided:
        console.print(
            f"Not judged (only likes or only dislikes): {', '.join(found.one_sided)}.",
            markup=False,
            soft_wrap=True,
        )
    if target is not None:
        console.print(f"Contact sheet: {target}", markup=False, soft_wrap=True)
        if missing:
            console.print(
                f"{_plural(missing, 'photo')} of {len(shown)} had no thumbnail and are named "
                "only: calico index makes them.",
                markup=False,
                soft_wrap=True,
            )


def _write_sheet(target: Path, found: Disagreements, paths: dict[str, Path], thumbs: Path) -> int:
    """A contact sheet of every disagreement shown, thumbnails inside, so it opens anywhere;
    how many photos had no thumbnail (named instead)."""
    missing = 0
    sections = []
    for (field, name, described), total in zip(_DISAGREEMENT_GROUPS, found.counts, strict=True):
        cells = []
        for d in getattr(found, field):
            path = paths.get(d.content_key)
            label = html.escape(f"{d.location} · {path.name if path else d.content_key}")
            thumb = thumbnail_path(thumbs, d.content_key)
            if thumb.is_file():
                data = base64.b64encode(thumb.read_bytes()).decode("ascii")
                image = f'<img src="data:image/jpeg;base64,{data}" alt="{label}">'
            else:
                missing += 1
                image = '<div class="none">no thumbnail</div>'
            cells.append(
                f"<figure>{image}<figcaption>{label}<br>General {d.general:.2f} · Yours "
                f"{d.yours:.2f}</figcaption></figure>"
            )
        sections.append(
            f"<h2>{html.escape(name)}</h2><p>{html.escape(described)}: {len(cells)} of "
            f'{total}.</p><div class="grid">{"".join(cells)}</div>'
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        '<!doctype html><meta charset="utf-8"><title>Where General and Yours disagree</title>'
        "<style>body{font:14px system-ui;margin:16px;background:#fff;color:#111}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}"
        "figure{margin:0}img{width:100%;height:auto;display:block}"
        ".none{aspect-ratio:3/2;background:#eee;display:grid;place-items:center}"
        "figcaption{font-size:12px;color:#555;margin-top:4px}</style>"
        f"<h1>Where General and Yours disagree</h1><p>At {found.places} places, "
        f"{found.rated} rated photos.</p>{''.join(sections)}",
        encoding="utf-8",
    )
    return missing


@eval_app.command("retest")
def eval_retest(
    as_json: Annotated[bool, typer.Option("--json", help="Print it as JSON.")] = False,
) -> None:
    """How consistent you are: the control photos asked again, against how you rated them.
    A perfect ranker cannot beat that, whatever the bar says."""
    with Engine.open() as engine:
        report = Sessions(engine).retest()
    if as_json:
        typer.echo(json.dumps(dataclasses.asdict(report), indent=2, default=str))
        return
    _print_retest(_out(), report)


def _print_retest(console: Console, r: Retest) -> None:
    def shares(s: Share) -> str:
        if not s.n:
            return "none yet"
        lo, hi = s.interval
        return f"{s.kept} of {s.n} ({s.value:.0%}; 95% interval {lo:.0%} to {hi:.0%})"

    console.print(
        f"Control photos asked again: {_plural(r.asked, 'answer')} to "
        f"{_plural(r.photos, 'photo')} in {_plural(r.sessions, 'session')}.",
        markup=False,
    )
    if not r.asked:
        console.print(
            "Nothing has been asked yet: each Compare Strongest session made after the pool "
            "was drawn asks some of it.",
            markup=False,
            soft_wrap=True,
        )
        return
    console.print(f"Agree with how you rated them: {shares(r.against_baseline)}", markup=False)
    console.print(f"  liked then, liked again: {shares(r.from_liked)}", markup=False)
    console.print(f"  not liked then, not liked again: {shares(r.from_disliked)}", markup=False)
    console.print(
        f"Consecutive asks of one photo that agree: {shares(r.between_asks)}", markup=False
    )
    if r.ceiling is None:
        console.print("The ceiling for a perfect ranker's first picks: not yet.", markup=False)
    else:
        console.print(
            "If the noise were symmetric, a perfect ranker's first picks could be liked "
            f"{r.ceiling.value:.0%} of the time ({r.ceiling.lo:.0%} to {r.ceiling.hi:.0%}) at "
            "most: the ceiling the bar sits under.",
            markup=False,
            soft_wrap=True,
        )
    if r.asked > r.photos:
        console.print(
            f"{r.asked} answers are to {r.photos} photos, so the intervals are narrower than "
            "the evidence: they treat every answer as another photo.",
            markup=False,
            soft_wrap=True,
        )
    for s in r.by_session:
        console.print(
            f"  {s.at:%Y-%m-%d %H:%M}: {s.liked} of {s.n} liked (the same photos, other weeks)",
            markup=False,
        )


@eval_app.command("compare")
def eval_compare(
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    holdout: Annotated[
        bool,
        typer.Option(
            "--holdout",
            help="Report on the hold-out alone: a last check of a final candidate, never for "
            "tuning (calico holdout show).",
        ),
    ] = False,
    min_bursts: Annotated[
        int | None,
        typer.Option(
            "--min-bursts",
            min=1,
            help="Report apart the places whose pool held fewer bursts than this (the number "
            "of distinct moments before one pick per view, shown in the table). The roadmap "
            "reads the bar at places of 100 or more; no size is built in.",
        ),
    ] = None,
) -> None:
    """Whose first picks did you like more? For each Compare Strongest session (calico eval
    new-compare), the photos you liked of those you rated (Not sure is left out) among the
    first 30 of General's picks and Yours's, and among the photos only one of them chose.
    Sessions at a place the taste had already learned from are reported apart: there it is
    judged on what it learned. So are sessions whose Yours was fitted without their place
    (calico eval new-compare --cross-fit)."""
    with Engine.open() as engine:
        sessions = Sessions(engine)
        report = sessions.evaluate_compare(only_holdout=holdout, min_bursts=min_bursts)
        held = _held_out(sessions, holdout)
        tried = _record_tried(engine.settings, "compare", ["defaults"], holdout)
    if as_json:
        out = _compare_json(report)
        out["in_sample"] = None if report.in_sample is None else _compare_json(report.in_sample)
        out["small"] = None if report.small is None else _compare_json(report.small)
        out["cross_fitted"] = (
            None if report.cross_fitted is None else _compare_cross_fitted_json(report.cross_fitted)
        )
        out["holdout"] = held
        out["tried"] = dataclasses.asdict(tried)
        typer.echo(json.dumps(_nan_to_null(out), indent=2, allow_nan=False))
        return
    console = _out()
    console.print(held, markup=False, soft_wrap=True)
    _print_tried(console, "compare", tried, holdout)
    apart = report.in_sample
    small = report.small
    crossed = report.cross_fitted
    if (
        not report.rows
        and (apart is None or not apart.rows)
        and (small is None or not small.rows)
        and (crossed is None or not (crossed.rows or (crossed.small and crossed.small.rows)))
    ):
        console.print(
            "No Compare Strongest ratings yet: rate a place with Compare Strongest (the Label "
            "page) or: calico eval new-compare FOLDER",
            markup=False,
        )
        return
    if report.rows:
        _print_compare(
            console, report, f"Liked of rated, among each template's first {BAR_AT} picks"
        )
    if small is not None and small.rows:
        _print_compare(
            console,
            small,
            f"Fewer than {min_bursts} bursts, measured apart: too small a place for the bar",
        )
    if crossed is not None:
        whole = (
            "Yours fitted without the place: sessions at a place the saved Yours learned "
            "from, judged by a Yours that had not"
        )
        small_title = f"Fewer than {min_bursts} bursts, Yours fitted without the place"
        for part, title in ((crossed, whole), (crossed.small, small_title)):
            if part is not None and part.rows:
                _print_compare(console, part, title)
    if apart is not None and apart.rows:
        _print_compare(
            console,
            apart,
            "In-sample: sessions at a place your taste had learned from (or a taste that "
            "did not record its places)",
        )
    console.print(
        "A photo both chose counts for both; the only-columns are where the two differ. "
        f"The table and its shares read each list's first {BAR_AT}; the precision table reads "
        "more of a longer list, at the places that have it. "
        "[lo, hi]: where 95% of a difference fell when the places, then the photos within "
        "them, were drawn again.",
        markup=False,
        soft_wrap=True,
    )
    _print_unsized(console, report, min_bursts)
    if crossed is not None:
        _print_unsized(console, crossed, min_bursts)
    for place in [
        *report.empty,
        *(apart.empty if apart is not None else []),
        *(crossed.empty if crossed is not None else []),
    ]:
        console.print(f"{place}: nothing rated Like or Don't like yet.", markup=False)


def _print_unsized(console: Console, report: CompareReport, min_bursts: int | None) -> None:
    """Say which places a size was asked of but have none to go by (their saved run cannot be
    read, or predates the stage that counts bursts): they stay in the first table, not set apart."""
    if min_bursts is None:
        return
    unsized = [r.place for r in report.rows if r.bursts is None]
    if unsized:
        console.print(
            f"No burst count for {', '.join(unsized)} (its saved run cannot say), so "
            f"{'it stays' if len(unsized) == 1 else 'they stay'} in its table, not "
            f"set apart as under {min_bursts} bursts.",
            markup=False,
            soft_wrap=True,
        )


def _compare_cross_fitted_json(report: CompareReport) -> dict[str, Any]:
    """The cross-fitted sessions' report, with the places under a size apart in its own key."""
    out = _compare_json(report)
    out["small"] = None if report.small is None else _compare_json(report.small)
    return out


def _compare_json(report: CompareReport) -> dict[str, Any]:
    return {
        "rows": [dataclasses.asdict(r) for r in report.rows],
        "general": _first_json(report.general),
        "yours": _first_json(report.yours),
        "general_only": _first_json(report.general_only),
        "yours_only": _first_json(report.yours_only),
        "empty": report.empty,
        "shown": None if report.shown is None else dataclasses.asdict(report.shown),
        "only": None if report.only is None else dataclasses.asdict(report.only),
        "macro": dict(report.macro),
        "curve": {t: [_first_json(b) for b in buckets] for t, buckets in report.curve.items()},
        "fall": {t: dataclasses.asdict(gap) for t, gap in report.fall.items()},
        "precision": {k: dataclasses.asdict(c) for k, c in report.precision.items()},
        "drop": {
            t: {"positions": list(d.positions), "median": d.median} for t, d in report.drop.items()
        },
        "marker": None
        if report.marker is None
        else {
            "above": _first_json(report.marker.above),
            "below": _first_json(report.marker.below),
            "difference": dataclasses.asdict(report.marker.difference),
            "at": list(report.marker.at),
            "fell": list(report.marker.fell),
        },
    }


def _print_compare(console: Console, report: CompareReport, title: str) -> None:
    table = Table(
        "place", "bursts", "rated", "General", "Yours", "only General", "only Yours", title=title
    )
    for r in report.rows:
        table.add_row(
            Text(r.place),
            "" if r.bursts is None else str(r.bursts),
            str(r.rated),
            *(_liked_of(f) for f in (r.general, r.yours, r.general_only, r.yours_only)),
        )
    table.add_row(
        Text("all places", style="bold"),
        "",
        str(sum(r.rated for r in report.rows)),
        *(
            _liked_of(f, share=True)
            for f in (report.general, report.yours, report.general_only, report.yours_only)
        ),
    )
    if report.macro:
        table.add_row(
            Text("average of places"),
            "",
            "",
            _fmt(report.macro["general"]),
            _fmt(report.macro["yours"]),
            "",
            "",
        )
    console.print(table)
    if report.shown is not None and report.only is not None:
        console.print(
            f"Yours minus General, liked of their picks: {_difference(report.shown.difference)}; "
            f"of the picks only one chose: {_difference(report.only.difference)}.",
            markup=False,
            soft_wrap=True,
        )
    if report.curve:
        _print_compare_curve(console, report)
    _print_compare_precision(console, report)


def _print_compare_precision(console: Console, report: CompareReport) -> None:
    """The bar read as the owner uses a list: the liked share of its first 10, 20, 30 and 60,
    where the share so far first falls under 80%, and whether Yours's line sits there."""
    if report.precision:
        table = Table(
            "list",
            *(f"first {k} ({_plural(c.a.places, 'place')})" for k, c in report.precision.items()),
            title="Liked of rated in the first picks of each list, where the lists reach",
        )
        table.add_row("General", *(_range(c.b) for c in report.precision.values()))
        table.add_row("Yours", *(_range(c.a) for c in report.precision.values()))
        table.add_row(
            "Yours minus General", *(_signed_range(c.difference) for c in report.precision.values())
        )
        console.print(table)
    if report.drop:
        parts = []
        for template in ("yours", "general"):
            drop = report.drop.get(template)
            if drop is None:
                continue
            lists = _plural(len(drop.positions), "list")
            parts.append(
                f"{template.capitalize()} at pick {drop.median:g} (median; it fell in "
                f"{len(drop.fell)} of {lists})"
                if drop.fell
                else f"{template.capitalize()} never fell within its list ({lists})"
            )
        console.print(
            f"Where the liked share of the picks so far first falls below {RUNNING_LINE:.0%} "
            f"(read from the {RUNNING_AFTER}th rated pick): {'; '.join(parts)}.",
            markup=False,
            soft_wrap=True,
        )
    check = report.marker
    if check is not None:
        fell = [p for p in check.fell if p is not None]
        console.print(
            f"Yours's marker, drawn after pick {_median(check.at):g} in the median of "
            f"{_plural(len(check.at), 'list')}: liked above it "
            f"{_liked_of(check.above, share=True)}, "
            f"below it {_liked_of(check.below, share=True)} (above minus below "
            f"{_difference(check.difference.difference)}). The share so far first fell under "
            f"{RUNNING_LINE:.0%} at "
            + (f"pick {_median(fell):g} in the median" if fell else "no pick")
            + f" ({len(fell)} of {len(check.fell)} lists).",
            markup=False,
            soft_wrap=True,
        )


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _print_compare_curve(console: Console, report: CompareReport) -> None:
    """How the liked share falls down each list, in tens (``evaluation.CURVE_STEP``)."""
    tens = max(len(buckets) for buckets in report.curve.values())
    table = Table(
        "list",
        *(f"{k * CURVE_STEP + 1}-{(k + 1) * CURVE_STEP}" for k in range(tens)),
        "first minus last",
        title="Liked of rated, down each list",
    )
    for template in ("general", "yours"):
        if template not in report.curve:
            continue
        buckets = report.curve[template]
        gap = report.fall.get(template)
        table.add_row(
            template.capitalize(),
            *(_liked_of(b, share=True) if b.rated else "-" for b in buckets),
            *[""] * (tens - len(buckets)),
            "" if gap is None else _difference(gap),
        )
    console.print(table)
    console.print(
        "A list made before 11 November 2026 was shown in the order variety chose, not strongest "
        "first, so it is flat: a fall is read from the lists made since.",
        markup=False,
        soft_wrap=True,
    )


@eval_app.command("taste")
def eval_taste(  # noqa: PLR0917
    embedder: Annotated[
        list[str] | None,
        typer.Option(
            help="The embedding a taste is learned on; repeatable (default: each one Calico "
            "computes for every photo). Another, such as clip_large (CLIP ViT-L/14; its first "
            "run downloads about 1.7 GB), is computed for the rated photos alone, without "
            "Strongest's first picks."
        ),
    ] = None,
    l2: Annotated[
        list[float] | None,
        typer.Option(min=0, help="How strongly the weights are held back; repeatable."),
    ] = None,
    quality_weight: Annotated[
        list[float] | None,
        typer.Option(
            min=0,
            max=1,
            help="Quality's share of strength; repeatable (default: Strongest's).",
        ),
    ] = None,
    technical: Annotated[
        Literal["2", "3"] | None,
        typer.Option(
            help="Also measure General's strength with this version of the technical "
            "measures (3: sharpness with the noise taken off), against today's. Computed "
            "for the rated photos alone; the first run reads each one.",
        ),
    ] = None,
    aesthetic: Annotated[
        str | None,
        typer.Option(
            help="Also measure General with this extractor's score as its aesthetic, against "
            "today's (laion_aesthetic: LAION's aesthetic predictor on CLIP ViT-L/14; its "
            "first run downloads about 1.7 GB). Computed for the rated photos alone.",
        ),
    ] = None,
    fit: Annotated[
        FitChoice | None,
        typer.Option(
            help="Also measure Yours fitted this way against today's (centred: each photo "
            "against the mean of every photo at its place; offset: General's view as one more "
            "input, so Yours learns what General misses; measure it at more than one --l2; "
            "leaning: Yours leans on General where your ratings hold few photos like the one "
            "scored; by_scale: how much you like each scale, from aerial to macro, learned "
            "apart from what makes a photo strong within its scale; top_pairs: learned from "
            "pairs of a liked and a disliked photo a taste chose to show at one place, "
            "Compare Strongest's above all). Shown place by place "
            "and, when you rated in more than one collection, collection by collection.",
        ),
    ] = None,
    prompts: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Also measure General's aesthetic worded as this file says (the format of "
            "models/taxonomies/aesthetic.toml, with a strong label) against today's wording, "
            "on the rated photos alone; no search uses it.",
        ),
    ] = None,
    yours_embedder: Annotated[
        str | None,
        typer.Option(
            help="Also measure Yours learned on this embedding (such as clip_large: CLIP "
            "ViT-L/14) against today's Yours, paired, by collection too. Computed for the "
            "rated photos alone; no search uses it.",
        ),
    ] = None,
    general_embedder: Annotated[
        str | None,
        typer.Option(
            help="Also measure General with today's wording read by this embedder (one that "
            "reads text, such as clip_large: CLIP ViT-L/14), against today's. Computed for "
            "the rated photos alone; no search uses it.",
        ),
    ] = None,
    blend: Annotated[
        bool,
        typer.Option(
            "--blend",
            help="Measure the candidate for General (--aesthetic, --prompts or "
            "--general-embedder) averaged with today's General by rank within each place, "
            "half each, rather than alone.",
        ),
    ] = False,
    scores: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Also measure a file of one number per photo beside General and Yours: JSON "
            "Lines, each with the photo's content key as 'key' and its number as 'score' "
            "(--score-field names another field; higher is better). Only the photos it scores "
            "are judged, among the whole range and, apart, the photos a taste chose to show "
            "(thumbs on results too); every rating still teaches Yours. Measure it alone.",
        ),
    ] = None,
    score_field: Annotated[
        str,
        typer.Option(help="With --scores: the field that holds the number (default: score)."),
    ] = "score",
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    holdout: Annotated[
        bool,
        typer.Option(
            "--holdout",
            help="Report on the hold-out alone: a last check of a final candidate, never for "
            "tuning (calico holdout show).",
        ),
    ] = False,
) -> None:
    """Would the Yours template rank your ratings better than General? Each place you
    rated for your taste is judged by a taste learned from every other rating.

    AUC: how often a photo you liked outscores one you did not, by strength (what
    Strongest and Stories rank by) and by the aesthetic alone, each with its range when the
    places are drawn again. Yours is saved by calico train taste only when it is shown to
    beat General by strength: the range of Yours minus General above zero, at five places or
    more.
    """
    with Engine.open() as engine, _place_progress() as progress:
        sessions = Sessions(engine)
        reports = sessions.evaluate_taste(
            embedder or None,
            l2 or [TASTE_L2],
            only_holdout=holdout,
            quality_weights=quality_weight or None,
            technical=technical,
            aesthetic=aesthetic,
            fit=fit,
            progress=progress,
            prompts=prompts,
            blend=blend,
            general_embedder=general_embedder,
            yours_embedder=yours_embedder,
            scores=scores,
            score_field=score_field,
        )
        held = _held_out(sessions, holdout)
        tried = _record_tried(
            engine.settings,
            "taste",
            [
                f"{r.feature} l2={r.l2:g} quality_weight={r.quality_weight:g}"
                + (f" {r.technical}" if r.technical else "")
                + (f" aesthetic={r.aesthetic_candidate}" if r.aesthetic_candidate else "")
                + (" blend" if r.general_blend else "")
                + (f" fit={r.fit}" if r.fit else "")
                + (f" yours={r.yours_candidate}" if r.yours_candidate else "")
                + (f" scores={r.scores.name}" if r.scores else "")
                for r in reports
            ],
            holdout,
        )
    if as_json:
        out = {
            "reports": [_taste_json(r) for r in reports],
            "holdout": held,
            "tried": dataclasses.asdict(tried),
        }
        typer.echo(json.dumps(_nan_to_null(out), indent=2, allow_nan=False))
        return
    console = _out()
    console.print(held, markup=False, soft_wrap=True)
    _print_tried(console, "taste", tried, holdout)
    if not any(r.rows for r in reports):
        console.print(
            "No taste ratings yet that can be judged: rate at least two places with Rate for "
            "your taste (the Label page) or: calico eval new-taste FOLDER",
            markup=False,
        )
        return
    for report in reports:
        _print_taste_report(console, report)


def _print_taste_chosen(console: Console, report: TasteReport) -> None:
    """The photos a taste chose to show, read apart, and the file of scores beside them."""
    chosen = report.chosen
    if chosen is not None:
        console.print(
            "Among the photos a taste chose to show (drawn for their value, or shown in "
            f"Compare Strongest): General {_range(chosen.b)}, Yours {_range(chosen.a)}; Yours "
            f"minus General: {_difference(chosen.difference)}. Read apart from the AUCs above, "
            "which leave them out: a taste's ranking chose these photos, so a taste that chose "
            "some of them has less to rank them by than one that chose none, and this is not a "
            "test of which chooses better.",
            markup=False,
            soft_wrap=True,
        )
    scores = report.scores
    if scores is None:
        return
    console.print(
        f"{scores.name}, {_plural(scores.scored, 'photo')} judged. Yours plus {scores.name} is "
        "the mean of each one's z-scores at a place.",
        markup=False,
        soft_wrap=True,
    )
    for population, reading in (
        ("On the whole range", scores.whole),
        ("Among the photos a taste chose to show", scores.chosen),
    ):
        general, blend = reading.versus_general, reading.blend_versus_yours
        if not general.a.places:
            console.print(
                f"{population}: no location has both a like and a dislike among the photos the "
                "file scores.",
                markup=False,
                soft_wrap=True,
            )
            continue
        console.print(
            f"{population}: {scores.name} {_range(general.a)}, General {_range(general.b)}; "
            f"{scores.name} minus General: {_difference(general.difference)}. Yours plus "
            f"{scores.name} {_range(blend.a)}, Yours {_range(blend.b)}; minus Yours: "
            f"{_difference(blend.difference)}.",
            markup=False,
            soft_wrap=True,
        )
    table = Table(
        "location",
        "whole range: liked/not",
        "scores",
        "General",
        "Yours",
        "chosen: liked/not",
        "scores",
        "General",
        "Yours",
        title="AUC at each judged location, where it has both a like and a dislike",
        title_justify="left",
    )
    for row in scores.rows:
        cells = [row.location]
        for side in (row.whole, row.chosen):
            cells += [
                f"{side.liked}/{side.disliked}",
                _fmt(side.scores),
                _fmt(side.general),
                _fmt(side.yours),
            ]
        table.add_row(*cells)
    console.print(table)


def _print_taste_alternatives(console: Console, report: TasteReport) -> None:
    """Each other way measured beside today's, paired: no quality, another technical
    measure, another aesthetic for General, another fit of Yours."""
    general, yours = report.general_with_quality, report.yours_with_quality
    if general is not None and yours is not None and general.a.places:
        console.print(
            f"Strength minus the aesthetic alone, at quality weight {report.quality_weight:g}: "
            f"General {_difference(general.difference)}; Yours "
            f"{_difference(yours.difference)}. Above zero, mixing in quality helps.",
            markup=False,
            soft_wrap=True,
        )
    if report.candidate is not None:
        console.print(
            f"General's strength with {report.technical}: {_interval(report.candidate.a)}; "
            f"minus today's: {_difference(report.candidate.difference)}.",
            markup=False,
            soft_wrap=True,
        )
    if report.candidate_general is not None and report.candidate_general_alone is not None:
        console.print(
            f"General with {report.aesthetic_candidate}"
            + (" blended half and half with today's" if report.general_blend else "")
            + " as its aesthetic, by strength: "
            f"{_interval(report.candidate_general.a)}; minus today's: "
            f"{_difference(report.candidate_general.difference)}. By the aesthetic alone: "
            f"{_interval(report.candidate_general_alone.a)}; minus today's: "
            f"{_difference(report.candidate_general_alone.difference)}.",
            markup=False,
            soft_wrap=True,
        )
        for collection, measured in report.candidate_general_by_collection:
            console.print(
                f"  In {collection}: {_interval(measured.a)}; minus today's: "
                f"{_difference(measured.difference)}.",
                markup=False,
                soft_wrap=True,
            )
    if report.fitted is not None:
        console.print(
            f"Yours {_fit_name(report)}, by strength: "
            f"{_interval(report.fitted.a)}; minus today's (l2 {TASTE_L2:g}): "
            f"{_difference(report.fitted.difference)}.",
            markup=False,
            soft_wrap=True,
        )
    if report.top_candidate_general is not None:
        top = report.top_candidate_general
        console.print(
            f"Liked among the top quarter of each place's rated photos: General with "
            f"{report.aesthetic_candidate} {_interval(top.a)}, today's {_fmt(top.b.value)}; "
            f"minus today's: {_difference(top.difference)}.",
            markup=False,
            soft_wrap=True,
        )
    if report.top_fitted is not None:
        top = report.top_fitted
        console.print(
            f"Liked among the top quarter of each place's rated photos: Yours "
            f"{_fit_name(report)} {_interval(top.a)}, today's "
            f"{_fmt(top.b.value)}; minus today's: {_difference(top.difference)}.",
            markup=False,
            soft_wrap=True,
        )
    if report.offset_on_candidate:
        console.print(
            f"  The offset read General with {report.aesthetic_candidate}, the candidate.",
            markup=False,
            soft_wrap=True,
        )
    for scale, today, fitted in report.liked_rank_by_scale:
        console.print(
            f"  Photos you liked, {scale}: ranked {today:.2f} at their place by today's Yours, "
            f"{fitted:.2f} fitted by scale (0 the weakest, 1 the strongest).",
            markup=False,
            soft_wrap=True,
        )
    for collection, measured in report.fitted_by_collection:
        console.print(
            f"  In {collection}: {_interval(measured.a)}; minus today's: "
            f"{_difference(measured.difference)}.",
            markup=False,
            soft_wrap=True,
        )
    if report.uncentred:
        console.print(
            f"Not centred (no place name, so no mean): {', '.join(report.uncentred)}.",
            markup=False,
            soft_wrap=True,
        )
    if report.unread:
        candidates = " or ".join(c for c in (report.technical, report.aesthetic_candidate) if c)
        console.print(
            f"Left out of every measure here: {_plural(report.unread, 'rated photo')} "
            f"{candidates or 'the candidate'} could not read (moved, or on a drive not "
            "mounted).",
            markup=False,
            soft_wrap=True,
        )


def _fit_name(report: TasteReport) -> str:
    if report.fit == "embedding":
        return f"learned on {report.yours_candidate}"
    return _FIT_NAMES.get(report.fit, report.fit)


_FIT_NAMES = {
    "centred": "fitted centred on each place",
    "offset": "fitted with General as an offset",
    "leaning": "leaning on General where it has seen little",
    "by_scale": "fitted by scale",
    "top_pairs": "fitted on pairs among the first picks",
}


def _taste_json(report: TasteReport) -> dict[str, object]:
    return {
        "feature": report.feature,
        "l2": report.l2,
        "trained_on": report.trained_on,
        "places": [dataclasses.asdict(r) for r in report.rows],
        "one_sided": report.one_sided,
        "general_strength": report.general_strength,
        "yours_strength": report.yours_strength,
        "general_aesthetic": report.general_aesthetic,
        "yours_aesthetic": report.yours_aesthetic,
        "yours_wins": report.yours_wins,
        "notes": list(report.notes),
        "strength": None if report.strength is None else dataclasses.asdict(report.strength),
        "aesthetic": None if report.aesthetic is None else dataclasses.asdict(report.aesthetic),
        "macro": dict(report.macro),
        "first_at": list(report.first_at),
        "general_first": [_first_json(f) for f in report.general_first],
        "yours_first": [_first_json(f) for f in report.yours_first],
        "first": [dataclasses.asdict(f) for f in report.first],
        "quality_weight": report.quality_weight,
        "technical": report.technical,
        "candidate": None if report.candidate is None else dataclasses.asdict(report.candidate),
        "aesthetic_candidate": report.aesthetic_candidate,
        "candidate_general": _compared_json(report.candidate_general),
        "candidate_general_alone": _compared_json(report.candidate_general_alone),
        "with_quality": {
            "general": _compared_json(report.general_with_quality),
            "yours": _compared_json(report.yours_with_quality),
        },
        "unread": report.unread,
        "fit": report.fit,
        "fitted": _compared_json(report.fitted),
        "yours_candidate": report.yours_candidate,
        "candidate_general_by_collection": {
            name: dataclasses.asdict(measured)
            for name, measured in report.candidate_general_by_collection
        },
        "fitted_by_collection": {
            name: dataclasses.asdict(measured) for name, measured in report.fitted_by_collection
        },
        "uncentred": list(report.uncentred),
        "chosen": _compared_json(report.chosen),
        "scores": None if report.scores is None else dataclasses.asdict(report.scores),
        "general_blend": report.general_blend,
        "offset_on_candidate": report.offset_on_candidate,
        "top_candidate_general": _compared_json(report.top_candidate_general),
        "top_fitted": _compared_json(report.top_fitted),
        "liked_rank_by_scale": [
            {"scale": scale, "today": today, "fitted": fitted}
            for scale, today, fitted in report.liked_rank_by_scale
        ],
        "left_out": {
            "copies": report.copies_left_out,
            "same_place": report.same_place_left_out,
            "same_places": [list(g) for g in report.same_places],
        },
    }


def _compared_json(measured: Compared | None) -> dict[str, object] | None:
    return None if measured is None else dataclasses.asdict(measured)


def _first_json(first: FirstPicks) -> dict[str, object]:
    return {"rated": first.rated, "liked": first.liked, "share": first.share}


def _liked_of(first: FirstPicks, *, share: bool = False) -> str:
    """Liked of rated ("3/5"), so a share is never read without how few it rests on."""
    text = f"{first.liked}/{first.rated}"
    return f"{text} ({_fmt(first.share)})" if share and first.rated else text


def _print_taste_report(console: Console, report: TasteReport) -> None:
    table = Table(
        "place",
        "rated",
        "liked",
        "not",
        "strength: General",
        "strength: Yours",
        *((f"strength: {report.fit}",) if report.fit else ()),
        *(("General's share",) if report.fit == "leaning" else ()),
        "aesthetic: General",
        "aesthetic: Yours",
        *(f"first {k}: {name}" for k in report.first_at for name in ("General", "Yours")),
        title=f"Learned on {report.feature}, l2 {report.l2:g}, quality weight "
        f"{report.quality_weight:g} ({report.trained_on} ratings)",
    )
    for r in report.rows:
        table.add_row(
            Text(r.location),
            str(r.rated),
            str(r.liked),
            str(r.disliked),
            _fmt(r.general_strength),
            _fmt(r.yours_strength),
            *((_fmt(r.fitted_strength),) if report.fit else ()),
            *((_fmt(r.leaned),) if report.fit == "leaning" else ()),
            _fmt(r.general_aesthetic),
            _fmt(r.yours_aesthetic),
            *(
                _liked_of(first)
                for pair in zip(r.general_first, r.yours_first, strict=True)
                for first in pair
            ),
        )
    strength, aesthetic = report.strength, report.aesthetic
    table.add_row(
        Text("all places", style="bold"),
        str(sum(r.rated for r in report.rows)),
        str(sum(r.liked for r in report.rows)),
        str(sum(r.disliked for r in report.rows)),
        *(
            (_fmt(report.general_strength), _fmt(report.yours_strength))
            if strength is None
            else (_range(strength.b), _range(strength.a))
        ),
        *((_range(report.fitted.a),) if report.fitted is not None else ()),
        *(("",) if report.fit == "leaning" else ()),
        *(
            (_fmt(report.general_aesthetic), _fmt(report.yours_aesthetic))
            if aesthetic is None
            else (_range(aesthetic.b), _range(aesthetic.a))
        ),
        *(
            _liked_of(first, share=True)
            for pair in zip(report.general_first, report.yours_first, strict=True)
            for first in pair
        ),
    )
    if report.macro:
        table.add_row(
            Text("average of places"),
            "",
            "",
            "",
            *(
                _fmt(report.macro[name])
                for name in (
                    "general_strength",
                    "yours_strength",
                )
            ),
            *(("",) if report.fit else ()),
            *(("",) if report.fit == "leaning" else ()),
            *(_fmt(report.macro[name]) for name in ("general_aesthetic", "yours_aesthetic")),
            *("" for _ in report.first_at for _ in range(2)),
        )
    console.print(table)
    for note in report.notes:
        console.print(f"Note: {note}", markup=False, soft_wrap=True)
    if strength is not None and strength.a.places:
        console.print(
            f"[lo, hi]: where 95% of each value fell when the places, then the photos within "
            f"them, were drawn again ({_counts(strength.a)}). The average of places counts "
            "each place once.",
            markup=False,
            soft_wrap=True,
        )
        console.print(
            f"Yours minus General, by strength: {_difference(strength.difference)}; by the "
            f"aesthetic: {_difference(aesthetic.difference) if aesthetic else 'n/a'}.",
            markup=False,
            soft_wrap=True,
        )
    _print_taste_alternatives(console, report)
    _print_taste_chosen(console, report)
    for at, measured in zip(report.first_at, report.first, strict=True):
        console.print(
            f"first {at}, Yours minus General: {_difference(measured.difference)}.",
            markup=False,
            soft_wrap=True,
        )
    if report.first_at:
        console.print(
            "first N: of the photos you rated among the first N each template would show at "
            "a place, how many you liked. Sessions before 31 October 2026 sampled the whole "
            "range, so few of their photos fall there: read a share with the count it rests on.",
            markup=False,
        )
    for_value = sum(r.for_value for r in report.rows)
    if for_value:
        console.print(
            "Chosen by a taste's ranking (drawn for their value, or shown in Compare "
            f"Strongest): {_plural(for_value, 'rated photo')}. They count among the first "
            "picks and are learned from; the AUCs over the whole range and the top quarter "
            "leave them out, since judging a candidate on them favours the taste that chose "
            "them" + (", and are read apart above." if report.chosen is not None else "."),
            markup=False,
        )
    for place in report.one_sided:
        console.print(
            f"{place}: only likes or only dislikes, so it could not be judged.", markup=False
        )
    if report.counted:
        console.print(f"Counted: {report.counted}.", markup=False, soft_wrap=True)
    console.print(_taste_leaks(report), markup=False, soft_wrap=True)
    if report.unsure or report.disagree:
        console.print(
            f"Left out as not sure: {_plural(report.unsure, 'photo')}. Liked in one use case "
            f"and not in another: {_plural(report.disagree, 'photo')} (the newest answer "
            "decided).",
            markup=False,
            soft_wrap=True,
        )
    console.print(_taste_verdict(report), markup=False, soft_wrap=True)


def _taste_leaks(report: TasteReport) -> str:
    """What was kept out of training so a judged place's answers could not leak in."""
    names = "; ".join(" = ".join(group) for group in report.same_places)
    grouped = f" ({names}; calico locations)" if names else ""
    return (
        "Never learned from where they could give the answer away: "
        f"{_plural(report.copies_left_out, 'copy', 'copies')} of a judged photo (one frame saved "
        f"twice), {_plural(report.same_place_left_out, 'photo')} at a place probably the "
        f"same as the judged one{grouped}."
    )


def _taste_verdict(report: TasteReport) -> str:
    """Beats only when the paired difference by strength is clear at enough places."""
    if report.yours_wins:
        return "Yours beats General by strength at places it never learned from."
    judged = len(report.rows)
    if judged < WIN_PLACES:
        return (
            f"Yours is not shown to beat General: {_plural(judged, 'place')} judged, at least "
            f"{WIN_PLACES} needed. Rate more places with Rate for your taste."
        )
    return "Yours is not shown to beat General: the difference by strength could be zero or below."


def _range(measured: Interval) -> str:
    """A value and its range: "0.71 [0.65, 0.77]"."""
    if math.isnan(measured.lo):
        return _fmt(measured.value)
    return f"{_fmt(measured.value)} [{_fmt(measured.lo)}, {_fmt(measured.hi)}]"


_UNITS = {"search": "searches", "run or session": "runs or sessions"}


def _counts(measured: Interval) -> str:
    """What a value rests on: "12 places, 480 rated", and whether that is too few to trust
    a range (``stats.MIN_UNITS``)."""
    unit = _plural(measured.places, measured.unit, _UNITS.get(measured.unit))
    few = f"; too few {_UNITS.get(measured.unit, measured.unit + 's')} to trust a range"
    return f"{unit}, {measured.rated} rated{few if measured.few and measured.places else ''}"


def _interval(measured: Interval) -> str:
    """A headline: "0.71 [0.65, 0.77] (12 places, 480 rated)"."""
    return f"{_range(measured)} ({_counts(measured)})"


def _signed_range(measured: Interval) -> str:
    """A difference in a table cell: signed, with "?" when its range holds zero."""
    if math.isnan(measured.value):
        return "n/a"
    sign = "+" if measured.value >= 0 else ""
    return f"{sign}{_range(measured)}{'' if measured.excludes_zero else ' ?'}"


def _difference(measured: Interval) -> str:
    """A paired difference, signed, and whether its range leaves out zero."""
    if math.isnan(measured.value):
        return "n/a"
    sign = "+" if measured.value >= 0 else ""
    shown = f"{sign}{_interval(measured)}"
    return shown if measured.excludes_zero else f"{shown}, could be zero"


@eval_app.command("stories")
def eval_stories(
    too_alike: Annotated[
        list[float] | None,
        typer.Option(min=0, max=1, help="Where variety starts to fall (CLIP cosine); repeatable."),
    ] = None,
    variety: Annotated[
        list[float] | None,
        typer.Option(min=0, max=1, help="Variety's share of the connection side; repeatable."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    holdout: Annotated[
        bool,
        typer.Option(
            "--holdout",
            help="Report on the hold-out alone: a last check of a final candidate, never for "
            "tuning (calico holdout show).",
        ),
    ] = False,
) -> None:
    """Would variety rank the stories you liked above the ones you rejected? Every story
    rated in a session (calico eval new-stories), or with a 👍/👎 on a result, is
    re-scored from the evidence saved with it.

    The two kinds of answer are reported apart, never pooled. A session's story is liked
    when it works, rejected when it doesn't, and left out when you were not sure. On a
    result it is liked with no 👎, rejected with two or more (around an anchor, both
    companions), mixed with one. AUC: how often a liked story
    outscores a rejected one from the same run or session. Sessions also tell stories
    without variety, so the report says how each way's stories were rated.
    """
    variants = list(
        itertools.product(too_alike or STORY_TOO_ALIKE, variety or STORY_VARIETY_WEIGHTS)
    )
    with Engine.open() as engine:
        sessions = Sessions(engine)
        reports = sessions.evaluate_stories(variants, only_holdout=holdout)
        held = _held_out(sessions, holdout)
        tried = _record_tried(
            engine.settings,
            "stories",
            [f"too_alike={t:g} variety_weight={w:g}" for t, w in variants],
            holdout,
        )
    if as_json:
        out = {**dataclasses.asdict(reports), "holdout": held, "tried": dataclasses.asdict(tried)}
        typer.echo(json.dumps(_nan_to_null(out), indent=2, allow_nan=False))
        return
    console = _out()
    console.print(held, markup=False, soft_wrap=True)
    _print_tried(console, "stories", tried, holdout)
    shown = False
    for heading, report in (
        ("Stories rated whole in sessions (works / not sure / doesn't)", reports.sessions),
        ("Stories with a 👍/👎 on their photos, on story results", reports.results),
    ):
        if report.liked or report.rejected or report.mixed or report.unsure:
            console.print(heading, style="bold", markup=False)
            _print_story_report(console, report)
            shown = True
    if not shown:
        console.print(
            "No rated stories yet: rate stories with Rate stories (the Label page) or: "
            "calico eval new-stories FOLDER",
            markup=False,
        )


def _print_story_report(console: Console, report: StoryReport) -> None:
    table = Table(
        "variety from",
        "weight",
        "liked over rejected (AUC)",
        "minus no variety",
        title=(
            f"{report.liked} liked, {report.rejected} rejected, {report.mixed} mixed, "
            f"{report.unsure} not sure stories from "
            f"{_plural(report.runs, 'run or session', 'runs or sessions')}"
        ),
    )
    for row in report.rows:
        table.add_row(
            "no variety" if row.too_alike is None else f"{row.too_alike:g}",
            f"{row.weight:g}",
            _fmt(row.auc) if row.interval is None else _range(row.interval),
            "" if row.versus is None else _signed_range(row.versus),
        )
    console.print(table)
    first = report.rows[0].interval if report.rows else None
    if first is not None and first.places:
        console.print(
            f"[lo, hi]: where 95% of each value fell when the runs or sessions, then the "
            f"stories within them, were drawn again ({_counts(first)}). A difference marked "
            "? could be zero.",
            markup=False,
            soft_wrap=True,
        )
    if report.arms:
        told = Table(
            "told",
            "works",
            "not sure",
            "doesn't",
            "works of works and doesn't",
            "minus today's way with variety",
            title="Stories rated in sessions",
        )
        for arm in report.arms:
            told.add_row(
                f"{arm.template}{'' if arm.variety else ' without variety'}"
                f"{'' if arm.way == 'today' else ' ' + arm.way}",
                str(arm.works),
                str(arm.unsure),
                str(arm.doesnt),
                "" if arm.works_share is None else _range(arm.works_share),
                "" if arm.versus_today is None else _signed_range(arm.versus_today),
            )
        console.print(told)
    if report.doesnt_fit:
        console.print(
            f"Of {report.doesnt_fit} photos that alone did not fit their story, "
            f"{report.odd_one_out} were the least like the other two.",
            markup=False,
        )
    if report.missing:
        console.print(
            f"{report.missing} rated stories were left out: a photo's embedding is gone.",
            markup=False,
        )


@eval_app.command("finished-stories")
def eval_finished_stories(
    path: Annotated[
        Path,
        typer.Argument(
            help='JSON file: {"stories": {"A": {"paths": [left, middle, right]}}}, the photos '
            "as catalogued."
        ),
    ],
    template: Annotated[
        list[str] | None,
        typer.Option(help="Whose taste ranks photos; repeatable (default: general and yours)."),
    ] = None,
    variant: Annotated[
        list[str] | None,
        typer.Option(
            help="Story options to try beside the defaults, as KEY=VALUE pairs separated by "
            "';', e.g. 'variety_weight=0; candidates=300'; repeatable."
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Would Three-photo stories tell the stories you finished yourself?

    Each story is scored as the task scores one, against every story made by swapping one
    of its photos for one of the task's candidates at the same places. Over swaps: how
    often yours scores higher. Shortlisted: how many of its photos are candidates on their
    own strength; the task never uses one that is not. Each template and set of options
    after the first is compared with the first, story by story.
    """
    variants: list[dict[str, Any]] = [
        {},
        *(dict(parse_option(p) for p in raw.split(";") if p.strip()) for raw in variant or []),
    ]
    with Engine.open() as engine:
        reports = Sessions(engine).evaluate_finished_stories(
            path, templates=template or ["general", "yours"], variants=variants
        )
        tried = _record_tried(
            engine.settings,
            "finished-stories",
            [f"{r.template}, {r.variant}" for r in reports],
            holdout=False,
        )
    if as_json:
        out = {"reports": [_finished_json(r) for r in reports], "tried": dataclasses.asdict(tried)}
        typer.echo(json.dumps(_nan_to_null(out), indent=2, allow_nan=False))
        return
    console = _out()
    for report in reports:
        _print_finished_report(console, report)
    summary = Table(
        "template",
        "story options",
        "over swaps",
        "minus the first",
        "photos shortlisted",
        "stories it could tell",
        "anchors tried",
        title="All finished stories",
    )
    for r in reports:
        summary.add_row(
            r.template,
            Text(r.variant),
            _fmt(r.mean_win) if r.win_interval is None else _range(r.win_interval),
            "" if r.versus is None else _signed_range(r.versus),
            _pct(r.shortlisted),
            _pct(r.all_three_shortlisted),
            _pct(r.anchors_tried),
        )
    console.print(summary)
    _print_tried(console, "finished-stories", tried, holdout=False)
    first = reports[0].win_interval if reports else None
    if first is not None and first.places:
        console.print(
            f"[lo, hi]: where 95% of each value fell when the stories were drawn again "
            f"({_plural(first.places, 'story', 'stories')}). A difference marked ? could be "
            "zero.",
            markup=False,
            soft_wrap=True,
        )
    if reports and reports[0].missing:
        console.print(
            f"Left out, a photo is not catalogued: {', '.join(reports[0].missing)}", markup=False
        )


def _finished_json(report: FinishedReport) -> dict[str, object]:
    return {
        "template": report.template,
        "variant": report.variant,
        "mean_win": report.mean_win,
        "win_interval": None
        if report.win_interval is None
        else dataclasses.asdict(report.win_interval),
        "versus": None if report.versus is None else dataclasses.asdict(report.versus),
        "shortlisted": report.shortlisted,
        "all_three_shortlisted": report.all_three_shortlisted,
        "anchors_tried": report.anchors_tried,
        "missing": report.missing,
        "stories": [
            {
                "name": s.name,
                "score": s.score,
                "win": s.win,
                "wins": list(s.wins),
                "shortlisted": list(s.shortlisted),
                "anchor_tried": s.anchor_tried,
                "swaps": [len(side) for side in s.swaps],
            }
            for s in report.stories
        ],
    }


def _print_finished_report(console: Console, report: FinishedReport) -> None:
    table = Table(
        "story",
        "shortlisted",
        "anchor tried",
        "over swaps: left",
        "middle",
        "right",
        "over swaps",
        title=f"{report.template}, {report.variant}",
    )
    for s in report.stories:
        table.add_row(
            Text(s.name),
            "".join("✓" if f else "·" for f in s.shortlisted),
            "✓" if s.anchor_tried else "·",
            *(_fmt(w) for w in s.wins),
            _fmt(s.win),
        )
    console.print(table)


@eval_app.command("learned-stories")
def eval_learned_stories(  # noqa: PLR0917
    finished: Annotated[
        Path | None,
        typer.Argument(
            help="Stories you finished, learned from as stories that work: the file calico "
            "eval finished-stories reads."
        ),
    ] = None,
    template: Annotated[
        str, typer.Option(help="Whose taste gives the stories' strength: general or yours.")
    ] = "general",
    l2: Annotated[
        float, typer.Option(min=0, help="How strongly the learned weights are held back.")
    ] = STORY_L2,
    swaps: Annotated[
        bool,
        typer.Option(
            "--swaps",
            help="Also learn from your finished stories with one photo swapped, as stories "
            "that don't work (needs FINISHED).",
        ),
    ] = False,
    variant: Annotated[
        list[str] | None,
        typer.Option(
            help="Story options to score today's way with beside it, as KEY=VALUE pairs "
            "separated by ';', e.g. 'distinct_weight=0.3'; repeatable."
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Would a scorer learned from your story answers tell stories that work better than
    today's way?

    Every story you rated works or doesn't in a session gets every signal the story task
    reads: strength, each connection, variety and each role. A regularised logistic learns
    their weights from the answers (and your finished stories, as stories that work). Each
    session is judged by a scorer that learned nothing at its places. AUC: how often a
    story that works outscores one that doesn't in the same session. Learned minus today
    is paired by session. The hold-out's story sessions are left out. With --swaps, a
    third scorer also learns from swapped stories; they are never judged. Each --variant
    re-scores the rated stories today's way with those options, paired against today.
    """
    variants: list[dict[str, Any]] = [
        dict(parse_option(p) for p in raw.split(";") if p.strip()) for raw in variant or []
    ]
    with Engine.open() as engine:
        report = Sessions(engine).learn_stories(
            finished, template=template, l2=l2, swaps=swaps, variants=variants
        )
        tried = _record_tried(
            engine.settings,
            "learned-stories",
            [f"{template}, l2={l2:g}{', swaps' if swaps else ''}", *report.variants],
            holdout=False,
        )
    if as_json:
        out = {**dataclasses.asdict(report), "tried": dataclasses.asdict(tried)}
        typer.echo(json.dumps(_nan_to_null(out), indent=2, allow_nan=False))
        return
    console = _out()
    if not report.sessions:
        console.print(
            "No stories rated works or doesn't yet: rate stories with Rate stories (the Label "
            "page) or: calico eval new-stories FOLDER",
            markup=False,
        )
        return
    _print_tried(console, "learned-stories", tried, holdout=False)
    _print_learned_stories(console, report, template)
    if swaps and not report.swaps:
        console.print(
            "No swapped stories could be made: no other candidate at the finished stories' "
            "places could take one of their photos' places.",
            markup=False,
            soft_wrap=True,
        )


def _print_learned_stories(console: Console, report: LearnedStories, template: str) -> None:
    learned = (LEARNED, LEARNED_ALONE, *((LEARNED_SWAPS,) if report.swaps else ()))
    ways = (TODAY, *report.variants, *learned)
    table = Table(
        "session",
        "works",
        "doesn't",
        *ways,
        title=f"Stories learned from your answers ({template}, l2 {report.l2:g})",
    )
    for s in report.sessions:
        table.add_row(
            Text(s.name),
            str(s.works),
            str(s.doesnt),
            *(_fmt(s.auc.get(way, math.nan)) for way in ways),
        )
    table.add_row(
        "all sessions",
        str(sum(s.works for s in report.sessions)),
        str(sum(s.doesnt for s in report.sessions)),
        *(_range(report.auc[way]) if way in report.auc else "n/a" for way in ways),
    )
    console.print(table)
    for way in report.variants:
        if way in report.versus:
            console.print(
                f"{way} minus today: {_difference(report.versus[way])}.",
                markup=False,
                soft_wrap=True,
            )
    for way in learned:
        console.print(
            f"{way} minus today: {_difference(report.versus[way])}."
            if way in report.versus
            else f"{way}: not measured. No session had stories that work and stories that "
            "don't at other places to learn from.",
            markup=False,
            soft_wrap=True,
        )
    weights = Table("signal", "weight", title="What it learned, from every story")
    for name, weight in sorted(report.weights.items(), key=lambda kv: -abs(kv[1])):
        weights.add_row(name, f"{weight:+.2f}")
    console.print(weights)
    console.print(
        "weight: how much a story's chance of working rises (+) or falls (-) per spread of "
        "the signal across your stories. Finished stories, learned from as stories that work "
        f"and never judged: {report.finished}. "
        + (
            f"Swapped stories, learned from as stories that don't work: {report.swaps}. "
            if report.swaps
            else ""
        )
        + f"Left out as not sure: {report.unsure}. Could "
        f"not be measured (a photo gone or without features): {report.missing}.",
        markup=False,
        soft_wrap=True,
    )
    if report.left_out:
        console.print(
            f"Not measured for every story, so not learned from: {', '.join(report.left_out)}.",
            markup=False,
            soft_wrap=True,
        )
    for note in report.notes:
        console.print(note, markup=False, soft_wrap=True)
    for way in (LEARNED, *((LEARNED_SWAPS,) if report.swaps else ())):
        better = report.versus.get(way)
        shown = better is not None and better.excludes_zero and better.value > 0
        scorer = "The learned scorer" if way == LEARNED else "Learned with swaps, the scorer"
        console.print(
            f"{scorer} is shown to beat today's way."
            if shown
            else f"{scorer} is not shown to beat today's way: the difference could be zero "
            "or below.",
            markup=False,
            soft_wrap=True,
        )


def _record_tried(settings: Settings, command: str, tried: Sequence[str], holdout: bool) -> Tried:
    """Note what this eval measured (``calico.experiments``), and how much has been tried;
    looks at the hold-out count from the day it was drawn."""
    held = read_holdout(settings.holdout_path) if holdout else None
    since = None if held is None else held.drawn_at
    return record(
        settings.experiments_path,
        command,
        tried,
        holdout=holdout,
        at=datetime.now(UTC),
        since=since,
    )


def _print_tried(console: Console, command: str, tried: Tried, holdout: bool) -> None:
    if holdout:
        console.print(
            f"Looks at the hold-out with calico eval {command} so far: {tried.holdout_looks}.",
            markup=False,
            soft_wrap=True,
        )
    elif tried.settings > 1:
        console.print(
            f"Settings tried with calico eval {command} so far: {tried.settings} "
            f"(experiments.jsonl). Read the best as a best of {tried.settings}: it looks "
            "better than it is, and the hold-out says by how much.",
            markup=False,
            soft_wrap=True,
        )


def _held_out(sessions: Sessions, only: bool) -> str:
    """What a report leaves out, or that it is on the hold-out alone."""
    held = sessions.holdout()
    if held is None:
        return "Nothing is held out yet: draw a hold-out once with calico holdout draw."
    if only:
        return (
            "On the hold-out alone: a last check of a final candidate. A look you change a "
            "setting after uses some of it up."
        )
    return f"Left out: the hold-out ({_holdout_counts(held)}; calico holdout show)."


def _holdout_counts(held: HoldOut) -> str:
    return (
        f"{_plural(len(held.places), 'place')}, "
        f"{_plural(len(held.references), 'same-view reference')}, "
        f"{_plural(len(held.story_sessions), 'story session')}"
    )


@holdout_app.command("draw")
def holdout_draw(
    leave_out: Annotated[
        list[str] | None,
        typer.Option(
            "--leave-out",
            help="A place never drawn, nor the places next to it (files in common): one you "
            "designed against. Repeat it for each.",
        ),
    ] = None,
    places: Annotated[
        int | None,
        typer.Option(
            "--places",
            min=1,
            help="How many places to hold out (default: 4). Each one is left out of what your "
            "taste learns, with the places next to it.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be held out, and write nothing: the draw is once.",
        ),
    ] = False,
) -> None:
    """Set aside, once, some of what you rated: places judged for your taste, same-view
    references and story sessions, drawn at random. Every calico eval then leaves them out,
    and --holdout reports on them alone, for a last check no tuning has seen. Your taste is
    not learned from them either, nor from a photo also filed under them or from a place
    that shares a tenth of its files with them."""
    with Engine.open() as engine:
        drawn = Sessions(engine).draw_holdout(
            leave_out=tuple(leave_out or ()), places=places, write=not dry_run
        )
    console = _out()
    if dry_run:
        console.print(
            f"Would hold out {_holdout_counts(drawn)}; nothing was written. Draw it once by "
            "running the same command without --dry-run:",
            markup=False,
            soft_wrap=True,
        )
    else:
        console.print(f"Held out {_holdout_counts(drawn)}:", markup=False)
    _print_holdout(console, drawn)


@holdout_app.command("show")
def holdout_show(
    as_json: Annotated[bool, typer.Option("--json", help="Print it as JSON.")] = False,
) -> None:
    """What is held out, and when it was drawn."""
    held = read_holdout(load_settings().holdout_path)
    if as_json:
        out = None if held is None else {**dataclasses.asdict(held), "drawn_at": str(held.drawn_at)}
        typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
        return
    console = _out()
    if held is None:
        console.print("Nothing is held out yet. Draw it once with: calico holdout draw")
        return
    console.print(
        f"Drawn {held.drawn_at:%Y-%m-%d %H:%M} (seed {held.seed}): {_holdout_counts(held)}.",
        markup=False,
    )
    _print_holdout(console, held)


def _print_holdout(console: Console, held: HoldOut) -> None:
    for heading, names in (
        ("Places", held.places),
        ("Same-view references", held.references),
        ("Story sessions", held.story_sessions),
    ):
        if names:
            console.print(heading, style="bold", markup=False)
            for name in names:
                console.print(f"  {name}", markup=False)


@controls_app.command("draw")
def controls_draw() -> None:
    """Draw the pool, once: photos you rated in Compare Strongest sessions (two thirds liked),
    outside what the hold-out reaches. Each Compare session made after it asks some of them,
    mixed in without saying so; calico eval retest reads what you answered."""
    with Engine.open() as engine:
        drawn = Sessions(engine).draw_controls()
    liked = sum(c.liked for c in drawn.photos)
    _out().print(
        f"Drew {_plural(len(drawn.photos), 'control photo')} ({liked} liked, "
        f"{len(drawn.photos) - liked} not liked when you rated them). Each Compare Strongest "
        f"session made from now on asks {CONTROLS_PER_SESSION} of them, among its picks, without "
        "saying so; calico eval retest reads the answers.",
        markup=False,
        soft_wrap=True,
    )


@controls_app.command("show")
def controls_show(
    as_json: Annotated[bool, typer.Option("--json", help="Print it as JSON.")] = False,
) -> None:
    """The pool, and when it was drawn."""
    pool = read_controls(load_settings().controls_path)
    if as_json:
        out = (
            None
            if pool is None
            else {
                "seed": pool.seed,
                "drawn_at": str(pool.drawn_at),
                "photos": [dataclasses.asdict(p) for p in pool.photos],
            }
        )
        typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
        return
    console = _out()
    if pool is None:
        console.print("Nothing is drawn yet. Draw the pool once with: calico controls draw")
        return
    liked = sum(c.liked for c in pool.photos)
    console.print(
        f"Drawn {pool.drawn_at:%Y-%m-%d %H:%M} (seed {pool.seed}): "
        f"{_plural(len(pool.photos), 'photo')}, {liked} liked and "
        f"{len(pool.photos) - liked} not liked when you rated them.",
        markup=False,
    )


@train_app.command("taste")
def train_taste(
    embedder: Annotated[
        str | None, typer.Option(help="The embedding to learn on (default: the main one).")
    ] = None,
    l2: Annotated[
        float, typer.Option(min=0, help="How strongly the weights are held back.")
    ] = TASTE_L2,
    force: Annotated[
        bool, typer.Option("--force", help="Save it even if it is not shown to beat General.")
    ] = False,
    accept_missing: Annotated[
        bool,
        typer.Option(
            "--accept-missing", help="Learn even though some labelling sets cannot be read."
        ),
    ] = False,
) -> None:
    """Learn your taste from your ratings, for the Yours template. It is measured first
    (as calico eval taste does) and saved only if it is shown to beat General at places it
    never learned from: by strength, clear of chance, at five places or more."""
    with Engine.open() as engine:
        trained = Sessions(engine).train_taste(
            embedder, l2, force=force, accept_missing=accept_missing
        )
    console = _out()
    _print_taste_report(console, trained.report)
    if trained.saved:
        console.print(
            f"Saved your taste ({trained.report.trained_on} ratings). Choose Yours on a search, "
            'or set template = "yours" in calico.toml.',
            markup=False,
        )
        console.print(_calibration_line(trained), markup=False, soft_wrap=True)
    else:
        console.print(
            "Not saved: it is not shown to beat General. Rate more places with Rate for your "
            "taste and try again, or save it anyway with --force.",
            markup=False,
        )


def _calibration_line(trained: TasteTraining) -> str:
    """What a probability from the taste just saved means, or why it shows none."""
    made = trained.calibration
    if made is None:
        shown = [f for f in trained.report.forecasts if f.first_pick]
        first, places = len(shown), len({f.place for f in shown})
        return (
            "Yours shows no probability per pick yet: it is read from what you said about the "
            "photos Strongest showed first at places Yours had not learned (Compare Strongest, "
            f"taste sessions), and needs {calibration_module.MIN_FORECASTS} of them at "
            f"{calibration_module.MIN_PLACES} places or more, and a probability that clearly "
            f"rises with Yours's (it had {first} at {places})."
        )
    return (
        f"Picks show how likely you are to like them, read from {made.photos} photos Strongest "
        f"showed first at {made.places} places Yours had not learned: one it gives 90% is one "
        f"you liked {float(made.probability(0.9)):.0%} of the time."
    )


@eval_app.command("list")
def eval_list() -> None:
    """Labelling sets, newest first, and how far each is labelled."""
    settings = load_settings()
    sets = LabelStore(settings.labels_dir).all()
    console = _out()
    if not sets:
        console.print("No labelling sets yet. Create one with: calico eval new TEXT FOLDER")
        return
    current = open_journal(settings).ratings()
    table = Table(Column("set", no_wrap=True), "created", "search", "labelled")
    for s in sets:
        table.add_row(
            s.set_id,
            s.created_at.strftime("%Y-%m-%d %H:%M"),
            Text(s.text),
            f"{answered(s, current)} of {s.total}",
        )
    console.print(table)


@eval_app.command("criteria")
def eval_criteria(
    scorer: Annotated[
        list[str] | None,
        typer.Option(help="Scorer to measure; repeatable (default: all of them)."),
    ] = None,
    embedder: Annotated[
        list[str] | None,
        typer.Option(
            help="Model that reads the text; repeatable (default: the “Describe it” "
            "embedder, then every other one that reads text)."
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """How well each criteria scorer ranks your labelled searches.

    precision@10: credit of the first ten (a match 1, partly 0.5). nDCG@10: the
    same, rewarding the best matches first. AUC: how often a match outscores a
    non-match anywhere in the list. Clear matches: of the photos the model called
    a clear match, how many you labelled Matches, and how many No.
    """
    with Engine.open() as engine:
        rows = Sessions(engine).evaluate_criteria(scorer, embedder)
        keys = list(dict.fromkeys(r.scorer for r in rows))
        tried = _record_tried(engine.settings, "criteria", keys, holdout=False)
    means = {
        key: {m: _mean([getattr(r.score, m) for r in rows if r.scorer == key]) for m in METRICS}
        for key in keys
    }
    pooled = {key: _pooled([r.clear for r in rows if r.scorer == key]) for key in keys}
    measured = criteria_intervals(rows)
    if as_json:
        report = {
            "searches": [
                {
                    "set_id": r.set_id,
                    "text": r.text,
                    "scorer": r.scorer,
                    "missing": r.missing,
                    **{k: v for k, v in dataclasses.asdict(r.score).items() if k != "threshold"},
                    "threshold": dataclasses.asdict(r.score.threshold),
                    "clear": _clear_json(r.clear),
                }
                for r in rows
            ],
            "mean": {
                key: {
                    **means[key],
                    "clear": _clear_json(pooled[key]),
                    "auc_interval": dataclasses.asdict(measured.auc[key]),
                    "versus": None
                    if key not in measured.versus
                    else dataclasses.asdict(measured.versus[key]),
                }
                for key in keys
            },
            "baseline": measured.baseline,
            "tried": dataclasses.asdict(tried),
        }
        typer.echo(json.dumps(_nan_to_null(report), indent=2, ensure_ascii=False, allow_nan=False))
        return
    console = _out()
    if not rows:
        console.print(
            "No labelled searches yet. Label some in the web interface (Label), "
            "or start one with: calico eval new TEXT FOLDER",
            markup=False,
        )
        return
    table = Table(
        "search",
        Column("scorer", no_wrap=True),
        "labelled",
        "matches",
        "p@10",
        "nDCG@10",
        "AUC",
        "clear matches",
    )
    for r in rows:
        table.add_row(
            Text(r.text),
            r.scorer,
            str(r.score.labelled) + (f" ({r.missing} gone)" if r.missing else ""),
            str(r.score.matches),
            *(_fmt(getattr(r.score, m)) for m in METRICS),
            _fmt_clear(r.clear),
        )
    for key, values in means.items():
        table.add_row(
            Text("mean", style="bold"),
            key,
            "",
            "",
            *(_range(measured.auc[key]) if m == "auc" else _fmt(values[m]) for m in METRICS),
            _fmt_clear(pooled[key]),
        )
    console.print(table)
    _print_search_note(console, measured.auc.values(), "AUC")
    _print_tried(console, "criteria", tried, holdout=False)
    for key, versus in measured.versus.items():
        console.print(
            f"{key} minus {measured.baseline}, AUC: {_difference(versus)}.",
            markup=False,
            soft_wrap=True,
        )


def _print_search_note(console: Console, measured: Iterable[Interval], what: str) -> None:
    """What a range over searches rests on."""
    first = next(iter(measured), None)
    if first is not None and first.places:
        console.print(
            f"[lo, hi] on the mean {what}: where 95% of it fell when the {first.unit}s, then "
            f"the photos labelled in each, were drawn again ({_counts(first)}).",
            markup=False,
            soft_wrap=True,
        )


@eval_app.command("same-view")
def eval_same_view(  # noqa: PLR0917
    embedder: Annotated[
        list[str] | None,
        typer.Option(
            help='What gives "same subject" (default: the main embedding); repeatable. An '
            "extractor Calico does not compute for every photo is run on the labelled photos "
            "only."
        ),
    ] = None,
    layout_weight: Annotated[
        list[float] | None,
        typer.Option(
            min=0,
            max=1,
            help='How much "same camera angle" counts (default: the task\'s); repeatable.',
        ),
    ] = None,
    option: Annotated[
        list[str] | None,
        typer.Option(
            "--option",
            "-o",
            help="An option for a candidate --embedder, NAME.KEY=VALUE (e.g. dinov2.pool=avg); "
            "repeatable. Not for extractors Calico computes for every photo.",
        ),
    ] = None,
    patches: Annotated[
        bool,
        typer.Option(
            "--patches",
            help="Also measure matching part by part: each photo's share of patches with the "
            "reference (the view embedder's), judged from the sure cut.",
        ),
    ] = False,
    geometry: Annotated[
        bool,
        typer.Option(
            "--geometry",
            help="Measure matching part by part with only the matches in place (one placement "
            "of the reference) against today's share: over every labelled photo, and where "
            "the share is least sure.",
        ),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    holdout: Annotated[
        bool,
        typer.Option(
            "--holdout",
            help="Report on the hold-out alone: a last check of a final candidate, never for "
            "tuning (calico holdout show).",
        ),
    ] = False,
) -> None:
    """Where should same view's min_similarity be, and which match finds same views best?
    Your labelled same-view searches, each photo's match to the reference against whether
    you called it the same view.

    AUC: how often a same view outscores a different one. at default: precision (of what
    the default keeps, how much is the same view) and recall (of the same views, how many
    it keeps). best: the cut with the best balance over all searches together, since one
    min_similarity serves every view. held out: each search at the cut chosen on the
    others, so a threshold is never judged on the labels that chose it. A frame saved twice
    counts once; frames of a burst each count.
    """
    if geometry:
        if embedder or layout_weight or option:
            raise InvalidRequestError(
                "--geometry compares the view embedder's patches, so --embedder, "
                "--layout-weight and --option do not apply to it. Run it on its own."
            )
        _eval_view_geometry(holdout=holdout, as_json=as_json)
        return
    embedders: list[str | None] = [*embedder] if embedder else [None]
    weights: list[float | None] = [*layout_weight] if layout_weight else [None]
    combos = [(e, w) for e in embedders for w in weights]
    settings = _with_candidate_options(load_settings(), option or ())
    with Engine.open(settings) as engine:
        sessions = Sessions(engine)
        reports = (
            []
            if patches and not (embedder or layout_weight)
            else [sessions.evaluate_same_view(e, w, only_holdout=holdout) for e, w in combos]
        )
        if patches:
            reports.append(sessions.evaluate_same_view(patches=True, only_holdout=holdout))
        held = _held_out(sessions, holdout)
        tried = _record_tried(
            engine.settings,
            "same-view",
            [f"{r.embedder} layout_weight={r.layout_weight:g}" for r in reports],
            holdout,
        )
    if as_json:
        out = {
            "reports": [_view_report_json(r) for r in view_differences(reports)],
            "holdout": held,
            "tried": dataclasses.asdict(tried),
        }
        typer.echo(json.dumps(_nan_to_null(out), indent=2, ensure_ascii=False, allow_nan=False))
        return
    console = _out()
    console.print(held, markup=False, soft_wrap=True)
    _print_tried(console, "same-view", tried, holdout)
    for set_id in reports[0].skipped:
        console.print(f"Skipped {set_id}: its reference photo is gone.", markup=False)
    if copies := reports[0].copies:
        console.print(
            f"Counted {copies} {'copy' if copies == 1 else 'copies'} of a frame saved twice "
            "(a re-export, a HEIC also kept as JPEG) once.",
            markup=False,
        )
    if not reports[0].rows:
        console.print(
            "No labelled same-view searches yet. Start one with: "
            "calico eval new-view REFERENCE FOLDER",
            markup=False,
        )
        return
    if len(reports) == 1:
        _print_view_report(console, reports[0])
    else:
        _print_view_comparison(console, view_differences(reports))


def _eval_view_geometry(*, holdout: bool, as_json: bool) -> None:
    """``eval same-view --geometry``: matches in place against today's share (A10-7)."""
    with Engine.open(load_settings()) as engine:
        sessions = Sessions(engine)
        measured = sessions.evaluate_view_geometry(only_holdout=holdout)
        held = _held_out(sessions, holdout)
        tried = _record_tried(
            engine.settings, "same-view", [r.embedder for r in measured.whole], holdout
        )
    if as_json:
        out = {
            "whole": [_view_report_json(r) for r in measured.whole],
            "within_band": [_view_report_json(r) for r in measured.within_band],
            "band": list(measured.band),
            "in_band": measured.in_band,
            "holdout": held,
            "tried": dataclasses.asdict(tried),
        }
        typer.echo(json.dumps(_nan_to_null(out), indent=2, ensure_ascii=False, allow_nan=False))
        return
    console = _out()
    console.print(held, markup=False, soft_wrap=True)
    _print_tried(console, "same-view", tried, holdout)
    today = measured.whole[0]
    for set_id in today.skipped:
        console.print(f"Skipped {set_id}: its reference photo is gone.", markup=False)
    if not today.rows:
        console.print(
            "No labelled same-view searches yet. Start one with: "
            "calico eval new-view REFERENCE FOLDER",
            markup=False,
        )
        return
    lo, hi = measured.band
    console.print("Over every labelled photo:", markup=False)
    _print_view_comparison(console, measured.whole)
    console.print(
        f"Where the share is {lo:.2f} to {hi:.2f} ({_plural(measured.in_band, 'labelled photo')}"
        f", {_plural(len(measured.within_band[0].rows), 'reference')}), where it is least sure:",
        markup=False,
    )
    _print_view_comparison(console, measured.within_band)
    console.print(
        "In place: only the matches that agree on one placement of the reference (a shift, "
        "a zoom, a small turn). Minus the first: its mean AUC minus the share's, photo by "
        "photo.",
        markup=False,
        soft_wrap=True,
    )


def _with_candidate_options(settings: Settings, raw: Sequence[str]) -> Settings:
    """``NAME.KEY=VALUE`` options for candidate extractors, applied to these settings only.

    Never for an extractor computed for every photo: a changed option is a new version,
    so its cached vectors would no longer be found, and every photo would drop out.
    """
    options = {name: dict(values) for name, values in settings.extractor_options.items()}
    for item in raw:
        key, value = parse_option(item)
        name, dot, field = key.partition(".")
        if not dot or not name or not field:
            raise InvalidRequestError(
                f"Option {item!r} must look like NAME.KEY=VALUE, e.g. -o dinov2.pool=avg"
            )
        if name in settings.feature_extractors:
            raise InvalidRequestError(
                f"{name!r} is an extractor Calico computes for every photo; options here "
                "are for candidates measured with --embedder. Change it in calico.toml "
                "(extractor_options) and re-index instead."
            )
        options.setdefault(name, {})[field] = value
    return settings.model_copy(update={"extractor_options": options})


def _view_report_json(report: ViewReport) -> dict[str, object]:
    return {
        "embedder": report.embedder,
        "layout_weight": report.layout_weight,
        "default": report.default,
        "searches": [
            {
                "set_id": r.set_id,
                "text": r.text,
                "labelled": r.labelled,
                "same": r.same,
                "auc": r.auc,
                "at_default": dataclasses.asdict(r.at_default),
                "best": dataclasses.asdict(r.best),
                "at_pooled": dataclasses.asdict(r.at_pooled),
                "held_out": dataclasses.asdict(r.held_out),
            }
            for r in report.rows
        ],
        "pooled": dataclasses.asdict(report.pooled),
        "pooled_at_default": dataclasses.asdict(report.pooled_at_default),
        **{
            name: None if value is None else dataclasses.asdict(value)
            for name, value in (
                ("auc", report.auc),
                ("recall", report.recall),
                ("precision", report.precision),
                ("versus", report.versus),
            )
        },
        "skipped": report.skipped,
        "copies": report.copies,
    }


def _print_view_report(console: Console, report: ViewReport) -> None:
    table = Table(
        "search",
        "labelled",
        "same view",
        "AUC",
        f"at {report.default:g}",
        "best cut",
        "held out",
    )
    for r in report.rows:
        table.add_row(
            Text(r.text),
            str(r.labelled),
            str(r.same),
            _fmt(r.auc),
            _fmt_cut(r.at_default),
            _fmt_cut(r.best),
            _fmt_cut(r.held_out),
        )
    table.add_row(
        Text("all searches", style="bold"),
        str(sum(r.labelled for r in report.rows)),
        str(sum(r.same for r in report.rows)),
        "" if report.auc is None else _range(report.auc),
        _fmt_cut(report.pooled_at_default),
        _fmt_cut(report.pooled),
        "",
    )
    console.print(table)
    if report.recall is not None and report.precision is not None:
        console.print(
            f"All searches at {report.default:g}: recall {_interval(report.recall)}, "
            f"precision {_range(report.precision)}. [lo, hi]: where 95% fell when the "
            "references, then the photos labelled against each, were drawn again.",
            markup=False,
            soft_wrap=True,
        )
    best, now = report.pooled, report.pooled_at_default
    if report.embedder.endswith("(patches)"):
        console.print(
            f"Matched part by part ({report.embedder}): at the sure cut {report.default:g}, "
            f"{_pct(now.recall)} of same views are kept and {_pct(now.precision)} of what is "
            f"kept is one; the best single cut is {best.threshold:.3f}.",
            markup=False,
        )
    elif not math.isnan(best.threshold):
        console.print(
            f"Best min_similarity over all searches: {best.threshold:.2f} "
            f"(keeps {_pct(best.recall)} of same views, {_pct(best.precision)} of what it keeps "
            f"is one; the default {report.default:g} keeps {_pct(now.recall)}, "
            f"{_pct(now.precision)}). Set it in Fine-tune or with -o min_similarity=VALUE.",
            markup=False,
        )


def _print_view_comparison(console: Console, reports: Sequence[ViewReport]) -> None:
    """One row per way of matching: how well it separates, the best single cut, and how
    the cut chosen without each search does on it (the worst site is what to watch)."""
    table = Table(
        "subject",
        "angle",
        "mean AUC",
        "minus the first",
        "best cut",
        "worst site at it",
        "held out: mean recall",
        "held out: worst recall",
        "held out: worst precision",
    )
    for r in reports:
        held = [row.held_out for row in r.rows]
        table.add_row(
            r.embedder,
            f"{r.layout_weight:g}",
            _fmt(_mean([row.auc for row in r.rows])) if r.auc is None else _range(r.auc),
            "" if r.versus is None else _signed_range(r.versus),
            _fmt_cut(r.pooled),
            _pct(min(row.at_pooled.recall for row in r.rows)),
            _pct(_mean([h.recall for h in held])),
            _pct(min((h.recall for h in held if not math.isnan(h.recall)), default=math.nan)),
            _pct(min((h.precision for h in held if not math.isnan(h.precision)), default=math.nan)),
        )
    console.print(table)
    _print_search_note(console, [r.auc for r in reports[:1] if r.auc is not None], "AUC")
    sites = Table("search", *(f"{r.embedder} · {r.layout_weight:g}" for r in reports))
    for i, row in enumerate(reports[0].rows):
        sites.add_row(Text(row.text), *(_fmt_cut(r.rows[i].held_out) for r in reports))
    console.print(Text("Held out, by search (cut chosen on the other searches):"))
    console.print(sites)


def _fmt_cut(cut: Threshold) -> str:
    if math.isnan(cut.threshold):
        return "n/a"
    return f"{cut.threshold:.2f}: precision {_pct(cut.precision)}, recall {_pct(cut.recall)}"


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.0%}"


@eval_app.command("ratings")
def eval_ratings(
    legacy: Annotated[
        list[Path] | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="A feedback.csv from the previous system (stars or thumbs); repeatable. "
            "Your 👍/👎 on Strongest results always count.",
        ),
    ] = None,
    quality_weight: Annotated[
        list[float] | None,
        typer.Option(
            min=0, max=1, help="Also try strength with quality at this weight; repeatable."
        ),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
    holdout: Annotated[
        bool,
        typer.Option(
            "--holdout",
            help="Report on the hold-out alone: a last check of a final candidate, never for "
            "tuning (calico holdout show).",
        ),
    ] = False,
) -> None:
    """How well today's scores agree with your ratings: within each location, how often
    a photo you liked (4-5 stars, 👍) outscores one you did not (1-2 stars, 👎).

    AUC 0.5 is chance, 1.0 always right. strength is what "Strongest photographs"
    ranks by; quality and aesthetic are its parts. The "all" row counts every
    like/dislike pair, so a location with one dislike counts little. This is the
    score a model trained on your ratings would have to beat.
    """
    ratings = [rating for path in legacy or () for rating in read_legacy(path)]
    with Engine.open() as engine:
        sessions = Sessions(engine)
        ratings += sessions.thumbs()
        report = sessions.evaluate_ratings(
            ratings, tuple(quality_weight or ()), only_holdout=holdout
        )
        held = _held_out(sessions, holdout)
        tried = _record_tried(
            engine.settings,
            "ratings",
            ["strength", *(f"quality_weight={w:g}" for w in quality_weight or ())],
            holdout,
        )
    signals = list(dict.fromkeys(name for r in report.rows for name in r.auc))
    means = {name: pair_weighted_auc(report.rows, name) for name in signals}
    if as_json:
        out = {
            "locations": [dataclasses.asdict(r) for r in report.rows],
            "all_auc": means,
            "measured": None if report.measured is None else dataclasses.asdict(report.measured),
            "rated": report.rated,
            "missing": report.missing,
            "one_sided": report.one_sided,
            "holdout": held,
            "tried": dataclasses.asdict(tried),
        }
        typer.echo(json.dumps(_nan_to_null(out), indent=2, ensure_ascii=False, allow_nan=False))
        return
    console = _out()
    console.print(held, markup=False, soft_wrap=True)
    _print_tried(console, "ratings", tried, holdout)
    if not ratings:
        console.print(
            "No ratings yet: rate Strongest results with 👍/👎, or pass --legacy feedback.csv.",
            markup=False,
        )
        return
    sources = Counter(r.source for r in ratings)
    console.print(
        f"Counted: {sources['calico']} 👍/👎 on Strongest results (thumbs on stories judge "
        f"fit, so they are left out), {sources['stars']} star and {sources['thumbs']} thumb "
        "ratings from --legacy files.",
        markup=False,
        soft_wrap=True,
    )
    if report.missing:
        verb = "is" if report.missing == 1 else "are"
        console.print(
            f"{_plural(report.missing, 'rated photo')} {verb} not in the catalog (moved, "
            "deleted, or in a folder not indexed) and left out.",
            markup=False,
        )
    if report.one_sided:
        console.print(
            f"{_plural(report.one_sided, 'location')} had only likes or only dislikes, so "
            "nothing could be measured there.",
            markup=False,
        )
    if not report.rows:
        return
    table = Table("location", "rated", "liked", "not", *(f"AUC {s}" for s in signals))
    for r in report.rows:
        table.add_row(
            Text(r.location),
            str(r.rated),
            str(r.liked),
            str(r.disliked),
            *(_fmt(r.auc.get(s, math.nan)) for s in signals),
        )
    measured = report.measured
    table.add_row(
        Text("all", style="bold"),
        str(sum(r.rated for r in report.rows)),
        str(sum(r.liked for r in report.rows)),
        str(sum(r.disliked for r in report.rows)),
        *(_fmt(means[s]) if measured is None else _range(measured.auc[s]) for s in signals),
    )
    if measured is not None:
        table.add_row(
            Text("average of places"), "", "", "", *(_fmt(measured.macro[s]) for s in signals)
        )
    console.print(table)
    if measured is not None:
        _print_signals_note(console, measured)


def _print_signals_note(console: Console, measured: Signals) -> None:
    """What the ranges rest on, and each signal against the baseline, paired."""
    first = next(iter(measured.auc.values()), None)
    if first is None:
        return
    console.print(
        f"[lo, hi]: where 95% of each value fell when the places, then the photos within "
        f"them, were drawn again ({_counts(first)}). The average of places counts each "
        "place once.",
        markup=False,
        soft_wrap=True,
    )
    for name, versus in measured.versus.items():
        console.print(
            f"{name} minus {measured.baseline}: {_difference(versus)}.",
            markup=False,
            soft_wrap=True,
        )


@eval_app.command("examples")
def eval_examples(
    k: Annotated[
        int, typer.Option("--k", min=1, help="How many labelled matches serve as examples.")
    ] = 3,
    draws: Annotated[
        int, typer.Option(min=1, help="How many times to draw them; scores are averaged.")
    ] = 5,
    weight: Annotated[
        list[float] | None,
        typer.Option(min=0, help="How much the examples count beside the text; repeatable."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Would example photos help? Your labelled searches read by text alone, by text
    plus examples, and by examples alone.

    same search: K labelled matches are the examples and the rest of the search is
    scored (photos shot within two minutes of an example are left out). other site:
    the examples come from a search with the same words in other folders.
    """
    with Engine.open() as engine:
        rows = Sessions(engine).evaluate_examples(k, draws, tuple(weight or EXAMPLE_WEIGHTS))
        tried = _record_tried(
            engine.settings,
            "examples",
            [f"{v} k={k} draws={draws}" for v in dict.fromkeys(r.variant for r in rows)],
            holdout=False,
        )
    measured = example_intervals(rows)
    groups = list(dict.fromkeys((r.protocol, r.variant) for r in rows))
    means = {
        group: {
            "searches": sum((r.protocol, r.variant) == group for r in rows),
            **{
                m: _mean([getattr(r.score, m) for r in rows if (r.protocol, r.variant) == group])
                for m in METRICS
            },
        }
        for group in groups
    }
    if as_json:
        report: dict[str, Any] = {
            "searches": [
                {
                    "set_id": r.set_id,
                    "text": r.text,
                    "site": r.site,
                    "protocol": r.protocol,
                    "source": r.source,
                    "variant": r.variant,
                    **dataclasses.asdict(r.score),
                    "left_out": {"copies": r.copies, "nearby": r.nearby},
                }
                for r in rows
            ],
            "mean": {},
            "tried": dataclasses.asdict(tried),
        }
        for (protocol, variant), values in means.items():
            interval, versus = measured[protocol, variant]
            report["mean"].setdefault(protocol, {})[variant] = {
                **values,
                "auc_interval": dataclasses.asdict(interval),
                "versus_text": None if versus is None else dataclasses.asdict(versus),
            }
        typer.echo(json.dumps(_nan_to_null(report), indent=2, ensure_ascii=False, allow_nan=False))
        return
    console = _out()
    if not rows:
        console.print(
            "No labelled searches with enough matches to use as examples. Label some in the "
            "web interface (Label), or start one with: calico eval new TEXT FOLDER",
            markup=False,
        )
        return
    summary = Table(
        "examples from",
        Column("reading", no_wrap=True),
        "searches",
        "p@10",
        "nDCG@10",
        "AUC",
        "minus text alone",
    )
    for (protocol, variant), values in means.items():
        interval, versus = measured[protocol, variant]
        summary.add_row(
            protocol,
            variant,
            str(values["searches"]),
            *(_range(interval) if m == "auc" else _fmt(values[m]) for m in METRICS),
            "" if versus is None else _signed_range(versus),
        )
    console.print(summary)
    console.print(
        "[lo, hi] on the mean AUC: where 95% of it fell when the searches were drawn again "
        "(each search's AUC is already averaged over draws of examples). A difference "
        "marked ? could be zero.",
        markup=False,
        soft_wrap=True,
    )
    console.print(_examples_left_out(rows), markup=False, soft_wrap=True)
    variants = list(dict.fromkeys(r.variant for r in rows))
    sites = {r.set_id: r.site for r in rows}
    auc_table = Table("search", "site", "examples from", *variants, title="AUC by search")
    for key in dict.fromkeys((r.set_id, r.protocol, r.source) for r in rows):
        found = {r.variant: r for r in rows if (r.set_id, r.protocol, r.source) == key}
        first = next(iter(found.values()))
        auc_table.add_row(
            Text(first.text),
            Text(first.site),
            Text(
                first.protocol
                if first.protocol == "same search"
                else sites.get(first.source, first.source)
            ),
            *(_fmt(found[v].score.auc) if v in found else "" for v in variants),
        )
    console.print(auc_table)
    _print_tried(console, "examples", tried, holdout=False)


def _examples_left_out(rows: Sequence[ExampleEvaluated]) -> str:
    """Per draw of examples, summed over searches (each search once, not per reading)."""
    once = {(r.set_id, r.protocol, r.source): r for r in rows}.values()
    parts = []
    for protocol in dict.fromkeys(r.protocol for r in once):
        here = [r for r in once if r.protocol == protocol]
        text = f"{protocol}: {_fmt_count(sum(r.copies for r in here))} saved again"
        if protocol == "same search":
            near = _fmt_count(sum(r.nearby for r in here))
            text += f", {near} shot within two minutes of one"
        parts.append(text)
    return (
        "Never scored, so no example counts for itself (labelled photos per draw): "
        + "; ".join(parts)
        + "."
    )


def _fmt_count(value: float) -> str:
    return f"{value:g}" if value == int(value) else f"{value:.1f}"


def _pooled(calls: list[ClearCalls | None]) -> ClearCalls | None:
    """Clear calls summed over searches (a mean of shares would let one photo count as much
    as fifty); None if the model cannot tell."""
    known = [c for c in calls if c is not None]
    return sum(known[1:], known[0]) if known else None


def _clear_json(calls: ClearCalls | None) -> dict[str, int] | None:
    return None if calls is None else dataclasses.asdict(calls)


def _fmt_clear(calls: ClearCalls | None) -> str:
    if calls is None:
        return "n/a"
    if not calls.called:
        return "none"
    wrong = f", {calls.wrong} wrong" if calls.wrong else ""
    return f"{calls.matches} of {calls.called} right{wrong}"


def _mean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else math.nan


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.2f}"


def _nan_to_null(value: object) -> object:
    """JSON has neither NaN nor infinity; an undefined metric is null. Printed with
    ``allow_nan=False``, so one this misses fails loudly instead of printing bad JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_null(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nan_to_null(v) for v in value]
    return value


# -- feedback ------------------------------------------------------------------------------


@ratings_app.command("migrate")
def ratings_migrate(
    as_json: Annotated[bool, typer.Option("--json", help="Print the counts as JSON.")] = False,
) -> None:
    """Copy every rating in the label sets and the catalog's 👍/👎 into the ratings journal
    (<home>/ratings), then count them again against those sources.

    The label sets and the catalog are only read: nothing in them is changed or deleted.
    Running it again adds nothing, and a run that was stopped carries on where it stopped.
    Exits 1 if any answer is missing from the journal afterwards.
    """
    settings = load_settings()
    labels, runs = LabelStore(settings.labels_dir), RunStore(settings.runs_dir)
    journal = RatingsJournal(settings.ratings_dir)
    if settings.catalog_path.exists():
        with Catalog(settings.catalog_path) as catalog:
            report = migrate(labels, catalog, runs, journal)
    else:
        report = migrate(labels, None, runs, journal)
    if as_json:
        out = {
            "journal": str(report.journal),
            "complete": report.complete,
            "counts": {
                c.use: {"found": c.found, "written": c.written, "in_journal": c.in_journal}
                for c in report.counts
            },
            "sets": report.sets,
            "feedback_rows": report.feedback_rows,
            "lost_runs": report.lost_runs,
            "unreadable": [{"path": str(p), "why": why} for p, why in report.unreadable],
        }
        typer.echo(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        _print_migration(settings, report)
    if not report.complete:
        missing = sum(c.found - c.in_journal for c in report.counts)
        found = sum(c.found for c in report.counts)
        _err().print(
            f"error: {missing} of {found} answers are not in the journal ({report.journal}). "
            "Nothing was deleted; run calico ratings migrate again, and if it says the same, "
            "keep the journal folder and report it.",
            style="red",
            markup=False,
            soft_wrap=True,
        )
        raise typer.Exit(1)


def _print_migration(settings: Settings, report: MigrationReport) -> None:
    console = _out()
    kinds = ", ".join(f"{n} {kind}" for kind, n in sorted(report.sets.items()))
    sets = sum(report.sets.values())
    console.print(
        f"Read {_plural(sets, 'labelling set')}{f' ({kinds})' if kinds else ''} from "
        f"{settings.labels_dir} and {report.feedback_rows} 👍/👎 from {settings.catalog_path}; "
        "nothing in them was changed.",
        markup=False,
        soft_wrap=True,
    )
    table = Table("use", "in the old files", "written now", "in the journal")
    for c in report.counts:  # every use, so one with nothing to migrate shows as 0, not missing
        table.add_row(c.use, str(c.found), str(c.written), str(c.in_journal))
    table.add_row(
        "total",
        str(sum(c.found for c in report.counts)),
        str(sum(c.written for c in report.counts)),
        str(sum(c.in_journal for c in report.counts)),
    )
    console.print(table)
    if report.lost_runs:
        one = report.lost_runs == 1
        console.print(
            f"{report.lost_runs} 👍/👎 {'is' if one else 'are'} on {'a run' if one else 'runs'} "
            "that cannot be read (deleted, or saved by another version): kept, with the task "
            "unknown.",
            markup=False,
            soft_wrap=True,
        )
    for path, why in report.unreadable:
        console.print(
            f"Left out, and left untouched: {path} cannot be read ({why}).",
            markup=False,
            soft_wrap=True,
        )
    if report.complete:
        total = sum(c.found for c in report.counts)
        console.print(f"All {total} answers are in the journal: {report.journal}", markup=False)


@feedback_app.command("export")
def feedback_export(
    run: Annotated[str | None, typer.Option(help="Only 👍/👎 given on this run.")] = None,
) -> None:
    """Write every 👍/👎 you gave on results as CSV to stdout, oldest first; a cleared one
    has no rating. task and template are the run's: they say which review it answers."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["run_id", "content_key", "rating", "note", "at", "task", "template"])
    for answer in open_journal(load_settings()).answers():
        given_on = answer.shown[0].run if answer.shown else None
        if answer.use != "result" or (run is not None and given_on != run):
            continue
        scope = answer.scope if isinstance(answer.scope, dict) else {}
        writer.writerow(
            [
                given_on or "",
                answer.subject[0],
                "" if answer.value is None else answer.value,
                answer.note or "",
                answer.at.isoformat(),
                scope.get("task") or "",
                scope.get("template") or "",
            ]
        )
    typer.echo(buffer.getvalue(), nl=False)
