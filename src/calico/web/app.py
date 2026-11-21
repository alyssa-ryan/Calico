"""The web interface: a typed JSON API over ``Engine.select``, plus the built SPA.

Nothing here ranks or filters photographs. The API turns a form into a
``SelectionRequest``, hands it to a background ``JobRunner`` and returns the
saved ``SelectionResult``. The option menu, the folder tree and the per-task
options all come from the registries and settings, so adding a task never means
editing this module. Every response is a pydantic model so the OpenAPI schema
is complete: the frontend generates its TypeScript types from it.
"""

from __future__ import annotations

import fnmatch
import functools
import ipaddress
import json
import logging
import os
import re
import shutil
import sqlite3
import time
import unicodedata
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, JsonValue, ValidationError

from calico.catalog.indexer import thumbnail_path
from calico.catalog.store import Catalog
from calico.domain import MAX_DESCRIPTION, Pick, SelectionRequest, SelectionResult, place_key
from calico.engine import Engine
from calico.errors import (
    CalicoError,
    CatalogBusyError,
    ImageDecodeError,
    LabelSetNotFoundError,
    LabelSetUnreadableError,
    RunNotFoundError,
    RunUnreadableError,
    TasteUnreadableError,
)
from calico.imaging import open_image, prepare
from calico.labels import LabelSet, LabelStore
from calico.looks import discover_looks
from calico.paths import missing_folder, normalise
from calico.plugins import TASKS_GROUP, Registry
from calico.ratings_migration import open_journal
from calico.reading import read_description
from calico.reviews import Ratings, answered, labels_of, story_label, unfit
from calico.runs import RunStore
from calico.trained import TrainedModels
from calico.web.fingerprint import api_fingerprint
from calico.web.jobs import GENERIC_ERROR, JobRunner

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from calico.settings import CollectionConfig, Settings
    from calico.tasks.base import Task

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
DIST = HERE / "static" / "dist"
KEY_PATTERN = re.compile(r"^[0-9a-f]{32}$")
RUN_PATTERN = re.compile(r"^[\w-]+$")
FULL_SIDE = 2048
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
UPLOAD_KEEP_SECONDS = 7 * 24 * 3600
NOT_BUILT = "The web interface hasn't been built. Run: make ui"
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}
NO_CACHE = {"Cache-Control": "no-cache"}
_CACHE = {"Cache-Control": "public, max-age=86400"}
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
OTHER_SITES = frozenset({"cross-site", "same-site"})
"""``Sec-Fetch-Site`` values a browser sends for a request another page made. Another port
on this computer is ``same-site``: a site ignores the port."""
SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
"""Framed by another page, the UI's buttons could be clicked for the user; the page's own
inline theme script is why the policy is about framing only."""
WEB_BUSY_SECONDS = 2.0
"""How long a page's write to the catalog waits for another process's write before the
page says the catalog is busy (503); reads never wait."""
ONLY_HERE = (
    "Calico answers only this computer: it has no login, so nobody else on the network may "
    "browse or change anything."
)
_HOST = re.compile(r"^(?:\[(?P<v6>[0-9a-f:.]+)\]|(?P<name>[a-z0-9.-]+))(?::(?P<port>\d{1,5}))?$")


# -- response models ---------------------------------------------------------------------


class Health(BaseModel):
    engine: Literal["starting", "ready", "error"]
    error: str | None
    api: str = Field(description="Fingerprint of this server's API; see calico.web.fingerprint.")


class TaskInfo(BaseModel):
    name: str
    title: str
    description: str
    reference_use: Literal["required", "optional", "unused"]
    reference_label: str
    reference_help: str
    default_limit: int
    default_describe_weight: float
    uses_taste: bool
    """Whether choosing General or Yours can change what it picks."""
    options_schema: dict[str, JsonValue]


class TasteInfo(BaseModel):
    default: Literal["general", "yours"]
    """The template a search uses when it does not choose (calico.toml)."""
    trained: bool
    """Whether Yours has been trained (calico train taste); until then it falls back."""


class RootFolder(BaseModel):
    name: str
    path: str
    mounted: bool


class ChildFolder(BaseModel):
    name: str
    path: str
    has_children: bool


class FolderMatch(BaseModel):
    name: str
    path: str


class JobCreated(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    status: Literal["starting", "queued", "running", "done", "error"]
    done: int
    total: int
    step: Literal["reading", "comparing"] = Field(
        description="What done and total count: photos read, or compared part by part."
    )
    run_id: str | None
    label_set_id: str | None
    error: str | None


class RunListItem(BaseModel):
    run_id: str
    created_at: datetime
    task: str
    task_title: str
    criteria: str
    groups: int
    picks: int
    cover_keys: list[str]


class RunDetail(BaseModel):
    result: SelectionResult
    task_title: str
    reference_label: str
    feedback: dict[str, Literal[-1, 1]]


class LabelSetRequest(BaseModel):
    text: str = Field(default="", description="What to look for; unused with a reference.")
    folders: Annotated[list[Path], Field(min_length=1)]
    reference: Path | None = Field(
        default=None, description="Label photos as the same view as this one, or not."
    )
    taste: bool = Field(
        default=False,
        description="Rate photos for your taste (like, not sure, don't like); no text or "
        "reference.",
    )
    stories: bool = Field(
        default=False,
        description="Rate stories told from the folders (works, not sure, doesn't); no text "
        "or reference.",
    )
    compare: bool = Field(
        default=False,
        description="Rate the first picks of General and of Yours, merged and shuffled (like, "
        "not sure, don't like); no text or reference.",
    )
    anchor: bool = Field(
        default=False,
        description="Rate whether each photo could anchor a story (could, not sure, could "
        "not); no text or reference.",
    )


type LabelKind = Literal["describe", "same_view", "taste", "stories", "compare", "anchor"]


class LabelSetSummary(BaseModel):
    set_id: str
    created_at: datetime
    kind: LabelKind
    text: str
    folders: list[str]
    total: int
    labelled: int


class LabelPhoto(BaseModel):
    content_key: str
    name: str
    folder: str
    label: Literal[0, 1, 2] | None


class StoryPhotoItem(BaseModel):
    content_key: str
    name: str
    folder: str
    fits: bool


class LabelStory(BaseModel):
    """A story to rate. How it was told (template, variety) stays on the server, so it
    cannot sway the rating."""

    index: int
    photos: list[StoryPhotoItem]
    label: Literal[0, 1, 2] | None


class LabelSetDetail(BaseModel):
    set_id: str
    created_at: datetime
    kind: LabelKind
    text: str
    folders: list[str]
    scorer: str
    items: list[LabelPhoto]
    stories: list[LabelStory] = []


class LabelBody(BaseModel):
    label: Literal[0, 1, 2] | None


class StoryBody(BaseModel):
    label: Literal[0, 1, 2] | None
    doesnt_fit: list[str] = Field(
        default=[], description="Content keys of the story's photos that do not fit it."
    )


class ReadingPart(BaseModel):
    kind: str
    text: str


class ReadingResponse(BaseModel):
    used: bool
    """False when the configured criteria scorer reads the description whole."""
    parts: list[ReadingPart]


class FeedbackBody(BaseModel):
    run_id: str
    content_key: str
    rating: Literal[-1, 1] | None = Field(description="👍 1, 👎 -1, or null to clear it.")
    note: str = ""


def _host(value: str) -> tuple[str, int | None] | None:
    """The lower-cased name and port in a ``Host`` header; None unless it is a plain
    ``name[:port]`` (no user, path or backslash for a URL parser to read differently)."""
    found = _HOST.match(value.lower())
    if found is None:
        return None
    port = int(found["port"]) if found["port"] else None
    if port is not None and not 0 < port < 65536:
        return None
    return found["v6"] or found["name"], port


def _from_this_computer(request: Request) -> bool:
    """Whether the request's connection comes from this computer, whatever it claims."""
    if request.client is None:
        return False
    try:
        address = ipaddress.ip_address(request.client.host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


def _own_origin(request: Request, name: str, port: int | None) -> str:
    """The Origin a browser sends for this server's own pages, as addressed."""
    scheme = request.url.scheme
    shown = f"[{name}]" if ":" in name else name
    default = {"http": 80, "https": 443}.get(scheme)
    return f"{scheme}://{shown}" + ("" if port in (None, default) else f":{port}")


def _refusal(request: Request) -> Response | None:
    """Why the request is refused, or None to answer it.

    The app has no login, so it answers only this computer, and only requests addressed
    to it: any web page the user visits could otherwise reach it by DNS rebinding (its
    requests carry the page's host name). Changes must come from its own pages: another
    page, even another server on this computer, could otherwise submit a form to it (the
    browser adds that page's origin). Clients that are not browsers send no Origin.
    """
    if not _from_this_computer(request):
        return JSONResponse({"detail": ONLY_HERE}, status_code=403)
    host = _host(request.headers.get("host", ""))
    if host is None or host[0] not in LOOPBACK_HOSTS:
        return JSONResponse({"detail": "Unknown host."}, status_code=400)
    if request.method in UNSAFE_METHODS:
        origin = request.headers.get("origin")
        site = request.headers.get("sec-fetch-site", "").lower()
        if site in OTHER_SITES or (
            origin is not None and origin.lower() != _own_origin(request, *host)
        ):
            return JSONResponse({"detail": "Cross-site request refused."}, status_code=403)
    return None


def create_app(  # noqa: PLR0915 - one closure per route keeps the wiring in one place
    settings: Settings,
    *,
    synchronous_jobs: bool = False,
    engine_factory: Callable[[], Engine] | None = None,
) -> FastAPI:
    runner = JobRunner(
        engine_factory or (lambda: Engine.open(settings)), synchronous=synchronous_jobs
    )
    registry: Registry[Any] = Registry.from_entry_points(TASKS_GROUP, "task")
    runs = RunStore(settings.runs_dir)
    labels = LabelStore(settings.labels_dir)

    @functools.cache
    def ratings() -> Ratings:
        """Opened on first use: the first opening of a home migrates its old ratings."""
        return Ratings(labels, open_journal(settings), runs)

    _prune_uploads(settings.home / "uploads")
    infos: list[TaskInfo] | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        runner.warm_up()
        yield
        runner.shutdown()

    app = FastAPI(title="Calico", lifespan=lifespan)
    app.openapi = lambda: _openapi(app)  # type: ignore[method-assign]
    fingerprint = functools.cache(lambda: api_fingerprint(app.openapi()))

    @app.middleware("http")
    async def only_this_machine(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Answer only this computer, and changes only from our own pages (``_refusal``)."""
        response = _refusal(request) or await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    # -- helpers -----------------------------------------------------------------

    def task_infos() -> list[TaskInfo]:
        nonlocal infos
        if infos is None:
            infos = []
            for name in registry.names():
                try:
                    task: Task[Any] = registry.get(name)()
                except Exception:
                    logger.warning(
                        "task %r could not be loaded; hidden from the menu", name, exc_info=True
                    )
                    continue
                infos.append(
                    TaskInfo(
                        name=task.name,
                        title=task.title,
                        description=task.description,
                        reference_use=task.reference_use.value,
                        reference_label=getattr(task, "reference_label", "Reference photo"),
                        reference_help=getattr(task, "reference_help", ""),
                        default_limit=getattr(task, "default_limit", 12),
                        default_describe_weight=getattr(task, "default_describe_weight", 0.5),
                        uses_taste=getattr(task, "uses_taste", False),
                        options_schema=task.options_model.model_json_schema(),
                    )
                )
        return infos

    def collection_of(path: Path) -> CollectionConfig:
        """The collection whose root contains ``path`` spelt as the disk spells it (another
        case, NFD, a link: ``calico.paths``), else 403. A NUL byte, which no file name can
        hold, is a 422 rather than the server's error (audit A06-3)."""
        if "\0" in str(path):
            raise HTTPException(422, "A path cannot contain a NUL character.")
        real = normalise(path)
        for collection in settings.collections:
            if real.is_relative_to(collection.root):
                return collection
        raise HTTPException(403, f"{path} is outside every configured collection.")

    def load_run(run_id: str) -> SelectionResult:
        if not RUN_PATTERN.match(run_id):
            raise HTTPException(404, "No such run.")
        try:
            return runs.load(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except RunUnreadableError as exc:
            raise HTTPException(422, str(exc)) from exc

    def find_pick(result: SelectionResult, content_key: str) -> Pick | None:
        for group in result.groups:
            for pick in group.picks:
                if pick.content_key == content_key:
                    return pick
        return None

    def load_label_set(set_id: str) -> LabelSet:
        try:
            return labels.load(set_id)
        except LabelSetNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except LabelSetUnreadableError as exc:
            raise HTTPException(422, str(exc)) from exc

    def invalid(message: str) -> HTTPException:
        return HTTPException(422, message)

    def info_for(name: str) -> TaskInfo | None:
        return next((t for t in task_infos() if t.name == name), None)

    # -- api ---------------------------------------------------------------------

    @app.get("/api/health")
    def api_health() -> Health:
        engine_state, error = runner.engine_status
        return Health(engine=engine_state, error=error, api=fingerprint())

    @app.get("/api/tasks")
    def api_tasks() -> list[TaskInfo]:
        return task_infos()

    @app.get("/api/taste")
    def api_taste() -> TasteInfo:
        try:
            trained = TrainedModels(settings.models_dir).load_taste() is not None
        except TasteUnreadableError as exc:  # a search says why, and falls back to General
            logger.warning("%s", exc)
            trained = False
        return TasteInfo(default=settings.template, trained=trained)

    @app.get("/api/folders")
    def api_folders(path: str | None = None) -> list[RootFolder] | list[ChildFolder]:
        if path is None:
            return [
                RootFolder(name=c.name, path=str(c.root), mounted=c.root.is_dir())
                for c in settings.collections
            ]
        folder = Path(path)
        collection = collection_of(folder)
        if not folder.is_dir():
            raise HTTPException(404, missing_folder(folder))
        try:
            return _subfolders(folder, collection)
        except OSError as exc:
            raise HTTPException(403, f"{folder} cannot be read: {exc.strerror}.") from exc

    @app.get("/api/folders/search")
    def api_folder_search(q: str) -> list[FolderMatch]:
        """Every folder whose name contains ``q``, in every mounted collection, so a
        search such as "state hospital" can tick all its sites at once."""
        if not q.strip():
            raise HTTPException(422, "Type part of a folder name to search for.")
        return [
            FolderMatch(name=f.name, path=str(f))
            for c in settings.collections
            if c.root.is_dir()
            for f in _matching_folders(c, q)
        ]

    @app.post("/api/select")
    def api_select(
        request: Annotated[str, Form()],
        references: Annotated[list[UploadFile] | None, File()] = None,
    ) -> JobCreated:
        try:
            raw = json.loads(request)
            if not isinstance(raw, dict):
                raise invalid("The request must be a JSON object.")
            if raw.get("limit") is None:
                raw.pop("limit", None)
            parsed = SelectionRequest.model_validate(raw)
        except json.JSONDecodeError as exc:
            raise invalid(f"The request is not valid JSON: {exc}") from exc
        except ValidationError as exc:
            raise invalid(_readable(exc)) from exc
        try:
            task = registry.get(parsed.task)()
            task.parse_options(parsed.options)
            for path in (*parsed.folders, *parsed.references, *parsed.examples):
                collection_of(path)
        except CalicoError as exc:
            raise invalid(str(exc)) from exc
        except HTTPException as exc:
            raise invalid(str(exc.detail)) from exc

        saved = _save_uploads(settings.home / "uploads", references or [])
        final = parsed.model_copy(update={"references": (*parsed.references, *saved)})
        return JobCreated(job_id=runner.submit(final))

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str) -> JobStatus:
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(404, "No such job.")
        engine_state, _ = runner.engine_status
        starting = job.status == "queued" and engine_state == "starting"
        return JobStatus(
            status="starting" if starting else job.status,
            done=job.done,
            total=job.total,
            step=job.step,
            run_id=job.run_id,
            label_set_id=job.label_set_id,
            error=job.error,
        )

    @app.get("/api/runs")
    def api_runs(limit: int = 50) -> list[RunListItem]:
        titles = {t.name: t.title for t in task_infos()}
        items = []
        for summary in runs.recent(max(1, min(limit, 500))):
            result = runs.load(summary.run_id)
            keys = [p.content_key for g in result.groups for p in g.picks]
            items.append(
                RunListItem(
                    run_id=summary.run_id,
                    created_at=summary.created_at,
                    task=summary.task,
                    task_title=titles.get(summary.task, summary.task),
                    criteria=summary.criteria,
                    groups=summary.groups,
                    picks=summary.picks,
                    cover_keys=keys[:4],
                )
            )
        return items

    @app.get("/api/runs/{run_id}")
    def api_run(run_id: str) -> RunDetail:
        result = load_run(run_id)
        info = info_for(result.request.task)
        return RunDetail(
            result=result,
            task_title=info.title if info else result.request.task,
            reference_label=info.reference_label if info else "Reference photo",
            feedback=ratings().on_result(result),
        )

    @app.post("/api/feedback", status_code=204)
    def api_feedback(body: FeedbackBody) -> Response:
        if find_pick(load_run(body.run_id), body.content_key) is None:
            raise HTTPException(404, "That photo is not part of this run.")
        ratings().rate_result(body.run_id, body.content_key, body.rating, body.note)
        return Response(status_code=204)

    # -- images ------------------------------------------------------------------

    @app.get("/api/reading")
    def api_reading(
        text: str, folders: Annotated[list[Path] | None, Query()] = None
    ) -> ReadingResponse:
        """How "Describe it" will read ``text`` for these folders (shown as you type)."""
        if len(text) > MAX_DESCRIPTION:
            raise invalid(
                f"The description is {len(text)} characters; Calico reads at most "
                f"{MAX_DESCRIPTION} characters, so shorten it."
            )
        for folder in folders or []:
            try:
                collection_of(folder)
            except HTTPException as exc:
                raise invalid(str(exc.detail)) from exc
        with Catalog(settings.catalog_path, busy_seconds=WEB_BUSY_SECONDS) as catalog:
            places = catalog.locations_under(folders or [])
        used = settings.criteria_scorer in ("parts", "parts-looks")
        measured = settings.criteria_scorer == "parts-looks"
        reading = read_description(
            text,
            places,
            hemisphere=settings.hemisphere,
            looks=discover_looks(settings.looks_files) if measured else (),
        )
        return ReadingResponse(
            used=used,
            parts=[ReadingPart(kind=k, text=t) for k, t in reading.summary()],
        )

    # -- labelling -------------------------------------------------------------

    @app.post("/api/label-sets")
    def api_new_label_set(body: LabelSetRequest) -> JobCreated:
        for folder in body.folders:
            try:
                collection_of(folder)
            except HTTPException as exc:
                raise invalid(str(exc.detail)) from exc
        sessions = [
            (name, submit)
            for asked, name, submit in (
                (body.anchor, "Rating anchors", runner.submit_anchor_labelling),
                (body.compare, "Compare Strongest", runner.submit_compare_labelling),
                (body.stories, "Rating stories", runner.submit_story_labelling),
                (body.taste, "Rating for your taste", runner.submit_taste_labelling),
            )
            if asked
        ]
        if sessions:
            name, submit = sessions[0]
            if len(sessions) > 1 or body.text.strip() or body.reference is not None:
                raise invalid(f"{name} takes folders only: no text, reference or other session.")
            return JobCreated(job_id=submit(body.folders))
        if body.reference is None:
            if not body.text.strip():
                raise invalid("Say what to look for: labelling needs a description.")
            return JobCreated(job_id=runner.submit_labelling(body.text, body.folders))
        uploads = (settings.home / "uploads").resolve()
        if not body.reference.resolve().is_relative_to(uploads):
            try:
                collection_of(body.reference)
            except HTTPException as exc:
                raise invalid(str(exc.detail)) from exc
        return JobCreated(job_id=runner.submit_view_labelling(body.reference, body.folders))

    @app.get("/api/label-sets")
    def api_label_sets() -> list[LabelSetSummary]:
        current = ratings().current()
        return [
            LabelSetSummary(
                set_id=s.set_id,
                created_at=s.created_at,
                kind=s.kind,
                text=s.text,
                folders=[str(f) for f in s.folders],
                total=s.total,
                labelled=answered(s, current),
            )
            for s in labels.all()
        ]

    @app.get("/api/label-sets/{set_id}")
    def api_label_set(set_id: str) -> LabelSetDetail:
        s = load_label_set(set_id)
        current = ratings().current()
        given = labels_of(s, current)
        strangers = [unfit(story, current) for story in s.stories]
        return LabelSetDetail(
            set_id=s.set_id,
            created_at=s.created_at,
            kind=s.kind,
            text=s.text,
            folders=[str(f) for f in s.folders],
            scorer=s.scorer,
            items=[
                LabelPhoto(
                    content_key=i.content_key,
                    name=i.path.name,
                    folder=i.path.parent.name,
                    label=given.get(i.content_key),
                )
                for i in s.items
            ],
            stories=[
                LabelStory(
                    index=n,
                    photos=[
                        StoryPhotoItem(
                            content_key=p.content_key,
                            name=p.path.name,
                            folder=p.path.parent.name,
                            fits=p.content_key not in strangers[n],
                        )
                        for p in story.photos
                    ],
                    label=story_label(story, current),
                )
                for n, story in enumerate(s.stories)
            ],
        )

    @app.put("/api/label-sets/{set_id}/labels/{content_key}", status_code=204)
    def api_set_label(set_id: str, content_key: str, body: LabelBody) -> None:
        load_label_set(set_id)
        try:
            ratings().rate_photo(set_id, content_key, body.label)
        except CalicoError as exc:
            raise invalid(str(exc)) from exc

    @app.put("/api/label-sets/{set_id}/stories/{index}", status_code=204)
    def api_set_story(set_id: str, index: int, body: StoryBody) -> None:
        load_label_set(set_id)
        try:
            ratings().rate_story(set_id, index, body.label, doesnt_fit=body.doesnt_fit)
        except CalicoError as exc:
            raise invalid(str(exc)) from exc

    @app.delete("/api/label-sets/{set_id}", status_code=204)
    def api_delete_label_set(set_id: str) -> None:
        """Moved to the labels' trash; a set this Calico cannot read can be deleted too."""
        try:
            labels.delete(set_id)
        except LabelSetNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/label-sets/{set_id}/image/{content_key}")
    def label_image(set_id: str, content_key: str) -> Response:
        label_set = load_label_set(set_id)
        paths = {i.content_key: i.path for i in label_set.items} | {
            p.content_key: p.path for s in label_set.stories for p in s.photos
        }
        path = paths.get(content_key)
        if path is None:
            raise HTTPException(404, "That photo is not part of this labelling set.")
        return _jpeg(path)

    @app.get("/api/label-sets/{set_id}/reference")
    def label_reference(set_id: str) -> Response:
        reference = load_label_set(set_id).reference
        if reference is None:
            raise HTTPException(404, "This labelling set has no reference photo.")
        return _jpeg(reference)

    @app.get("/thumbs/{content_key}")
    def thumb(content_key: str) -> FileResponse:
        if not KEY_PATTERN.match(content_key):
            raise HTTPException(404, "No such thumbnail.")
        path = thumbnail_path(settings.thumbs_dir, content_key)
        if not path.is_file():
            raise HTTPException(404, "No such thumbnail.")
        return FileResponse(path, media_type="image/jpeg", headers=_CACHE)

    @app.get("/runs/{run_id}/image/{content_key}")
    def run_image(run_id: str, content_key: str) -> Response:
        pick = find_pick(load_run(run_id), content_key)
        if pick is None:
            raise HTTPException(404, "That photo is not part of this run.")
        return _jpeg(pick.path)

    @app.get("/runs/{run_id}/reference/{index}")
    def run_reference(run_id: str, index: int) -> Response:
        refs = load_run(run_id).request.references
        if not 0 <= index < len(refs):
            raise HTTPException(404, "No such reference.")
        return _jpeg(refs[index])

    @app.exception_handler(sqlite3.OperationalError)
    async def catalog_busy(_: Request, exc: sqlite3.OperationalError) -> Response:
        if "locked" in str(exc) or "busy" in str(exc):
            return JSONResponse({"detail": str(CatalogBusyError())}, status_code=503)
        logger.error("unhandled error", exc_info=exc)
        return JSONResponse({"detail": GENERIC_ERROR}, status_code=500)

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception) -> Response:
        logger.error("unhandled error", exc_info=exc)
        return JSONResponse({"detail": GENERIC_ERROR}, status_code=500)

    # -- the built single-page app -------------------------------------------------

    @app.get("/assets/{asset:path}", include_in_schema=False)
    def asset(asset: str) -> FileResponse:
        target = _inside(DIST / "assets", asset)
        if target is None:
            raise HTTPException(404, "No such asset.")
        return FileResponse(target, headers=IMMUTABLE)

    @app.get("/{page:path}", include_in_schema=False)
    def spa(page: str) -> Response:
        if page.startswith(("api/", "thumbs/")):
            raise HTTPException(404, "Not found.")
        index = DIST / "index.html"
        if not index.is_file():
            return HTMLResponse(
                f"<!doctype html><title>Calico</title><p>{NOT_BUILT}</p>", status_code=503
            )
        target = _inside(DIST, page) if page else None
        return FileResponse(target or index, headers=NO_CACHE)

    return app


def _inside(base: Path, relative: str) -> Path | None:
    """``base/relative`` if it is an existing file that really lives under ``base``."""
    if "\0" in relative:  # no file is named so, and resolve() would raise
        return None
    candidate = (base / relative).resolve()
    if candidate.is_file() and candidate.is_relative_to(base.resolve()):
        return candidate
    return None


def _openapi(app: FastAPI) -> dict[str, Any]:
    """The schema, plus ``SelectionRequest`` (it travels as a JSON string inside a form)."""
    if app.openapi_schema is None:
        from fastapi.openapi.utils import get_openapi  # noqa: PLC0415

        schema = get_openapi(title=app.title, version="1", routes=app.routes)
        extra = SelectionRequest.model_json_schema(ref_template="#/components/schemas/{model}")
        defs = extra.pop("$defs", {})
        schemas = schema.setdefault("components", {}).setdefault("schemas", {})
        schemas.update({**defs, "SelectionRequest": extra})
        app.openapi_schema = schema
    return app.openapi_schema


def _jpeg(path: Path) -> Response:
    try:
        with open_image(path) as image:
            data = BytesIO()
            prepare(image, FULL_SIDE).save(data, "JPEG", quality=90)
    except (ImageDecodeError, FileNotFoundError) as exc:
        raise HTTPException(404, f"{path.name} cannot be read (is the drive mounted?).") from exc
    return Response(data.getvalue(), media_type="image/jpeg", headers=_CACHE)


def _visible(entry: Path, collection: CollectionConfig, root: Path) -> bool:
    """A folder the picker may show: not excluded, and not a link out of the collection."""
    if any(fnmatch.fnmatch(entry.name, pat) for pat in collection.exclude):
        return False
    try:
        return entry.is_dir() and entry.resolve().is_relative_to(root)
    except OSError:
        return False


def _name_key(name: str) -> str:
    """Case, spacing and Unicode form do not count (macOS may store names decomposed)."""
    return place_key(unicodedata.normalize("NFC", name))


def _matching_folders(collection: CollectionConfig, query: str) -> list[Path]:
    """Folders under the collection whose name contains ``query``, sorted by path.

    A match is not searched further: ticking it covers everything inside, so the
    matches below it would only repeat it. ``scandir`` tells folders from files
    without a stat each, which matters on a USB drive holding 100,000 photos; only
    links are resolved, and they are never followed, so a link cannot loop the walk.
    Unreadable folders are skipped.
    """
    root = collection.root.resolve()
    wanted = _name_key(query)
    found: list[Path] = []
    pending = [collection.root]
    while pending:
        folder = pending.pop()
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if any(fnmatch.fnmatch(entry.name, pat) for pat in collection.exclude):
                        continue
                    link = entry.is_symlink()
                    if not entry.is_dir() or (link and not _resolves_under(Path(entry.path), root)):
                        continue
                    if wanted in _name_key(entry.name):
                        found.append(Path(entry.path))
                    elif not link:
                        pending.append(Path(entry.path))
        except OSError:
            continue
    return sorted(found, key=lambda f: str(f).lower())


def _resolves_under(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root)
    except OSError:
        return False


def _subfolders(folder: Path, collection: CollectionConfig) -> list[ChildFolder]:
    root = collection.root.resolve()
    entries = sorted(
        (e for e in folder.iterdir() if _visible(e, collection, root)), key=lambda e: e.name.lower()
    )
    return [
        ChildFolder(name=e.name, path=str(e), has_children=_has_children(e, collection, root))
        for e in entries
    ]


def _has_children(folder: Path, collection: CollectionConfig, root: Path) -> bool:
    try:
        return any(
            e.is_dir()
            and not any(fnmatch.fnmatch(e.name, pat) for pat in collection.exclude)
            and e.resolve().is_relative_to(root)
            for e in folder.iterdir()
        )
    except OSError:
        return False


def _safe_name(raw: str) -> str:
    clean = re.sub(r"[^\w.-]+", "_", Path(raw).name).strip("._") or "reference"
    path = Path(clean)
    return path.stem[:100] + path.suffix[:16]


def _save_uploads(base: Path, files: list[UploadFile]) -> list[Path]:
    """Stream uploads to disk (capped per file); leave nothing behind on failure."""
    files = [f for f in files if f.filename]
    if not files:
        return []
    target = base / uuid.uuid4().hex
    target.mkdir(parents=True)
    saved = []
    try:
        for i, upload in enumerate(files):
            path = target / f"{i:02d}-{_safe_name(upload.filename or '')}"
            size = 0
            with path.open("wb") as out:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            413, f"{upload.filename} is larger than {MAX_UPLOAD_BYTES >> 20} MB."
                        )
                    out.write(chunk)
            _check_image(path, upload.filename or path.name)
            saved.append(path)
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return saved


def _check_image(path: Path, shown: str) -> None:
    """Refuse an unreadable reference now, not after the whole archive has been indexed."""
    try:
        with open_image(path) as image:
            image.load()
    except (ImageDecodeError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(
            422,
            f"“{shown}” is not an image Calico can read. Use a JPEG, PNG, HEIC, TIFF or WebP.",
        ) from exc


def _prune_uploads(base: Path) -> None:
    """Uploaded references are only needed while a run is fresh; drop week-old ones."""
    if not base.is_dir():
        return
    cutoff = time.time() - UPLOAD_KEEP_SECONDS
    for entry in base.iterdir():
        if entry.is_dir() and entry.stat().st_mtime < cutoff:
            shutil.rmtree(entry, ignore_errors=True)


def _readable(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'request'}: {e['msg']}" for e in exc.errors()
    )
