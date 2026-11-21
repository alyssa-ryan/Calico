"""Background jobs: selections, and labelling sets.

Indexing a few thousand photos takes minutes, so the web layer runs this work
off the request thread and lets the page poll for progress. SQLite connections
must stay on the thread that opened them, so one worker thread owns its own
``Engine`` (created lazily there); nothing else ever touches it.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from calico.errors import CalicoError
from calico.sessions import Sessions

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from calico.catalog.indexer import ProgressFn, ProgressStep
    from calico.domain import SelectionRequest
    from calico.engine import Engine

logger = logging.getLogger(__name__)

type Status = Literal["queued", "running", "done", "error"]
type JobKind = Literal["select", "label"]
type Work = Callable[[Engine, ProgressFn], str]
"""Does the job on the worker's engine and returns the id of what it made."""
type EngineState = Literal["starting", "ready", "error"]

KEEP_FINISHED_SECONDS = 3600
MAX_JOBS = 200

GENERIC_ERROR = "Something went wrong. The details are in the server log."


class EngineUnavailableError(CalicoError):
    """The worker could not build its engine; the message says why."""


@dataclass
class Job:
    work: Work
    kind: JobKind = "select"
    status: Status = "queued"
    done: int = 0
    total: int = 0
    step: ProgressStep = "reading"
    result_id: str | None = None
    error: str | None = None
    finished_at: float | None = None

    @property
    def run_id(self) -> str | None:
        return self.result_id if self.kind == "select" else None

    @property
    def label_set_id(self) -> str | None:
        return self.result_id if self.kind == "label" else None


class JobRunner:
    def __init__(self, engine_factory: Callable[[], Engine], *, synchronous: bool = False) -> None:
        self._factory = engine_factory
        self._synchronous = synchronous
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._eager = False
        self._engine_state: EngineState = "ready" if synchronous else "starting"
        self._engine_error: str | None = None

    @property
    def engine_status(self) -> tuple[EngineState, str | None]:
        with self._lock:
            return self._engine_state, self._engine_error

    def warm_up(self) -> None:
        """Start the worker now and build its engine, so the first request doesn't pay for it."""
        if not self._synchronous:
            self._eager = True
            self._ensure_worker()

    def submit(self, request: SelectionRequest) -> str:
        """Run a selection; the finished job's ``run_id`` names the saved run."""
        return self._submit(Job(lambda engine, progress: engine.select(request, progress).run_id))

    def submit_labelling(self, text: str, folders: Sequence[Path]) -> str:
        """Build a labelling set; the finished job's ``label_set_id`` names it."""
        return self._submit(
            Job(
                lambda engine, progress: Sessions(engine).label_set(text, folders, progress).set_id,
                kind="label",
            )
        )

    def submit_taste_labelling(self, folders: Sequence[Path]) -> str:
        """Build a taste-rating set from ``folders``."""
        return self._submit(
            Job(
                lambda engine, progress: Sessions(engine).label_taste(folders, progress).set_id,
                kind="label",
            )
        )

    def submit_anchor_labelling(self, folders: Sequence[Path]) -> str:
        """Build a set asking, photo by photo, whether it could anchor a story."""
        return self._submit(
            Job(
                lambda engine, progress: Sessions(engine).label_anchors(folders, progress).set_id,
                kind="label",
            )
        )

    def submit_story_labelling(self, folders: Sequence[Path]) -> str:
        """Build a story-rating set from ``folders``."""
        return self._submit(
            Job(
                lambda engine, progress: Sessions(engine).label_stories(folders, progress).set_id,
                kind="label",
            )
        )

    def submit_compare_labelling(self, folders: Sequence[Path]) -> str:
        """Build a Compare Strongest set from ``folders``."""
        return self._submit(
            Job(
                lambda engine, progress: Sessions(engine).label_compare(folders, progress).set_id,
                kind="label",
            )
        )

    def submit_view_labelling(self, reference: Path, folders: Sequence[Path]) -> str:
        """Build a same-view labelling set around ``reference``."""
        return self._submit(
            Job(
                lambda engine, progress: (
                    Sessions(engine).label_same_view(reference, folders, progress).set_id
                ),
                kind="label",
            )
        )

    def _submit(self, job: Job) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._prune()
            self._jobs[job_id] = job
        if self._synchronous:
            # A fresh engine per job: test client requests hop between threads.
            with self._factory() as engine:
                self._run(job_id, engine)
            return job_id
        self._ensure_worker()
        self._queue.put(job_id)
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else Job(**vars(job))

    def shutdown(self, timeout: float = 5) -> None:
        thread = self._thread
        if thread is not None:
            self._queue.put(None)
            thread.join(timeout=timeout)
            if not thread.is_alive():
                self._thread = None  # a still-running worker is kept so no second one starts

    def _prune(self) -> None:
        """Forget finished jobs after an hour, or the oldest ones beyond ``MAX_JOBS``."""
        now = time.time()
        finished = sorted(
            (j.finished_at, i) for i, j in self._jobs.items() if j.finished_at is not None
        )
        doomed = [i for t, i in finished if now - t > KEEP_FINISHED_SECONDS]
        excess = len(self._jobs) - len(doomed) - MAX_JOBS + 1
        doomed += [i for _, i in finished if i not in doomed][: max(0, excess)]
        for job_id in doomed:
            del self._jobs[job_id]

    # -- worker ------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, name="calico-jobs", daemon=True)
                self._thread.start()

    def _work(self) -> None:
        engine: Engine | None = None
        try:
            if self._eager:
                engine = self._build_engine()
            while (job_id := self._queue.get()) is not None:
                try:
                    if engine is None:
                        engine = self._build_engine()
                    if engine is None:
                        self._fail(job_id, EngineUnavailableError(self._engine_error or ""))
                    else:
                        self._run(job_id, engine)
                except Exception as exc:
                    self._fail(job_id, exc)
        finally:
            if engine is not None:
                engine.close()

    def _build_engine(self) -> Engine | None:
        try:
            engine = self._factory()
        except Exception as exc:
            message = self._describe(exc, "engine")
            with self._lock:
                self._engine_state, self._engine_error = "error", message
            return None
        with self._lock:
            self._engine_state, self._engine_error = "ready", None
        return engine

    @staticmethod
    def _describe(exc: Exception, what: str) -> str:
        if isinstance(exc, CalicoError):
            return str(exc)
        logger.error("%s failed", what, exc_info=exc)
        return GENERIC_ERROR

    def _run(self, job_id: str, engine: Engine) -> None:
        self._update(job_id, status="running")

        def progress(done: int, total: int, step: ProgressStep) -> None:
            self._update(job_id, done=done, total=total, step=step)

        try:
            made = self._jobs[job_id].work(engine, progress)
        except Exception as exc:
            self._fail(job_id, exc)
        else:
            self._update(job_id, status="done", result_id=made)

    def _fail(self, job_id: str, exc: Exception) -> None:
        self._update(job_id, status="error", error=self._describe(exc, f"job {job_id}"))

    def _update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            if job.status in ("done", "error"):
                job.finished_at = time.time()
