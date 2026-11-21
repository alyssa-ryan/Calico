"""The web interface end to end, on a synthetic archive and the ``tiny`` embedder."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import warnings
from datetime import datetime, timedelta
from typing import Any

import pytest

from calico.domain import SelectionRequest, SelectionResult
from calico.engine import Engine
from calico.errors import ConfigError
from calico.ratings_store import RatingsJournal
from calico.runs import RunStore
from calico.sessions import Sessions
from calico.settings import CollectionConfig, Settings
from calico.tasks.best import BestTask
from calico.web.app import create_app
from calico.web.jobs import GENERIC_ERROR, Job, JobRunner
from calico.web.openapi import fingerprint_ts, schema_json
from tests import factories as f

with warnings.catch_warnings():
    # Starlette's test client warns that it prefers httpx2; httpx is what the project pins.
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from pathlib import Path

T0 = datetime(2021, 3, 6, 10, 0, 0)
LOCAL = "http://127.0.0.1:8765"  # the app only answers requests addressed to this machine


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    for seed in (1, 2, 3, 4):
        f.save_jpeg(
            f.scene(seed),
            root / "Old Mill - March 2021" / f"view{seed}.jpg",
            captured_at=T0 + timedelta(hours=seed),
        )
    for seed in (10, 11):
        f.save_jpeg(f.scene(seed), root / "Harbor" / f"h{seed}.jpg", captured_at=T0)
    (root / ".hidden").mkdir()
    (root / "Harbor" / "deep" / "deeper").mkdir(parents=True)
    return root


@pytest.fixture
def settings(archive: Path, calico_home: Path) -> Settings:
    return Settings(
        home=calico_home,
        embedder="tiny",
        describe_embedder="tiny",
        view_embedder="tiny",
        collections=(CollectionConfig(name="archive", root=archive),),
    )


def local(app: Any, **kwargs: Any) -> TestClient:
    """A client on this computer, as the browser on it is."""
    return TestClient(app, base_url=LOCAL, client=("127.0.0.1", 50000), **kwargs)


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return local(create_app(settings, synchronous_jobs=True))


def submit(client: TestClient, request: dict[str, Any], files: list[Path] | None = None) -> Any:
    handles = [("references", (p.name, p.read_bytes(), "image/jpeg")) for p in files or []]
    return client.post("/api/select", data={"request": json.dumps(request)}, files=handles)


def run_best(client: TestClient, archive: Path, **extra: Any) -> str:
    response = submit(client, {"task": "best", "folders": [str(archive)], "limit": 3, **extra})
    assert response.status_code == 200, response.text
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "done", job
    return str(job["run_id"])


def saved(settings: Settings, run_id: str) -> SelectionResult:
    return RunStore(settings.runs_dir).load(run_id)


# -- spa ----------------------------------------------------------------------------------


@pytest.fixture
def dist(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    built = tmp_path / "dist"
    (built / "assets").mkdir(parents=True)
    (built / "index.html").write_text("<!doctype html><title>spa</title>")
    (built / "assets" / "app-123.js").write_text("console.log(1)")
    (built / "favicon.svg").write_text("<svg/>")
    monkeypatch.setattr("calico.web.app.DIST", built)
    return built


def test_unbuilt_ui_says_how_to_build_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("calico.web.app.DIST", Path("/nonexistent-dist"))
    response = client.get("/")
    assert response.status_code == 503
    assert "Run: make ui" in response.text
    assert "text/html" in response.headers["content-type"]


@pytest.mark.usefixtures("dist")
def test_spa_fallback_serves_index_without_caching(client: TestClient) -> None:
    for path in ("/", "/runs", "/runs/20250101-000000-abcd", "/anything/else"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert "<title>spa</title>" in response.text
        assert response.headers["cache-control"] == "no-cache"
    assert client.get("/favicon.svg").text == "<svg/>"


@pytest.mark.usefixtures("dist")
def test_assets_are_immutable_and_traversal_safe(client: TestClient) -> None:
    response = client.get("/assets/app-123.js")
    assert response.status_code == 200
    assert "immutable" in response.headers["cache-control"]
    assert client.get("/assets/missing.js").status_code == 404
    assert client.get("/assets/..%2Findex.html").status_code == 404
    assert "root:" not in client.get("/..%2F..%2Fetc/passwd").text


@pytest.mark.usefixtures("dist")
def test_unknown_api_paths_are_json_404_not_the_spa(client: TestClient) -> None:
    for path in ("/api/nope", "/thumbs/x/y"):
        response = client.get(path)
        assert response.status_code == 404
        assert "detail" in response.json()


def test_unknown_run_is_a_404(client: TestClient) -> None:
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/..%2Fetc").status_code == 404


def test_runs_list_starts_empty_then_has_covers(client: TestClient, archive: Path) -> None:
    assert client.get("/api/runs").json() == []
    run_id = run_best(client, archive, limit=6)
    (item,) = client.get("/api/runs").json()
    assert item["run_id"] == run_id
    assert item["task_title"] == "Strongest photographs"
    assert item["picks"] == 6
    assert len(item["cover_keys"]) == 4


def test_a_run_from_another_version_does_not_break_past_runs(
    client: TestClient, archive: Path, settings: Settings
) -> None:
    good, other = run_best(client, archive, limit=2), run_best(client, archive, limit=3)
    path = RunStore(settings.runs_dir).path(other) / "result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["request"]["from_a_newer_calico"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    listed = client.get("/api/runs")
    assert listed.status_code == 200
    assert [item["run_id"] for item in listed.json()] == [good]
    opened = client.get(f"/api/runs/{other}")
    assert opened.status_code == 422
    assert "another version" in opened.json()["detail"]


def test_unexpected_exceptions_are_json_with_a_generic_message(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: RunStore, run_id: str) -> None:
        raise RuntimeError("secret internals")

    monkeypatch.setattr(RunStore, "load", boom)
    app = create_app(settings, synchronous_jobs=True)
    response = local(app, raise_server_exceptions=False).get("/api/runs/abc")
    assert response.status_code == 500
    assert response.json() == {"detail": GENERIC_ERROR}


# -- openapi ---------------------------------------------------------------------------------


def test_openapi_describes_every_response_precisely(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    schemas = schema["components"]["schemas"]
    for name in (
        "Health", "TaskInfo", "RootFolder", "ChildFolder", "JobCreated", "JobStatus",
        "RunListItem", "RunDetail", "SelectionResult", "SelectionRequest", "FeedbackBody",
    ):  # fmt: skip
        assert name in schemas, name
    for path, method in (
        ("/api/health", "get"), ("/api/tasks", "get"), ("/api/folders", "get"),
        ("/api/select", "post"), ("/api/jobs/{job_id}", "get"), ("/api/runs", "get"),
        ("/api/runs/{run_id}", "get"),
    ):  # fmt: skip
        ok = schema["paths"][path][method]["responses"]["200"]["content"]["application/json"]
        assert ok["schema"], path
    assert schemas["JobStatus"]["properties"]["status"]["enum"] == [
        "starting", "queued", "running", "done", "error",
    ]  # fmt: skip


def test_committed_openapi_matches_the_app() -> None:
    committed = Path(__file__).parents[2] / "frontend" / "openapi.json"
    if not committed.exists():
        pytest.skip("frontend/openapi.json not generated yet")
    assert json.loads(committed.read_text()) == json.loads(schema_json()), "run make api-types"


def test_committed_fingerprint_matches_the_app() -> None:
    """The page compares it with the server's to tell a server that needs a restart."""
    committed = Path(__file__).parents[2] / "frontend" / "src" / "api" / "fingerprint.ts"
    assert committed.read_text() == fingerprint_ts(), "run make api-types"


def test_health_reports_the_api_fingerprint(client: TestClient) -> None:
    api = client.get("/api/health").json()["api"]
    assert re.fullmatch(r"[0-9a-f]{16}", api)
    assert f'"{api}"' in fingerprint_ts()


# -- api: tasks and folders ---------------------------------------------------------


def test_api_tasks_describes_each_task_with_a_schema(client: TestClient) -> None:
    tasks = {t["name"]: t for t in client.get("/api/tasks").json()}
    best = tasks["best"]
    assert best["title"]
    assert best["description"]
    assert best["reference_use"] == "unused"
    assert best["options_schema"]["properties"]["diversity"]["maximum"] == 1
    assert "criteria_weight" not in best["options_schema"]["properties"]
    assert best["default_describe_weight"] == 0.5
    assert tasks["same_view"]["default_describe_weight"] == 0.3


def test_folders_lists_roots(client: TestClient, archive: Path) -> None:
    assert client.get("/api/folders").json() == [
        {"name": "archive", "path": str(archive), "mounted": True}
    ]


def test_folders_lists_children_sorted_and_skips_hidden(client: TestClient, archive: Path) -> None:
    rows = client.get("/api/folders", params={"path": str(archive)}).json()
    assert [r["name"] for r in rows] == ["Harbor", "Old Mill - March 2021"]
    assert [r["has_children"] for r in rows] == [True, False]
    inner = client.get("/api/folders", params={"path": str(archive / "Harbor")}).json()
    assert [r["name"] for r in inner] == ["deep"]


@pytest.mark.parametrize("where", ["escape", "absolute", "symlink"])
def test_folders_rejects_paths_outside_collections(
    client: TestClient, archive: Path, tmp_path: Path, where: str
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (archive / "link").symlink_to(outside)
    path = {
        "escape": f"{archive}/../outside",
        "absolute": str(outside),
        "symlink": str(archive / "link"),
    }[where]
    assert client.get("/api/folders", params={"path": path}).status_code == 403


def test_folders_hides_symlinks_that_leave_the_collection(
    client: TestClient, archive: Path, tmp_path: Path
) -> None:
    (tmp_path / "outside").mkdir()
    (archive / "link").symlink_to(tmp_path / "outside")
    names = [r["name"] for r in client.get("/api/folders", params={"path": str(archive)}).json()]
    assert "link" not in names


def test_folders_missing_directory_is_404(client: TestClient, archive: Path) -> None:
    response = client.get("/api/folders", params={"path": str(archive / "gone")})
    assert response.status_code == 404
    assert "not mounted" in response.json()["detail"]


def search(client: TestClient, q: str) -> list[dict[str, Any]]:
    response = client.get("/api/folders/search", params={"q": q})
    assert response.status_code == 200, response.text
    rows: list[dict[str, Any]] = response.json()
    return rows


def test_folder_search_finds_names_at_any_depth_ignoring_case_and_spacing(
    client: TestClient, archive: Path
) -> None:
    assert search(client, "old  MILL") == [
        {"name": "Old Mill - March 2021", "path": str(archive / "Old Mill - March 2021")}
    ]
    assert [r["path"] for r in search(client, "deeper")] == [str(archive / "Harbor/deep/deeper")]


def test_folder_search_returns_a_match_but_not_the_matches_inside_it(
    client: TestClient, archive: Path
) -> None:
    """Ticking the outer match covers the inner one, so listing both would only repeat it."""
    assert [r["path"] for r in search(client, "deep")] == [str(archive / "Harbor/deep")]


def test_folder_search_lists_every_match_sorted(client: TestClient, archive: Path) -> None:
    (archive / "Mill Pond").mkdir()
    names = [r["name"] for r in search(client, "mill")]
    assert names == ["Mill Pond", "Old Mill - March 2021"]


def test_folder_search_skips_excluded_and_escaping_folders(
    client: TestClient, archive: Path, tmp_path: Path
) -> None:
    (archive / ".hidden mill").mkdir()
    (tmp_path / "outside mill").mkdir()
    (archive / "link mill").symlink_to(tmp_path / "outside mill")
    assert [r["name"] for r in search(client, "mill")] == ["Old Mill - March 2021"]


def test_folder_search_lists_links_inside_the_collection_but_never_follows_them(
    client: TestClient, archive: Path
) -> None:
    (archive / "Harbor" / "loop").symlink_to(archive)  # following it would never end
    (archive / "Mill shortcut").symlink_to(archive / "Old Mill - March 2021")
    assert [r["name"] for r in search(client, "mill")] == ["Mill shortcut", "Old Mill - March 2021"]
    assert [r["name"] for r in search(client, "loop")] == ["loop"]


def test_folder_search_needs_some_text(client: TestClient) -> None:
    assert client.get("/api/folders/search", params={"q": "   "}).status_code == 422


# -- selection flow -------------------------------------------------------------------


def test_select_flow_returns_the_saved_run(client: TestClient, archive: Path) -> None:
    run_id = run_best(client, archive, criteria="autumn light")
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["task_title"] == "Strongest photographs"
    assert detail["feedback"] == {}
    result = detail["result"]
    assert result["request"]["criteria"] == "autumn light"
    assert any("cannot read text" in n for n in result["notes"])
    assert result["stages"]
    pick = result["groups"][0]["picks"][0]
    assert pick["path"].endswith(".jpg")
    assert pick["evidence"]


def test_null_limit_means_the_default(client: TestClient, archive: Path) -> None:
    response = submit(client, {"task": "best", "folders": [str(archive)], "limit": None})
    assert response.status_code == 200


def test_archive_paths_can_be_references(
    client: TestClient, archive: Path, settings: Settings
) -> None:
    ref = archive / "Old Mill - March 2021" / "view1.jpg"
    response = submit(
        client,
        {"task": "same_view", "folders": [str(archive)], "references": [str(ref)], "limit": 3},
    )
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "done", job
    assert saved(settings, job["run_id"]).request.references == (ref,)


def test_archive_references_outside_collections_are_422(client: TestClient, archive: Path) -> None:
    response = submit(
        client,
        {"task": "best", "folders": [str(archive)], "references": ["/etc/hosts"]},
    )
    assert response.status_code == 422


def test_examples_outside_collections_are_422(client: TestClient, archive: Path) -> None:
    response = submit(
        client, {"task": "best", "folders": [str(archive)], "examples": ["/etc/hosts"]}
    )
    assert response.status_code == 422


def test_a_run_with_examples_records_their_keys(
    client: TestClient, archive: Path, settings: Settings
) -> None:
    example = archive / "Harbor" / "h10.jpg"
    run_id = run_best(client, archive, examples=[str(example)])
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["result"]["request"]["examples"] == [str(example.resolve())]
    (key,) = detail["result"]["example_keys"]
    assert client.get(f"/thumbs/{key}").status_code == 200
    assert saved(settings, run_id).example_keys == (key,)


def test_invalid_requests_are_422_with_readable_messages(client: TestClient, archive: Path) -> None:
    ok = {"task": "best", "folders": [str(archive)]}
    cases: list[tuple[dict[str, Any], str]] = [
        ({**ok, "limit": 0}, "limit"),
        ({**ok, "folders": []}, "folders"),
        ({**ok, "task": "nonsense"}, "nonsense"),
        ({**ok, "options": {"diversity": 7}}, "Invalid options"),
        ({**ok, "options": {"criteria_weight": 1}}, "describe_weight"),
        ({**ok, "describe_weight": 1.5}, "describe_weight"),
        ({**ok, "folders": ["/not/a/collection"]}, "outside every configured collection"),
        ({**ok, "folders": [f"{archive}/../.."]}, "outside"),
    ]
    for body, expected in cases:
        response = submit(client, body)
        assert response.status_code == 422, body
        assert expected in response.json()["detail"], response.json()
    bad_json = client.post("/api/select", data={"request": "{oops"})
    assert bad_json.status_code == 422
    assert "JSON" in bad_json.json()["detail"]
    not_object = client.post("/api/select", data={"request": "[1]"})
    assert not_object.status_code == 422


def test_calico_error_reaches_the_user_verbatim(client: TestClient, archive: Path) -> None:
    response = submit(client, {"task": "best", "folders": [str(archive / "missing")]})
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "error"
    assert "does not exist or is not mounted" in job["error"]
    assert job["run_id"] is None


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/api/jobs/abc").status_code == 404


def test_uploads_are_saved_and_used_as_references(
    client: TestClient, archive: Path, settings: Settings
) -> None:
    reference = archive / "Old Mill - March 2021" / "view1.jpg"
    response = submit(
        client,
        {"task": "same_view", "folders": [str(archive)], "limit": 3},
        files=[reference],
    )
    assert response.status_code == 200, response.text
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "done", job
    result = saved(settings, job["run_id"])
    (ref,) = result.request.references
    assert ref.is_relative_to(settings.home / "uploads")
    assert ref.read_bytes() == reference.read_bytes()
    assert client.get(f"/runs/{job['run_id']}/reference/0").headers["content-type"] == "image/jpeg"
    assert client.get(f"/runs/{job['run_id']}/reference/5").status_code == 404


def test_upload_filenames_are_sanitised(
    client: TestClient, archive: Path, settings: Settings
) -> None:
    src = archive / "Harbor" / "h10.jpg"
    response = client.post(
        "/api/select",
        data={"request": json.dumps({"task": "best", "folders": [str(archive)]})},
        files=[("references", ("../../evil name?.jpg", src.read_bytes(), "image/jpeg"))],
    )
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    (ref,) = saved(settings, job["run_id"]).request.references
    assert ref.parent.parent == settings.home / "uploads"
    assert ref.name == "00-evil_name_.jpg"


# -- images ---------------------------------------------------------------------------


def test_thumbs_served_and_bad_keys_404(
    client: TestClient, archive: Path, settings: Settings
) -> None:
    run_id = run_best(client, archive)
    key = saved(settings, run_id).groups[0].picks[0].content_key
    response = client.get(f"/thumbs/{key}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert client.get("/thumbs/" + "0" * 32).status_code == 404
    assert client.get("/thumbs/..%2F..%2Fcalico.toml").status_code == 404
    assert client.get("/thumbs/short").status_code == 404


def test_full_image_only_for_keys_in_the_run(
    client: TestClient, archive: Path, settings: Settings
) -> None:
    run_id = run_best(client, archive, limit=1)
    result = saved(settings, run_id)
    (pick,) = result.groups[0].picks
    response = client.get(f"/runs/{run_id}/image/{pick.content_key}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert client.get(f"/runs/{run_id}/image/{'a' * 32}").status_code == 404
    assert (
        client.get(f"/runs/{run_id}/image/../../etc/passwd").headers["content-type"] != "image/jpeg"
    )


def test_full_image_missing_file_is_404(
    client: TestClient, archive: Path, settings: Settings
) -> None:
    run_id = run_best(client, archive, limit=1)
    pick = saved(settings, run_id).groups[0].picks[0]
    moved = pick.path.with_suffix(".moved")
    pick.path.rename(moved)
    assert client.get(f"/runs/{run_id}/image/{pick.content_key}").status_code == 404


# -- feedback -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        "evil.example:8765",
        "127.0.0.1.evil.example",
        "localhost.:80",
        "evil.example\\@localhost",
        "localhost:99999",
    ],
)
def test_requests_addressed_to_another_host_are_refused(client: TestClient, host: str) -> None:
    """DNS rebinding: a hostile page's requests reach us carrying its own host name."""
    for path in ("/api/runs", "/api/folders", "/"):
        response = client.get(path, headers={"Host": host})
        assert response.status_code == 400, path


@pytest.mark.parametrize("host", ["localhost:8765", "127.0.0.1", "[::1]:8765"])
def test_loopback_host_names_are_accepted(client: TestClient, host: str) -> None:
    assert client.get("/api/health", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize("address", ["192.168.1.20", "10.0.0.7", "fe80::1", "testclient"])
def test_requests_from_another_computer_are_refused(settings: Settings, address: str) -> None:
    """The app has no login: nobody else on the network may browse or change anything,
    whatever Host header they send."""
    app = create_app(settings, synchronous_jobs=True)
    other = TestClient(app, base_url=LOCAL, client=(address, 50000))
    for method, path in (("GET", "/api/health"), ("GET", "/api/folders"), ("GET", "/")):
        assert other.request(method, path, headers={"Host": "localhost"}).status_code == 403
    assert other.delete("/api/label-sets/x").status_code == 403
    assert "only this computer" in other.get("/api/health").json()["detail"]


def test_requests_from_this_computer_over_ipv6_are_answered(settings: Settings) -> None:
    app = create_app(settings, synchronous_jobs=True)
    here = TestClient(app, base_url="http://[::1]:8765", client=("::1", 50000))
    assert here.get("/api/health").status_code == 200


def test_cross_site_posts_are_refused(client: TestClient, archive: Path) -> None:
    """A page on another site, or another server on this computer, can submit a form to
    us; it must not start jobs or rate photos."""
    request = {"task": "best", "folders": [str(archive)]}
    origins = (
        "http://evil.example",
        "null",
        "http://127.0.0.1.evil.example",
        "http://localhost:3000",  # another local server
        "http://127.0.0.1:1",
        "https://127.0.0.1:8765",  # the right host and port, the wrong scheme
    )
    for origin in origins:
        headers = {"Origin": origin}
        response = client.post(
            "/api/select", data={"request": json.dumps(request)}, headers=headers
        )
        assert response.status_code == 403, origin
        feedback = {"run_id": "r", "content_key": "k", "rating": 1}
        assert client.post("/api/feedback", json=feedback, headers=headers).status_code == 403


@pytest.mark.parametrize("site", ["cross-site", "same-site"])
def test_a_browser_saying_the_request_comes_from_another_site_is_refused(
    client: TestClient, site: str
) -> None:
    feedback = {"run_id": "r", "content_key": "k", "rating": 1}
    headers = {"Sec-Fetch-Site": site}
    assert client.post("/api/feedback", json=feedback, headers=headers).status_code == 403


def test_same_origin_and_originless_posts_are_accepted(client: TestClient, archive: Path) -> None:
    request = {"task": "best", "folders": [str(archive)]}
    for headers in (
        {"Origin": LOCAL},
        {"Origin": LOCAL, "Sec-Fetch-Site": "same-origin"},
        {"Origin": "http://localhost:5173", "Host": "localhost:5173"},  # make ui-dev's proxy
        {"Sec-Fetch-Site": "none"},  # typed into the address bar
        {},
    ):
        response = client.post(
            "/api/select", data={"request": json.dumps(request)}, headers=headers
        )
        assert response.status_code == 200, headers


def test_pages_cannot_be_framed_or_sniffed(client: TestClient) -> None:
    """Framed by another page, the UI's Delete buttons could be clicked for you."""
    headers = client.get("/api/health").headers
    assert headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"


def test_pages_and_ratings_answer_while_another_process_writes_the_catalog(
    client: TestClient, archive: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = run_best(client, archive, limit=1)
    pick = saved(settings, run_id).groups[0].picks[0]
    monkeypatch.setattr("calico.web.app.WEB_BUSY_SECONDS", 0.2)
    writer = sqlite3.connect(settings.catalog_path, autocommit=True)
    writer.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        assert client.get("/api/reading", params={"text": "a red door"}).status_code == 200
        assert client.get(f"/api/runs/{run_id}").status_code == 200
        assert time.monotonic() - started < 2
        feedback = {"run_id": run_id, "content_key": pick.content_key, "rating": 1}
        # A 👍 goes to the ratings journal, so it never waits for the catalog.
        assert client.post("/api/feedback", json=feedback).status_code == 204
    finally:
        writer.execute("ROLLBACK")
        writer.close()


def test_a_label_set_that_cannot_be_read_leaves_the_others_usable_and_can_be_deleted(
    client: TestClient, settings: Settings
) -> None:
    settings.labels_dir.mkdir(parents=True)
    bad = settings.labels_dir / "20990101-000000-bad0.json"
    bad.write_text('{"set_id": "x"}')  # written by another branch
    assert client.get("/api/label-sets").status_code == 200
    detail = client.get(f"/api/label-sets/{bad.stem}")
    assert detail.status_code == 422
    assert "left untouched" in detail.json()["detail"]
    assert client.delete(f"/api/label-sets/{bad.stem}").status_code == 204
    assert not bad.exists()


def test_a_description_too_long_to_read_is_refused_while_typing(client: TestClient) -> None:
    response = client.get("/api/reading", params={"text": "in " * 400})
    assert response.status_code == 422
    assert "500 characters" in response.json()["detail"]


def test_feedback_is_stored(client: TestClient, archive: Path, settings: Settings) -> None:
    run_id = run_best(client, archive)
    pick = saved(settings, run_id).groups[0].picks[0]
    response = client.post(
        "/api/feedback",
        json={"run_id": run_id, "content_key": pick.content_key, "rating": -1, "note": "dull"},
    )
    assert response.status_code == 204
    (answer,) = RatingsJournal(settings.ratings_dir).answers()
    assert (answer.use, answer.subject, answer.value, answer.note) == (
        "result",
        (pick.content_key,),
        -1,
        "dull",
    )
    assert answer.scope == {"task": "best", "template": "general"}
    assert answer.shown[0].run == run_id
    assert client.get(f"/api/runs/{run_id}").json()["feedback"] == {pick.content_key: -1}


def test_feedback_can_be_cleared(client: TestClient, archive: Path, settings: Settings) -> None:
    run_id = run_best(client, archive)
    key = saved(settings, run_id).groups[0].picks[0].content_key
    client.post("/api/feedback", json={"run_id": run_id, "content_key": key, "rating": 1})
    cleared = client.post(
        "/api/feedback", json={"run_id": run_id, "content_key": key, "rating": None}
    )
    assert cleared.status_code == 204
    assert client.get(f"/api/runs/{run_id}").json()["feedback"] == {}


def test_feedback_rejected_for_photo_not_in_run(client: TestClient, archive: Path) -> None:
    run_id = run_best(client, archive, limit=1)
    response = client.post(
        "/api/feedback", json={"run_id": run_id, "content_key": "b" * 32, "rating": 1}
    )
    assert response.status_code == 404
    bad_rating = client.post(
        "/api/feedback", json={"run_id": run_id, "content_key": "b" * 32, "rating": 3}
    )
    assert bad_rating.status_code == 422
    unknown_run = client.post(
        "/api/feedback", json={"run_id": "nope", "content_key": "b" * 32, "rating": 1}
    )
    assert unknown_run.status_code == 404


# -- job runner -----------------------------------------------------------------------


def wait_for(runner: JobRunner, job_id: str, status: str) -> Job:
    for _ in range(300):
        job = runner.get(job_id)
        assert job is not None
        if job.status in (status, "done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_background_worker_runs_jobs_and_shuts_down(settings: Settings, archive: Path) -> None:
    runner = JobRunner(lambda: Engine.open(settings))
    job_id = runner.submit(SelectionRequest(task="best", folders=(archive,), limit=2))
    job = wait_for(runner, job_id, "done")
    runner.shutdown()
    assert job.status == "done", job.error
    assert job.total > 0
    assert job.done == job.total
    assert job.step == "reading"


def test_unexpected_errors_are_generic_and_engine_failure_is_reported(
    settings: Settings, archive: Path
) -> None:
    def broken() -> Any:
        raise RuntimeError("secret internals")

    runner = JobRunner(broken)
    job_id = runner.submit(SelectionRequest(task="best", folders=(archive,)))
    job = wait_for(runner, job_id, "error")
    runner.shutdown()
    assert job.error == GENERIC_ERROR
    assert runner.get("missing") is None


def test_lifespan_shuts_the_runner_down(settings: Settings) -> None:
    with local(create_app(settings)) as client:
        assert client.get("/api/tasks").status_code == 200


# -- review fixes ---------------------------------------------------------------------


def test_unreadable_folder_is_403_json(client: TestClient, archive: Path) -> None:
    locked = archive / "Harbor"
    locked.chmod(0)
    try:
        response = client.get("/api/folders", params={"path": str(locked)})
    finally:
        locked.chmod(0o755)
    assert response.status_code == 403
    assert "cannot be read" in response.json()["detail"]


def test_long_upload_names_are_truncated(
    client: TestClient, archive: Path, settings: Settings
) -> None:
    src = archive / "Harbor" / "h10.jpg"
    name = "a" * 400 + ".jpg"
    response = client.post(
        "/api/select",
        data={"request": json.dumps({"task": "best", "folders": [str(archive)]})},
        files=[("references", (name, src.read_bytes(), "image/jpeg"))],
    )
    assert response.status_code == 200
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    (ref,) = saved(settings, job["run_id"]).request.references
    assert len(ref.name) < 150
    assert ref.suffix == ".jpg"


def test_oversize_upload_is_413_and_leaves_nothing(
    client: TestClient, archive: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("calico.web.app.MAX_UPLOAD_BYTES", 10)
    response = client.post(
        "/api/select",
        data={"request": json.dumps({"task": "best", "folders": [str(archive)]})},
        files=[("references", ("big.jpg", b"x" * 100, "image/jpeg"))],
    )
    assert response.status_code == 413
    assert list((settings.home / "uploads").iterdir()) == []


def test_an_upload_that_is_not_an_image_is_refused_at_once(
    client: TestClient, archive: Path, settings: Settings
) -> None:
    response = client.post(
        "/api/select",
        data={"request": json.dumps({"task": "same_view", "folders": [str(archive)]})},
        files=[("references", ("notes.jpg", b"just some text", "image/jpeg"))],
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "notes.jpg" in detail
    assert "uploads" not in detail  # names the user's file, not where we stored it
    assert list((settings.home / "uploads").iterdir()) == []


def test_old_upload_dirs_are_pruned_at_startup(settings: Settings) -> None:
    old = settings.home / "uploads" / "old"
    fresh = settings.home / "uploads" / "fresh"
    old.mkdir(parents=True)
    fresh.mkdir()
    ancient = time.time() - 8 * 24 * 3600
    os.utime(old, (ancient, ancient))
    create_app(settings, synchronous_jobs=True)
    assert not old.exists()
    assert fresh.exists()


def test_finished_jobs_are_evicted(settings: Settings, archive: Path) -> None:
    runner = JobRunner(lambda: Engine.open(settings), synchronous=True)
    request = SelectionRequest(task="best", folders=(archive,), limit=1)
    first = runner.submit(request)
    job = runner.get(first)
    assert job is not None
    assert job.finished_at is not None
    runner._jobs[first].finished_at = time.time() - 7200
    second = runner.submit(request)
    assert runner.get(first) is None
    assert runner.get(second) is not None


def test_job_count_is_capped(
    settings: Settings, archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("calico.web.jobs.MAX_JOBS", 2)
    runner = JobRunner(lambda: Engine.open(settings), synchronous=True)
    request = SelectionRequest(task="best", folders=(archive,), limit=1)
    ids = [runner.submit(request) for _ in range(4)]
    assert [runner.get(i) is not None for i in ids] == [False, False, True, True]


def test_no_second_worker_after_shutdown_timeout(settings: Settings, archive: Path) -> None:
    release = threading.Event()

    def slow() -> Engine:
        release.wait(5)
        return Engine.open(settings)

    runner = JobRunner(slow)
    request = SelectionRequest(task="best", folders=(archive,), limit=1)
    runner.submit(request)
    runner.shutdown(timeout=0.05)
    before = threading.active_count()
    runner.submit(request)
    assert threading.active_count() == before
    release.set()
    runner.shutdown()


def test_health_is_ready_in_synchronous_mode(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert (health["engine"], health["error"]) == ("ready", None)


def test_lifespan_warms_the_engine_up(settings: Settings) -> None:
    with local(create_app(settings)) as client:
        for _ in range(100):
            health = client.get("/api/health").json()
            if health["engine"] != "starting":
                break
            time.sleep(0.05)
        assert (health["engine"], health["error"]) == ("ready", None)


def test_health_reports_engine_failure_and_queued_jobs_say_starting(
    settings: Settings, archive: Path
) -> None:
    release = threading.Event()

    def broken() -> Engine:
        release.wait(5)
        raise ConfigError("embedder missing: install the ml extra")

    with local(create_app(settings, engine_factory=broken)) as client:
        job_id = submit(client, {"task": "best", "folders": [str(archive)]}).json()["job_id"]
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "starting"
        release.set()
        for _ in range(100):
            health = client.get("/api/health").json()
            if health["engine"] == "error":
                break
            time.sleep(0.05)
        assert health["error"] == "embedder missing: install the ml extra"
        for _ in range(100):
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] == "error":
                break
            time.sleep(0.05)
        assert job["error"] == "embedder missing: install the ml extra"


def test_saved_ratings_are_returned_with_the_run(
    client: TestClient, archive: Path, settings: Settings
) -> None:
    run_id = run_best(client, archive)
    key = saved(settings, run_id).groups[0].picks[0].content_key
    body = {"run_id": run_id, "content_key": key}
    client.post("/api/feedback", json={**body, "rating": 1})
    client.post("/api/feedback", json={**body, "rating": -1})
    assert client.get(f"/api/runs/{run_id}").json()["feedback"] == {key: -1}


def test_options_keep_the_order_the_task_declares(client: TestClient) -> None:
    """The form lists options as the task orders them (most important first), not A-Z."""
    tasks = {t["name"]: t for t in client.get("/api/tasks").json()}
    declared = list(BestTask.options_model.model_fields)
    assert list(tasks["best"]["options_schema"]["properties"]) == declared
    assert tasks["best"]["default_limit"] >= 1
    assert "reference_label" in tasks["best"]
    assert "reference_help" in tasks["best"]


# -- labelling ------------------------------------------------------------------------------


@pytest.fixture
def reads_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tiny embedder cannot read text; give the baseline scorer a fake reader."""
    from calico.criteria import SCORERS, BaselineScorer, PartsScorer  # noqa: PLC0415

    monkeypatch.setitem(SCORERS, "baseline", lambda e, _s: BaselineScorer(f.FirstAxisReader(e)))
    monkeypatch.setitem(
        SCORERS, "parts", lambda e, s: PartsScorer(f.FirstAxisReader(e), s.hemisphere)
    )


def new_label_set(client: TestClient, archive: Path, text: str = "fog") -> str:
    created = client.post("/api/label-sets", json={"text": text, "folders": [str(archive)]})
    assert created.status_code == 200, created.text
    job = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert job["status"] == "done", job
    assert job["run_id"] is None
    return str(job["label_set_id"])


def test_tasks_say_whether_your_taste_changes_them(client: TestClient) -> None:
    uses = {t["name"]: t["uses_taste"] for t in client.get("/api/tasks").json()}
    assert uses == uses | {"best": True, "triptych": True, "same_view": False}
    assert uses["recurring_views"] is False


def test_the_taste_endpoint_says_which_template_is_default_and_if_yours_is_trained(
    client: TestClient,
) -> None:
    assert client.get("/api/taste").json() == {"default": "general", "trained": False}


def test_a_taste_session_needs_no_description(client: TestClient, archive: Path) -> None:
    created = client.post("/api/label-sets", json={"folders": [str(archive)], "taste": True})
    assert created.status_code == 200, created.text
    job = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert job["status"] == "done", job
    detail = client.get(f"/api/label-sets/{job['label_set_id']}").json()
    assert detail["kind"] == "taste"
    assert detail["text"] == "Your taste: archive"
    assert client.get("/api/label-sets").json()[0]["kind"] == "taste"


def test_a_taste_session_takes_no_description_or_reference(
    client: TestClient, archive: Path
) -> None:
    body = {"folders": [str(archive)], "taste": True, "text": "fog"}
    assert client.post("/api/label-sets", json=body).status_code == 422


def test_an_anchor_session_needs_folders_alone(client: TestClient, archive: Path) -> None:
    created = client.post("/api/label-sets", json={"folders": [str(archive)], "anchor": True})
    assert created.status_code == 200, created.text
    job = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert job["status"] == "done", job
    detail = client.get(f"/api/label-sets/{job['label_set_id']}").json()
    assert (detail["kind"], detail["text"]) == ("anchor", "Could it anchor a story: archive")
    body = {"folders": [str(archive)], "anchor": True, "taste": True}
    assert client.post("/api/label-sets", json=body).status_code == 422


def new_story_set(client: TestClient, archive: Path) -> str:
    created = client.post("/api/label-sets", json={"folders": [str(archive)], "stories": True})
    assert created.status_code == 200, created.text
    job = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert job["status"] == "done", job
    return str(job["label_set_id"])


def test_a_story_session_is_rated_story_by_story_without_saying_how_each_was_told(
    client: TestClient, archive: Path
) -> None:
    set_id = new_story_set(client, archive)
    detail = client.get(f"/api/label-sets/{set_id}").json()
    assert (detail["kind"], detail["text"], detail["items"]) == ("stories", "Stories: archive", [])
    first = detail["stories"][0]
    assert set(first) == {"index", "photos", "label"}  # not its arms or evidence
    assert first["index"] == 0
    assert first["label"] is None
    photo = first["photos"][2]
    assert set(photo) == {"content_key", "name", "folder", "fits"}
    assert photo["fits"] is True
    (summary,) = client.get("/api/label-sets").json()
    assert summary == summary | {"kind": "stories", "total": len(detail["stories"]), "labelled": 0}

    body = {"label": 2, "doesnt_fit": [photo["content_key"]]}
    assert client.put(f"/api/label-sets/{set_id}/stories/0", json=body).status_code == 204
    rated = client.get(f"/api/label-sets/{set_id}").json()["stories"][0]
    assert rated["label"] == 2
    assert [p["fits"] for p in rated["photos"]] == [True, True, False]
    assert client.get("/api/label-sets").json()[0]["labelled"] == 1
    image = client.get(f"/api/label-sets/{set_id}/image/{photo['content_key']}")
    assert image.headers["content-type"] == "image/jpeg"


def test_bad_story_ratings_are_refused(client: TestClient, archive: Path) -> None:
    set_id = new_story_set(client, archive)
    url = f"/api/label-sets/{set_id}/stories"
    assert client.put(f"{url}/99", json={"label": 1, "doesnt_fit": []}).status_code == 422
    assert client.put(f"{url}/0", json={"label": 1, "doesnt_fit": ["nope"]}).status_code == 422
    assert client.put(f"{url}/0", json={"label": 3, "doesnt_fit": []}).status_code == 422
    assert client.put("/api/label-sets/nope/stories/0", json={"label": 1}).status_code == 404


def test_a_story_session_takes_folders_only(client: TestClient, archive: Path) -> None:
    for extra in ({"text": "fog"}, {"taste": True}):
        body = {"folders": [str(archive)], "stories": True, **extra}
        assert client.post("/api/label-sets", json=body).status_code == 422


def new_compare_job(client: TestClient, archive: Path) -> dict[str, Any]:
    body = {"folders": [str(archive / "Harbor")], "compare": True}
    created = client.post("/api/label-sets", json=body)
    assert created.status_code == 200, created.text
    return dict(client.get(f"/api/jobs/{created.json()['job_id']}").json())


def test_compare_strongest_needs_a_trained_yours(client: TestClient, archive: Path) -> None:
    job = new_compare_job(client, archive)
    assert job["status"] == "error"
    assert "calico train taste" in job["error"]


def test_a_compare_session_is_rated_photo_by_photo_without_saying_whose_they_are(
    client: TestClient, archive: Path, settings: Settings
) -> None:
    with Engine.open(settings) as engine:
        for folder in ("Old Mill - March 2021", "Harbor"):
            made = Sessions(engine).label_taste([archive / folder])
            for item in made.items:  # by name: a small location's session is shuffled
                liked = item.path.name in {"view3.jpg", "h11.jpg"}
                engine.ratings.rate_photo(made.set_id, item.content_key, 2 if liked else 0)
        assert Sessions(engine).train_taste(force=True).saved
    job = new_compare_job(client, archive)
    assert job["status"] == "done", job
    detail = client.get(f"/api/label-sets/{job['label_set_id']}").json()
    assert (detail["kind"], detail["text"]) == ("compare", "Compare Strongest: Harbor")
    assert detail["items"]
    assert all(set(i) == {"content_key", "name", "folder", "label"} for i in detail["items"])
    listed = {s["set_id"]: s["kind"] for s in client.get("/api/label-sets").json()}
    assert listed[job["label_set_id"]] == "compare"


def test_a_compare_session_takes_folders_only(client: TestClient, archive: Path) -> None:
    for extra in ({"text": "fog"}, {"taste": True}, {"stories": True}):
        body = {"folders": [str(archive)], "compare": True, **extra}
        assert client.post("/api/label-sets", json=body).status_code == 422


def test_a_search_is_sampled_labelled_and_listed(
    client: TestClient, archive: Path, reads_text: None
) -> None:
    set_id = new_label_set(client, archive)
    (summary,) = client.get("/api/label-sets").json()
    assert summary == summary | {"set_id": set_id, "text": "fog", "total": 6, "labelled": 0}

    detail = client.get(f"/api/label-sets/{set_id}").json()
    assert detail["folders"] == [str(archive.resolve())]
    first, second = detail["items"][:2]
    assert first["label"] is None
    assert first["name"].endswith(".jpg")

    url = f"/api/label-sets/{set_id}/labels/{first['content_key']}"
    assert client.put(url, json={"label": 2}).status_code == 204
    client.put(f"/api/label-sets/{set_id}/labels/{second['content_key']}", json={"label": 0})
    items = client.get(f"/api/label-sets/{set_id}").json()["items"]
    assert [i["label"] for i in items[:2]] == [2, 0]
    assert client.get("/api/label-sets").json()[0]["labelled"] == 2

    assert client.put(url, json={"label": None}).status_code == 204
    assert client.get("/api/label-sets").json()[0]["labelled"] == 1


def test_a_labelled_photo_can_be_seen_full_size_but_only_from_its_set(
    client: TestClient, archive: Path, reads_text: None
) -> None:
    set_id = new_label_set(client, archive)
    key = client.get(f"/api/label-sets/{set_id}").json()["items"][0]["content_key"]
    image = client.get(f"/api/label-sets/{set_id}/image/{key}")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/label-sets/{set_id}/image/{'0' * 32}").status_code == 404


def test_bad_labels_and_unknown_sets_are_refused(
    client: TestClient, archive: Path, reads_text: None
) -> None:
    set_id = new_label_set(client, archive)
    key = client.get(f"/api/label-sets/{set_id}").json()["items"][0]["content_key"]
    assert (
        client.put(f"/api/label-sets/{set_id}/labels/{key}", json={"label": 3}).status_code == 422
    )
    assert client.put(f"/api/label-sets/{set_id}/labels/nope", json={"label": 1}).status_code == 422
    for bad in ("nope", "..%2Fruns", "a b"):
        assert client.get(f"/api/label-sets/{bad}").status_code == 404
    cross_site = client.put(
        f"/api/label-sets/{set_id}/labels/{key}",
        json={"label": 1},
        headers={"Origin": "http://evil.example"},
    )
    assert cross_site.status_code == 403


def test_a_set_can_be_deleted(client: TestClient, archive: Path, reads_text: None) -> None:
    set_id = new_label_set(client, archive)
    assert client.delete(f"/api/label-sets/{set_id}").status_code == 204
    assert client.get(f"/api/label-sets/{set_id}").status_code == 404
    assert client.get("/api/label-sets").json() == []


def test_a_same_view_set_is_built_from_a_reference(client: TestClient, archive: Path) -> None:
    reference = archive / "Old Mill - March 2021" / "view1.jpg"
    created = client.post(
        "/api/label-sets", json={"folders": [str(archive)], "reference": str(reference)}
    )
    assert created.status_code == 200, created.text
    job = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert job["status"] == "done", job
    assert job["step"] == "comparing"  # the last thing it did: matching photos part by part
    assert job["done"] == job["total"]
    set_id = job["label_set_id"]
    detail = client.get(f"/api/label-sets/{set_id}").json()
    assert (detail["kind"], detail["text"]) == ("same_view", "Same view as view1.jpg")
    assert client.get("/api/label-sets").json()[0]["kind"] == "same_view"
    image = client.get(f"/api/label-sets/{set_id}/reference")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"


def test_a_same_view_reference_must_be_in_a_collection_or_uploaded(
    client: TestClient, archive: Path, tmp_path: Path, settings: Settings
) -> None:
    stray = f.save_jpeg(f.scene(3), tmp_path / "elsewhere.jpg")
    body = {"folders": [str(archive)], "reference": str(stray)}
    assert client.post("/api/label-sets", json=body).status_code == 422
    upload = f.save_jpeg(f.scene(3), settings.home / "uploads" / "x" / "ref.jpg")
    body = {"folders": [str(archive)], "reference": str(upload)}
    assert client.post("/api/label-sets", json=body).status_code == 200
    assert client.post("/api/label-sets", json={"folders": [str(archive)]}).status_code == 422


def test_a_description_set_has_no_reference_image(
    client: TestClient, archive: Path, reads_text: None
) -> None:
    set_id = new_label_set(client, archive)
    assert client.get(f"/api/label-sets/{set_id}").json()["kind"] == "describe"
    assert client.get(f"/api/label-sets/{set_id}/reference").status_code == 404


def test_labelling_checks_folders_up_front_and_text_in_the_job(
    client: TestClient, archive: Path, tmp_path: Path
) -> None:
    outside = client.post("/api/label-sets", json={"text": "fog", "folders": [str(tmp_path)]})
    assert outside.status_code == 422
    empty = client.post("/api/label-sets", json={"text": "fog", "folders": []})
    assert empty.status_code == 422
    created = client.post("/api/label-sets", json={"text": "fog", "folders": [str(archive)]})
    job = client.get(f"/api/jobs/{created.json()['job_id']}").json()
    assert job["status"] == "error"
    assert "cannot read text" in job["error"]  # tiny embedder, no fake reader here


# -- reading a description --------------------------------------------------------------------


def test_the_reading_of_a_description_uses_the_places_in_the_chosen_folders(
    client: TestClient, archive: Path
) -> None:
    run_best(client, archive)  # index, so the folders' place names are known
    params = {"text": "fog without people at Harbor in winter", "folders": [str(archive)]}
    reading = client.get("/api/reading", params=params).json()
    assert reading == {
        "used": True,
        "parts": [
            {"kind": "looks for", "text": "fog"},
            {"kind": "without", "text": "people"},
            {"kind": "when", "text": "winter (Dec, Jan, Feb)"},
            {"kind": "where", "text": "Harbor"},
        ],
    }
    plain = client.get("/api/reading", params={"text": "fog"}).json()
    assert plain["parts"] == [{"kind": "looks for", "text": "fog"}]
    golden = client.get("/api/reading", params={"text": "golden hour"}).json()
    assert golden["parts"] == [{"kind": "looks for", "text": "golden hour"}]  # CLIP reads looks


def test_the_reading_shows_measured_looks_when_that_scorer_is_chosen(
    settings: Settings,
) -> None:
    looks = settings.model_copy(update={"criteria_scorer": "parts-looks"})
    client = local(create_app(looks, synchronous_jobs=True))
    golden = client.get("/api/reading", params={"text": "golden hour"}).json()
    assert golden == {
        "used": True,
        "parts": [{"kind": "look", "text": "golden hour (warmer, more saturated)"}],
    }


def test_the_reading_says_when_the_configured_scorer_does_not_use_it(
    settings: Settings,
) -> None:
    baseline = settings.model_copy(update={"criteria_scorer": "baseline"})
    client = local(create_app(baseline, synchronous_jobs=True))
    assert client.get("/api/reading", params={"text": "fog in winter"}).json()["used"] is False


def test_the_reading_refuses_folders_outside_the_collections(
    client: TestClient, tmp_path: Path
) -> None:
    params = {"text": "fog", "folders": [str(tmp_path)]}
    assert client.get("/api/reading", params=params).status_code == 422


@pytest.mark.parametrize(
    ("url", "status"), [("/api/folders?path=a%00b", 422), ("/assets/a%00b", 404)]
)
def test_a_nul_byte_in_a_path_is_refused_not_a_server_error(
    settings: Settings, url: str, status: int
) -> None:
    """Audit A06-3: ``embedded null character`` reached the 500 handler. Pages go through
    the same check as assets (``_inside``)."""
    app = create_app(settings, synchronous_jobs=True)
    response = local(app, raise_server_exceptions=False).get(url)
    assert response.status_code == status


def test_the_generic_error_does_not_say_what_failed() -> None:
    """Audit A06-3: every unexpected 500 said "while selecting photos"."""
    assert "selecting" not in GENERIC_ERROR
    assert "server log" in GENERIC_ERROR
