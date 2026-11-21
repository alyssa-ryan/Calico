"""The web interface in a real Chrome (Playwright): the React app against the real API.

Skipped when Chrome cannot be launched or the UI has not been built (``make ui``).
Set CALICO_SCREENSHOTS=DIR to save screenshots of every page, light and dark.
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
from datetime import UTC, datetime, timedelta
from email import message_from_bytes, policy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx
import pytest
import uvicorn
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright

from calico.catalog.store import Catalog
from calico.labels import LabelItem, LabelSet, LabelStore
from calico.ratings_store import RatingsJournal
from calico.reviews import labels_of, story_label, unfit
from calico.settings import CollectionConfig, Settings
from calico.web.app import create_app
from tests import factories as f

DIST_INDEX = Path(create_app.__code__.co_filename).parent / "static" / "dist" / "index.html"

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser, Page, Request

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(not DIST_INDEX.exists(), reason="UI not built: run make ui"),
]

T0 = datetime(2021, 3, 6, 10, 0, 0)
TIMEOUT_MS = 5000


# -- fixtures ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with sync_playwright() as p:
        try:
            instance = p.chromium.launch(channel="chrome", headless=True)
        except PlaywrightError as exc:
            pytest.skip(f"Chrome cannot be launched: {str(exc).splitlines()[0]}")
        yield instance
        instance.close()


@pytest.fixture(scope="module")
def archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("browser") / "archive"
    seed = 1
    for place in ("Alpha - 2021", "Beta - 2022", "Gamma - 2023"):
        for _ in range(4):
            f.save_jpeg(
                f.scene(seed),
                root / place / f"p{seed:02d}.jpg",
                captured_at=T0 + timedelta(hours=seed * 5),
            )
            seed += 1
    (root / "Alpha - 2021" / "inner").mkdir()
    return root


@pytest.fixture(scope="module")
def base_url(archive: Path) -> Iterator[str]:
    home = archive.parent / "home"
    settings = Settings(
        home=home,
        embedder="tiny",
        describe_embedder="tiny",
        view_embedder="tiny",
        collections=(CollectionConfig(name="archive", root=archive),),
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            if httpx.get(f"{url}/api/health").json()["engine"] == "ready":
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    else:
        pytest.fail("server did not become ready")
    yield url
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def page(browser: Browser) -> Iterator[Page]:
    context = browser.new_context()
    context.set_default_timeout(TIMEOUT_MS)
    yield context.new_page()
    context.close()


def make_run(base_url: str, archive: Path, task: str, **extra: Any) -> str:
    body = {"task": task, "folders": [str(archive)], "limit": 3, **extra}
    response = httpx.post(f"{base_url}/api/select", data={"request": json.dumps(body)})
    response.raise_for_status()
    job_id = response.json()["job_id"]
    for _ in range(200):
        job = httpx.get(f"{base_url}/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            assert job["status"] == "done", job
            return str(job["run_id"])
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def make_view_label_set(base_url: str, archive: Path) -> str:
    """A same-view labelling set, made through the API as "Label for calibration" makes it."""
    reference = str(archive / "Alpha - 2021" / "p01.jpg")
    return new_label_job(base_url, {"folders": [str(archive)], "reference": reference})


def make_story_label_set(base_url: str, archive: Path) -> str:
    """A story session, made through the API as *Rate stories* makes it."""
    return new_label_job(base_url, {"folders": [str(archive)], "stories": True})


def new_label_job(base_url: str, body: dict[str, Any]) -> str:
    response = httpx.post(f"{base_url}/api/label-sets", json=body)
    response.raise_for_status()
    job_id = response.json()["job_id"]
    for _ in range(200):
        job = httpx.get(f"{base_url}/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            assert job["status"] == "done", job
            return str(job["label_set_id"])
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def make_label_set(base_url: str, archive: Path, text: str = "stairs") -> str:
    """A labelling set over six indexed photos, written as the app would write it.

    The browser tests run the ``tiny`` embedder, which cannot read text to sample one.
    """
    make_run(base_url, archive, "best")  # indexes the archive and writes its thumbnails
    home = archive.parent / "home"
    with Catalog(home / "catalog.sqlite") as catalog:
        photos = catalog.photos_under([archive])[:6]
    store = LabelStore(home / "labels")
    now = datetime.now(UTC)
    made = LabelSet(
        set_id=store.new_id(now),
        created_at=now,
        text=text,
        folders=(archive,),
        scorer="baseline@1",
        items=tuple(LabelItem(content_key=p.content_key, path=p.path) for p in photos),
    )
    store.save(made)
    return made.set_id


def screenshot(page: Page, name: str) -> None:
    target = os.environ.get("CALICO_SCREENSHOTS")
    if target:
        Path(target).mkdir(parents=True, exist_ok=True)
        page.mouse.move(0, 0)  # no hover effects in the picture
        for img in page.locator("img[loading=lazy]").all():
            img.scroll_into_view_if_needed()
        page.wait_for_function("[...document.images].every(i => i.complete)")
        page.wait_for_timeout(600)  # thumbnails fade in
        page.evaluate("window.scrollTo(0, 0)")
        page.screenshot(path=str(Path(target) / f"{name}.png"), full_page=True)


EVERY_PHOTO_MATCHES = {"sure_share": 0.0, "possible_share": 0.0, "shortlist_similarity": 0.0}
"""Same-view options under which every photo counts as the same view: the synthetic
scenes are unalike, and these tests are about the page, not the matching."""


def posted_request(request: Request) -> dict[str, Any]:
    """The JSON ``request`` field of a multipart POST."""
    header = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    message = message_from_bytes(header + (request.post_data_buffer or b""), policy=policy.HTTP)
    for part in message.iter_parts():
        if part.get_param("name", header="content-disposition") == "request":
            return dict(json.loads(str(part.get_content())))
    raise AssertionError("no request field")


def open_composer(page: Page, base_url: str) -> None:
    page.goto(base_url)
    page.get_by_test_id("task-card").first.wait_for()


def tick_root(page: Page) -> None:
    page.get_by_role("checkbox", name="archive").check()


def find(page: Page) -> None:
    page.get_by_role("button", name="Find photographs").click()
    page.wait_for_url("**/runs/*")


def assert_no_horizontal_overflow(page: Page) -> None:
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 0, f"page is {overflow}px wider than the viewport"


# -- composer -----------------------------------------------------------------------------


def test_reference_zone_follows_the_task(page: Page, base_url: str) -> None:
    open_composer(page, base_url)
    assert page.get_by_test_id("task-card").count() == 4
    assert page.get_by_test_id("reference-drop").count() == 0

    page.get_by_text("Same view over time", exact=True).click()
    assert page.get_by_role("heading", name="Reference photo").is_visible()
    assert page.get_by_text("required", exact=True).is_visible()

    page.get_by_text("Three-photo stories", exact=True).click()
    assert page.get_by_role("heading", name="Anchor photo").is_visible()
    assert page.get_by_text("optional", exact=True).is_visible()

    page.get_by_text("Views you keep coming back to", exact=True).click()
    assert page.get_by_test_id("reference-drop").count() == 0
    screenshot(page, "composer-desktop")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_artwork_frames_the_page_without_hiding_the_controls(
    browser: Browser, base_url: str, theme: Literal["light", "dark"]
) -> None:
    context = browser.new_context(viewport={"width": 1280, "height": 800}, color_scheme=theme)
    context.set_default_timeout(TIMEOUT_MS)
    page = context.new_page()
    try:
        open_composer(page, base_url)
        page.get_by_text("Fine-tune").click()
        page.wait_for_function("[...document.images].every(i => i.complete)")
        for art in (
            page.get_by_test_id("masthead").locator("img"),
            page.get_by_role("contentinfo").locator("img"),
        ):
            assert art.evaluate("i => i.naturalWidth") > 0, "artwork failed to load"

        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        bar = page.get_by_role("button", name="Find photographs").bounding_box()
        footer = page.get_by_role("contentinfo").bounding_box()
        assert bar is not None
        assert footer is not None
        assert bar["y"] + bar["height"] <= footer["y"], "the action bar covers the footer art"
        assert page.get_by_role("navigation", name="Main").is_visible()  # nav stays put
        screenshot(page, f"artwork-{theme}")
    finally:
        context.close()


def test_a_page_built_with_the_server_asks_for_no_restart(page: Page, base_url: str) -> None:
    open_composer(page, base_url)
    expect(page.get_by_text("Ready", exact=True)).to_be_visible()
    expect(page.get_by_role("alert").filter(has_text="restart calico serve")).to_have_count(0)


def test_describe_it_shows_how_it_reads_the_description(page: Page, base_url: str) -> None:
    open_composer(page, base_url)
    page.get_by_label("What are you looking for?").fill("fog without people in winter")
    reading = page.get_by_test_id("reading")
    expect(reading).to_contain_text("without people")
    expect(reading).to_contain_text("winter (Dec, Jan, Feb)")
    screenshot(page, "composer-reading")


def test_only_clear_matches_is_offered_for_a_description_and_sent(
    page: Page, base_url: str
) -> None:
    open_composer(page, base_url)
    tick_root(page)
    switch = page.get_by_role("checkbox", name="Only clear matches")
    expect(switch).to_have_count(0)
    page.get_by_label("What are you looking for?").fill("a staircase")
    switch.check()
    with page.expect_request("**/api/select") as posted:
        find(page)
    assert posted_request(posted.value)["only_clear_matches"] is True
    expect(page.get_by_text("only clear matches", exact=True)).to_be_visible()
    expect(page.get_by_role("note").filter(has_text="Only clear matches")).to_be_visible()


def test_how_much_the_description_counts_is_set_by_keyboard_and_sent(
    page: Page, base_url: str
) -> None:
    open_composer(page, base_url)
    tick_root(page)
    slider = page.get_by_role("slider", name="How much the description counts")
    expect(slider).to_have_count(0)
    page.get_by_label("What are you looking for?").fill("a staircase")
    expect(slider).to_have_value("0.5")
    slider.focus()
    page.keyboard.press("End")
    expect(page.get_by_test_id("describe-weight")).to_have_text("100%")
    screenshot(page, "composer-describe-weight")
    with page.expect_request("**/api/select") as posted:
        find(page)
    assert posted_request(posted.value)["describe_weight"] == 1
    expect(page).to_have_url(re.compile(r"/runs/"))
    expect(page.get_by_text("counts 100%", exact=True)).to_be_visible()


def test_photos_that_clearly_match_are_badged(page: Page, base_url: str, archive: Path) -> None:
    """The tiny embedder cannot tell a clear match, so mark one in the saved run."""
    run_id = make_run(base_url, archive, "best", criteria="a staircase")
    saved = archive.parent / "home" / "runs" / run_id / "result.json"
    result = json.loads(saved.read_text(encoding="utf-8"))
    result["groups"][0]["picks"][0]["clear_match"] = True
    saved.write_text(json.dumps(result), encoding="utf-8")
    page.goto(f"{base_url}/runs/{run_id}")
    cards = page.get_by_test_id("pick-card")
    expect(cards.first.get_by_text("Clear match")).to_be_visible()
    expect(cards.nth(1).get_by_text("Clear match")).to_have_count(0)
    screenshot(page, "results-clear-match")


def test_ticking_a_parent_covers_its_children_and_only_the_parent_is_sent(
    page: Page, base_url: str, archive: Path
) -> None:
    open_composer(page, base_url)
    page.get_by_role("button", name="Show subfolders of archive").click()
    child = page.get_by_role("checkbox", name="Alpha - 2021")
    child.wait_for()
    tick_root(page)
    assert child.is_checked()
    assert child.is_disabled()
    assert page.get_by_role("button", name="Hide subfolders of archive").is_visible()

    with page.expect_request("**/api/select") as posted:
        page.get_by_role("button", name="Find photographs").click()
    assert posted_request(posted.value)["folders"] == [str(archive)]


def test_a_folder_search_ticks_every_match_with_one_click_and_sends_them(
    page: Page, base_url: str, archive: Path
) -> None:
    open_composer(page, base_url)
    page.get_by_role("searchbox").fill("MA - 20")  # Gamma, not Alpha or Beta; any case
    page.get_by_text("1 folder match “MA - 20”").wait_for()
    page.get_by_role("searchbox").fill("pha")
    page.get_by_text("1 folder match “pha”").wait_for()
    page.get_by_role("button", name="Select all 1").click()
    page.get_by_role("searchbox").fill("- 202")
    page.get_by_text("3 folders match “- 202”").wait_for()
    screenshot(page, "composer-folder-search")
    page.get_by_role("button", name="Select all 3").click()
    assert page.get_by_role("checkbox", name="Gamma - 2023").is_checked()

    with page.expect_request("**/api/select") as posted:
        page.get_by_role("button", name="Find photographs").click()
    sent = posted_request(posted.value)["folders"]
    assert sent == [str(archive / p) for p in ("Alpha - 2021", "Beta - 2022", "Gamma - 2023")]


def test_fine_tune_controls_come_from_the_task_and_block_invalid_choices(
    page: Page, base_url: str
) -> None:
    open_composer(page, base_url)
    page.get_by_text("Three-photo stories", exact=True).click()
    tick_root(page)
    page.get_by_text("Fine-tune").click()
    for label in ("Theme", "Place type", "Mood", "Color"):
        if page.get_by_role("checkbox", name=label, exact=True).is_checked():
            page.locator("label").filter(has_text=re.compile(rf"^{label}$")).click()
    assert page.get_by_text("Pick at least one.", exact=True).is_visible()
    assert "Pick at least one." in page.get_by_test_id("summary").inner_text()
    assert page.get_by_role("button", name="Find photographs").is_disabled()


def test_strongest_photographs_end_to_end(page: Page, base_url: str) -> None:
    open_composer(page, base_url)
    tick_root(page)
    find(page)
    cards = page.get_by_test_id("pick-card")
    cards.first.wait_for()
    assert cards.count() > 0
    images = "[...document.querySelectorAll('[data-testid=pick-card] img')]"
    page.wait_for_function(f"{images}.every(i => i.naturalWidth > 0)")
    screenshot(page, "results-best")


def test_job_errors_are_shown_and_polling_stops(page: Page, base_url: str, archive: Path) -> None:
    open_composer(page, base_url)
    page.get_by_text("Same view over time", exact=True).click()
    tick_root(page)
    page.get_by_label("Add reference photo").set_input_files(
        {"name": "not-an-image.jpg", "mimeType": "image/jpeg", "buffer": b"not a jpeg"}
    )
    page.get_by_role("button", name="Find photographs").click()
    alert = page.get_by_role("alert")
    alert.wait_for()
    assert "not-an-image" in alert.inner_text()
    assert page.url.rstrip("/") == base_url


# -- results ------------------------------------------------------------------------------


def test_feedback_persists_and_lightbox_keyboard(page: Page, base_url: str, archive: Path) -> None:
    run_id = make_run(base_url, archive, "best")
    page.goto(f"{base_url}/runs/{run_id}")
    keep = page.get_by_role("button", name="Keep").first
    keep.click()
    page.wait_for_function("document.querySelector('[aria-pressed=true]') !== null")
    page.wait_for_timeout(200)  # let the optimistic POST land before reloading
    page.reload()
    assert page.get_by_role("button", name="Keep").first.get_attribute("aria-pressed") == "true"

    opener = page.get_by_role("button", name="Open").first
    opener.click()
    lightbox = page.get_by_test_id("lightbox")
    first = lightbox.get_attribute("aria-label")
    page.keyboard.press("ArrowRight")
    assert lightbox.get_attribute("aria-label") != first
    page.keyboard.press("Escape")
    assert lightbox.count() == 0
    assert page.evaluate("document.activeElement.getAttribute('aria-label')").startswith("Open ")


def test_stories_mark_one_anchor_each_and_list_connections(
    page: Page, base_url: str, archive: Path
) -> None:
    run_id = make_run(base_url, archive, "triptych", options={"connections": ["theme", "color"]})
    page.goto(f"{base_url}/runs/{run_id}")
    stories = page.get_by_test_id("story")
    stories.first.wait_for()
    for i in range(stories.count()):
        story = stories.nth(i)
        assert story.get_by_test_id("anchor-badge").count() == 1
        assert story.get_by_test_id("pick-card").count() == 3
        assert "theme" in story.get_by_label("How these photos connect").inner_text()
    screenshot(page, "results-stories")


def test_build_a_story_around_a_result(page: Page, base_url: str, archive: Path) -> None:
    run_id = make_run(base_url, archive, "best")
    page.goto(f"{base_url}/runs/{run_id}")
    first_card = page.get_by_test_id("pick-card").first
    name = first_card.get_by_role("button", name="Open").get_attribute("aria-label")
    assert name is not None
    anchor = name.removeprefix("Open ")
    first_card.get_by_role("button", name="More actions").click()
    page.get_by_role("menuitem", name="Build a story around this").click()

    page.get_by_role("heading", name="Anchor photo").wait_for()
    assert anchor in page.get_by_test_id("summary").inner_text()
    find(page)
    stories = page.get_by_test_id("story")
    stories.first.wait_for()
    for i in range(stories.count()):
        middle = stories.nth(i).get_by_test_id("pick-card").nth(1)
        assert middle.get_by_role("button", name=f"Open {anchor}").count() == 1


def test_more_like_a_result_describes_the_next_search_by_it(
    page: Page, base_url: str, archive: Path
) -> None:
    run_id = make_run(base_url, archive, "best")
    page.goto(f"{base_url}/runs/{run_id}")
    first_card = page.get_by_test_id("pick-card").first
    name = first_card.get_by_role("button", name="Open").get_attribute("aria-label")
    assert name is not None
    example = name.removeprefix("Open ")
    first_card.get_by_role("button", name="More actions").click()
    page.get_by_role("menuitem", name="More like this").click()

    examples = page.get_by_role("list", name="Like these photos")
    expect(examples.get_by_role("img", name=example)).to_be_visible()
    expect(page.get_by_test_id("summary")).to_contain_text("like 1 example")
    screenshot(page, "composer-examples")
    with page.expect_request("**/api/select") as posted:
        find(page)
    assert posted_request(posted.value)["examples"][0].endswith(example)
    expect(page).to_have_url(re.compile(r"/runs/"))
    expect(page.get_by_role("list", name="Like these photos")).to_be_visible()
    expect(page.get_by_role("button", name=f"Open {example}")).to_have_count(0)
    expect(page.get_by_role("note").filter(has_text="left out of the results")).to_be_visible()
    screenshot(page, "results-examples")


def test_same_view_timeline_starts_with_the_reference(
    page: Page, base_url: str, archive: Path
) -> None:
    run_id = make_run(
        base_url,
        archive,
        "same_view",
        references=[str(archive / "Alpha - 2021" / "p01.jpg")],
        options=EVERY_PHOTO_MATCHES,
    )
    page.goto(f"{base_url}/runs/{run_id}")
    page.get_by_alt_text("Your reference").wait_for()
    assert page.get_by_test_id("timeline-month").count() >= 1
    screenshot(page, "results-timeline")


def test_photos_that_are_possibly_the_same_view_follow_the_timeline(
    page: Page, base_url: str, archive: Path
) -> None:
    run_id = make_run(
        base_url,
        archive,
        "same_view",
        references=[str(archive / "Alpha - 2021" / "p01.jpg")],
        options={**EVERY_PHOTO_MATCHES, "sure_share": 1.0},
    )
    page.goto(f"{base_url}/runs/{run_id}")
    possible = page.get_by_role("region", name="Possibly the same view")
    possible.wait_for()
    assert possible.get_by_test_id("pick-card").count() >= 1
    assert page.get_by_role("button", name="Label for calibration").is_visible()
    screenshot(page, "results-possible")


def test_a_same_view_run_is_labelled_for_calibration(
    page: Page, base_url: str, archive: Path
) -> None:
    run_id = make_run(
        base_url,
        archive,
        "same_view",
        references=[str(archive / "Alpha - 2021" / "p01.jpg")],
        options=EVERY_PHOTO_MATCHES,
    )
    page.goto(f"{base_url}/runs/{run_id}")
    page.get_by_role("button", name="Label for calibration").click()
    expect(page.get_by_role("heading", level=1)).to_have_text("Same view as p01.jpg")
    expect(page.get_by_alt_text("The reference view")).to_be_visible()
    page.get_by_test_id("label-card").first.get_by_role("button", name="Enlarge").focus()
    page.keyboard.press("1")
    expect(page.get_by_test_id("label-progress")).to_contain_text("1 of")
    expect(page.get_by_role("button", name="Same view:").first).to_be_visible()
    screenshot(page, "label-same-view")


def test_label_a_search_with_the_keyboard(page: Page, base_url: str, archive: Path) -> None:
    set_id = make_label_set(base_url, archive)
    page.goto(f"{base_url}/label/{set_id}")
    page.get_by_test_id("label-card").first.get_by_role("button", name="Enlarge").focus()
    for key in ("1", "2", "3"):
        page.keyboard.press(key)
    progress = page.get_by_test_id("label-progress")
    expect(progress).to_have_text("3 of 6 labelled")
    page.wait_for_function(
        f"fetch('/api/label-sets/{set_id}').then(r => r.json())"
        ".then(s => s.items.filter(i => i.label !== null).length === 3)"
    )
    page.reload()
    expect(progress).to_have_text("3 of 6 labelled")
    home = archive.parent / "home"
    made = LabelStore(home / "labels").load(set_id)
    given = labels_of(made, RatingsJournal(home / "ratings").ratings())
    assert sorted(given.values()) == [0, 1, 2]
    screenshot(page, "label-set")

    page.get_by_role("link", name="Label", exact=True).click()
    expect(page.get_by_test_id("label-set-card").first).to_contain_text("labelled")
    screenshot(page, "labels")


def test_rate_stories_with_the_keyboard(page: Page, base_url: str, archive: Path) -> None:
    set_id = make_story_label_set(base_url, archive)
    total = len(LabelStore(archive.parent / "home" / "labels").load(set_id).stories)
    assert total >= 2
    page.goto(f"{base_url}/label/{set_id}")
    page.get_by_role("group", name=f"Story 1 of {total}").focus()
    page.keyboard.press("1")
    expect(page.get_by_role("group", name=f"Story 2 of {total}")).to_be_focused()
    page.keyboard.press("3")
    first = page.get_by_test_id("story-card").first
    first.get_by_role("button", name="Doesn't fit:").nth(1).click()
    progress = page.get_by_test_id("label-progress")
    expect(progress).to_have_text(f"2 of {total} labelled")
    page.wait_for_function(
        f"fetch('/api/label-sets/{set_id}').then(r => r.json())"
        ".then(s => !s.stories[0].photos[1].fits)"
    )
    page.reload()
    expect(progress).to_have_text(f"2 of {total} labelled")
    home = archive.parent / "home"
    stories = LabelStore(home / "labels").load(set_id).stories
    current = RatingsJournal(home / "ratings").ratings()
    assert [story_label(s, current) for s in stories[:2]] == [2, 0]
    assert unfit(stories[0], current) == {stories[0].photos[1].content_key}
    screenshot(page, "label-stories")


def test_compare_strongest_says_how_to_train_yours_first(
    page: Page, base_url: str, archive: Path
) -> None:
    page.goto(f"{base_url}/label")
    form = page.get_by_role("form", name="Compare Strongest")
    start = form.get_by_role("button", name="Compare first picks")
    expect(start).to_be_disabled()
    form.get_by_role("checkbox", name="archive").check()
    start.click()
    expect(page.get_by_role("alert")).to_contain_text("calico train taste")
    screenshot(page, "labels-compare")


def test_past_runs_lists_runs(page: Page, base_url: str, archive: Path) -> None:
    make_run(base_url, archive, "best")
    page.goto(f"{base_url}/runs")
    page.get_by_test_id("run-card").first.wait_for()
    assert page.get_by_test_id("run-card").count() >= 1


# -- every page, phone width, both themes ---------------------------------------------------


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_no_horizontal_overflow_on_a_phone(
    browser: Browser, base_url: str, archive: Path, theme: Literal["light", "dark"]
) -> None:
    best = make_run(base_url, archive, "best")
    stories = make_run(base_url, archive, "triptych")
    timeline = make_run(
        base_url,
        archive,
        "same_view",
        references=[str(archive / "Alpha - 2021" / "p01.jpg")],
        options=EVERY_PHOTO_MATCHES,
    )
    label_set = make_label_set(base_url, archive)
    view_set = make_view_label_set(base_url, archive)
    story_set = make_story_label_set(base_url, archive)
    context = browser.new_context(viewport={"width": 390, "height": 844}, color_scheme=theme)
    context.set_default_timeout(TIMEOUT_MS)
    page = context.new_page()
    try:
        for name, path in [
            ("composer", "/"),
            ("best", f"/runs/{best}"),
            ("stories", f"/runs/{stories}"),
            ("timeline", f"/runs/{timeline}"),
            ("runs", "/runs"),
            ("labels", "/label"),
            ("label-set", f"/label/{label_set}"),
            ("label-same-view", f"/label/{view_set}"),
            ("label-stories", f"/label/{story_set}"),
        ]:
            page.goto(base_url + path)
            page.wait_for_load_state("networkidle")
            if name == "composer":
                page.get_by_label("What are you looking for?").fill("a staircase")
                page.get_by_text("Fine-tune").click()
            assert_no_horizontal_overflow(page)
            screenshot(page, f"mobile-{theme}-{name}")
    finally:
        context.close()
