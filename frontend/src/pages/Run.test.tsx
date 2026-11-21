import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { POSSIBLY_SAME, SAME_VIEW } from "../lib/format";
import { pick, renderAt, runDetail, testClient } from "../test/render";
import { Run } from "./Run";

vi.mock("../api/client", async (original) => {
  const actual = await original<typeof import("../api/client")>();
  return {
    ...actual,
    api: { run: vi.fn(), feedback: vi.fn(), newLabelSet: vi.fn(), job: vi.fn() },
  };
});
const mocked = vi.mocked(api);

const show = (client = testClient()) =>
  renderAt(<Run />, { path: "/runs/r1", route: "/runs/:runId", client });

beforeEach(() => vi.resetAllMocks());

describe("Run page", () => {
  it("lays out stories as rows of three with the anchor marked and connections listed", async () => {
    const story = (n: number) => ({
      title: `Story ${n} with b.jpg`,
      score: 0.7,
      picks: [pick(`a${n}.jpg`), pick(`b${n}.jpg`), pick(`c${n}.jpg`)],
      evidence: [
        { signal: "strength", value: 0.6, weight: 0.4, detail: "" },
        { signal: "theme", value: 0.82, weight: 0.3, detail: "" },
        { signal: "color", value: 0.71, weight: 0.3, detail: "" },
      ],
    });
    mocked.run.mockResolvedValue(runDetail("triptych", [story(1), story(2)]));
    show();
    const stories = await screen.findAllByTestId("story");
    expect(stories).toHaveLength(2);
    for (const s of stories) {
      expect(within(s).getAllByTestId("anchor-badge")).toHaveLength(1);
      expect(within(s).getByText("theme 82%")).toBeInTheDocument();
    }
  });

  it("shows same-view matches as a timeline after the reference", async () => {
    const group = {
      title: "Same view as ref.jpg",
      score: 0.9,
      evidence: [],
      picks: [
        pick("x.jpg", { captured_at: "2021-03-08T09:00:00" }),
        pick("y.jpg", { captured_at: "2024-05-01T09:00:00" }),
      ],
    };
    const detail = runDetail("same_view", [group], { reference_label: "Reference photo" });
    detail.result.request.references = ["/uploads/ref.jpg"];
    mocked.run.mockResolvedValue(detail);
    show();
    expect(await screen.findByAltText("Your reference")).toBeInTheDocument();
    expect(
      screen.getAllByTestId("timeline-month").map((m) => m.querySelector("h2")?.textContent),
    ).toEqual(["March 2021", "May 2024"]);
  });

  it("marks the photos that clearly match the description", async () => {
    const detail = runDetail("best", [
      {
        title: "g",
        score: 1,
        evidence: [],
        picks: [pick("a.jpg", { clear_match: true }), pick("b.jpg", { clear_match: false })],
      },
    ]);
    detail.result.request.criteria = "a staircase";
    detail.result.request.only_clear_matches = true;
    detail.result.request.describe_weight = 0.4;
    mocked.run.mockResolvedValue(detail);
    show();
    const [first, second] = await screen.findAllByTestId("pick-card");
    expect(within(first as HTMLElement).getByText("Clear match")).toBeInTheDocument();
    expect(within(second as HTMLElement).queryByText("Clear match")).toBeNull();
    expect(screen.getByText("only clear matches")).toBeInTheDocument();
    expect(screen.getByText("counts 40%")).toBeInTheDocument();
  });

  it("says how likely you are to like each pick and draws the line where the list stops being good", async () => {
    const detail = runDetail("best", [
      {
        title: "g",
        score: 1,
        evidence: [],
        marker: { before: 2, probability: 0.8 },
        picks: [
          pick("a.jpg", { probability: 0.93 }),
          pick("b.jpg", { probability: 0.84 }),
          pick("c.jpg", { probability: 0.61 }),
          pick("d.jpg", { probability: 0.42 }),
        ],
      },
    ]);
    mocked.run.mockResolvedValue(detail);
    show();
    const cards = await screen.findAllByTestId("pick-card");
    expect(within(cards[0] as HTMLElement).getByText("93% likely")).toBeInTheDocument();
    expect(within(cards[3] as HTMLElement).getByText("42% likely")).toBeInTheDocument();
    const line = screen.getByTestId("marker");
    expect(line).toHaveTextContent(
      "Below this line a pick is less than 80% likely to be one you like",
    );
    const row = [...(line.parentElement as HTMLElement).children];
    expect(row.indexOf(line)).toBe(2); // after the second pick, before the third
  });

  it("says so when every pick is above the line, and shows no line", async () => {
    const detail = runDetail("best", [
      {
        title: "g",
        score: 1,
        evidence: [],
        marker: { before: 2, probability: 0.8 },
        picks: [pick("a.jpg", { probability: 0.93 }), pick("b.jpg", { probability: 0.84 })],
      },
    ]);
    mocked.run.mockResolvedValue(detail);
    show();
    await screen.findAllByTestId("pick-card");
    expect(screen.queryByTestId("marker")).toBeNull();
    expect(
      screen.getByText("Every pick here is at least 80% likely to be one you like"),
    ).toBeInTheDocument();
  });

  it("shows no probability and no line for a taste that cannot say", async () => {
    const detail = runDetail("best", [
      {
        title: "g",
        score: 1,
        evidence: [],
        picks: [pick("a.jpg"), pick("b.jpg", { probability: null })],
      },
    ]);
    mocked.run.mockResolvedValue(detail);
    show();
    await screen.findAllByTestId("pick-card");
    expect(screen.queryByText(/likely/)).toBeNull();
    expect(screen.queryByTestId("marker")).toBeNull();
  });

  it("says when your taste ranked a run", async () => {
    const detail = runDetail("best", [
      { title: "g", score: 1, evidence: [], picks: [pick("a.jpg")] },
    ]);
    detail.result.request.template = "yours";
    mocked.run.mockResolvedValue(detail);
    show();
    expect(await screen.findByText(/· your taste/)).toBeInTheDocument();
  });

  it("shows the examples a run was described by, and offers more like each pick", async () => {
    const user = userEvent.setup();
    const detail = runDetail("best", [
      { title: "g", score: 1, evidence: [], picks: [pick("a.jpg")] },
    ]);
    detail.result.request.examples = ["/archive/Old Mill/e.jpg"];
    detail.result.request.describe_weight = 0.5;
    detail.result.example_keys = ["e"];
    mocked.run.mockResolvedValue(detail);
    show();
    const list = await screen.findByRole("list", { name: "Like these photos" });
    expect(within(list).getByAltText("e.jpg")).toBeInTheDocument();
    expect(screen.getByText("counts 50%")).toBeInTheDocument(); // no words, still counted
    await user.click(screen.getByRole("button", { name: "More actions for a.jpg" }));
    await user.click(screen.getByRole("menuitem", { name: "More like this" }));
    expect(await screen.findByText("navigated away")).toBeInTheDocument();
  });

  it("follows the timeline with the photos that are possibly the same view", async () => {
    const detail = runDetail("same_view", [
      { title: "Same view as ref.jpg", score: 0.4, evidence: [], picks: [pick("sure.jpg")] },
      {
        title: "Possibly the same view",
        score: 0.2,
        evidence: [],
        picks: [pick("maybe1.jpg"), pick("maybe2.jpg")],
      },
    ]);
    detail.result.request.references = ["/uploads/ref.jpg"];
    mocked.run.mockResolvedValue(detail);
    show();
    const possible = await screen.findByRole("region", { name: "Possibly the same view" });
    expect(within(possible).getAllByTestId("pick-card")).toHaveLength(2);
    expect(within(possible).getByText(POSSIBLY_SAME.hint)).toBeInTheDocument();
    const [month, ...others] = screen.getAllByTestId("timeline-month");
    expect(others).toHaveLength(0);
    expect(within(month as HTMLElement).getAllByTestId("pick-card")).toHaveLength(1);
  });

  it("says up front how many photos are in each tier and links to the second", async () => {
    const detail = runDetail("same_view", [
      {
        title: "Same view as ref.jpg",
        score: 0.4,
        evidence: [],
        picks: [pick("a.jpg"), pick("b.jpg")],
      },
      { title: "Possibly the same view", score: 0.2, evidence: [], picks: [pick("c.jpg")] },
    ]);
    detail.result.request.references = ["/uploads/ref.jpg"];
    mocked.run.mockResolvedValue(detail);
    show();
    expect(await screen.findByText(SAME_VIEW.summary(2, 1))).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: SAME_VIEW.sureHeading(2) })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Possibly the same view · 1" })).toBeInTheDocument();
    const jump = screen.getByRole("link", { name: SAME_VIEW.jump });
    expect(jump).toHaveAttribute("href", "#possibly-the-same-view");
    expect(document.getElementById("possibly-the-same-view")).not.toBeNull();
  });

  it("still offers calibration when nothing matched the reference", async () => {
    const detail = runDetail("same_view", [
      { title: "Same view as ref.jpg", score: 0, evidence: [], picks: [] },
    ]);
    detail.result.request.references = ["/uploads/ref.jpg"];
    mocked.run.mockResolvedValue(detail);
    show();
    expect(await screen.findByAltText("Your reference")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /label for calibration/i })).toBeInTheDocument();
    expect(screen.getByText(/no photographs matched/i)).toBeInTheDocument();
  });

  it("labels a same-view search for calibration from its run", async () => {
    const user = userEvent.setup();
    const detail = runDetail("same_view", [
      { title: "g", score: 1, evidence: [], picks: [pick("a.jpg")] },
    ]);
    detail.result.request.references = ["/uploads/ref.jpg"];
    mocked.run.mockResolvedValue(detail);
    mocked.newLabelSet.mockResolvedValue({ job_id: "j1" });
    mocked.job.mockResolvedValue({
      status: "done",
      done: 1,
      total: 1,
      step: "comparing",
      run_id: null,
      label_set_id: "s1",
      error: null,
    });
    show();
    await user.click(await screen.findByRole("button", { name: /label for calibration/i }));
    expect(mocked.newLabelSet).toHaveBeenCalledWith("", ["/archive"], "/uploads/ref.jpg");
    expect(await screen.findByText("navigated away")).toBeInTheDocument();
  });

  it("shows how far calibration labelling has got", async () => {
    const user = userEvent.setup();
    const detail = runDetail("same_view", [
      { title: "g", score: 1, evidence: [], picks: [pick("a.jpg")] },
    ]);
    detail.result.request.references = ["/uploads/ref.jpg"];
    mocked.run.mockResolvedValue(detail);
    mocked.newLabelSet.mockResolvedValue({ job_id: "j1" });
    mocked.job.mockResolvedValue({
      status: "running",
      done: 240,
      total: 600,
      step: "comparing",
      run_id: null,
      label_set_id: null,
      error: null,
    });
    show();
    await user.click(await screen.findByRole("button", { name: /label for calibration/i }));
    expect(await screen.findByRole("status")).toHaveTextContent("Comparing details: 240 of 600");
  });

  it("offers calibration only on same-view runs", async () => {
    mocked.run.mockResolvedValue(
      runDetail("best", [{ title: "g", score: 1, evidence: [], picks: [pick("a.jpg")] }]),
    );
    show();
    await screen.findAllByTestId("pick-card");
    expect(screen.queryByRole("button", { name: /label for calibration/i })).toBeNull();
  });

  it("offers no more-like-this on a same-view timeline", async () => {
    const user = userEvent.setup();
    const detail = runDetail("same_view", [
      { title: "g", score: 1, evidence: [], picks: [pick("a.jpg")] },
    ]);
    detail.result.request.references = ["/uploads/ref.jpg"];
    mocked.run.mockResolvedValue(detail);
    show();
    await user.click(await screen.findByRole("button", { name: "More actions for a.jpg" }));
    expect(screen.queryByRole("menuitem", { name: "More like this" })).toBeNull();
  });

  it("surfaces every note", async () => {
    const detail = runDetail("best", [
      { title: "g", score: 1, evidence: [], picks: [pick("a.jpg")] },
    ]);
    detail.result.notes = ["The criteria were ignored."];
    mocked.run.mockResolvedValue(detail);
    show();
    expect(await screen.findByRole("note")).toHaveTextContent("The criteria were ignored.");
  });

  it("restores saved ratings and reverts a rating the server refuses", async () => {
    const user = userEvent.setup();
    const detail = runDetail(
      "best",
      [{ title: "g", score: 1, evidence: [], picks: [pick("a.jpg"), pick("b.jpg")] }],
      {
        feedback: { "a.jpg": 1 },
      },
    );
    mocked.run.mockResolvedValue(detail);
    mocked.feedback.mockRejectedValue(new Error("nope"));
    show();
    expect(await screen.findByRole("button", { name: "Keep a.jpg" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    const reject = screen.getByRole("button", { name: "Reject b.jpg" });
    await user.click(reject);
    await waitFor(() => expect(reject).toHaveAttribute("aria-pressed", "false"));
    expect(mocked.feedback).toHaveBeenCalledWith("r1", "b.jpg", -1);
    expect(screen.getByRole("alert")).toHaveTextContent("nope"); // not silently
  });

  it("shows ratings given earlier when you come back to the run", async () => {
    const user = userEvent.setup();
    const picks = [pick("a.jpg"), pick("b.jpg")];
    mocked.run.mockResolvedValue(
      runDetail("best", [{ title: "g", score: 1, evidence: [], picks }]),
    );
    mocked.feedback.mockResolvedValue(undefined);
    const client = testClient();
    const first = show(client);
    await user.click(await screen.findByRole("button", { name: "Keep b.jpg" }));
    first.unmount();

    show(client);
    expect(await screen.findByRole("button", { name: "Keep b.jpg" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(mocked.run).toHaveBeenCalledTimes(1); // the cache answered, as in the app
  });

  it("opens the lightbox, moves with the arrow keys and closes with Escape", async () => {
    const user = userEvent.setup();
    mocked.run.mockResolvedValue(
      runDetail("best", [
        { title: "g", score: 1, evidence: [], picks: [pick("a.jpg"), pick("b.jpg")] },
      ]),
    );
    show();
    const opener = await screen.findByRole("button", { name: "Open a.jpg" });
    await user.click(opener);
    expect(screen.getByTestId("lightbox")).toHaveAccessibleName("a.jpg");
    await user.keyboard("{ArrowRight}");
    expect(screen.getByTestId("lightbox")).toHaveAccessibleName("b.jpg");
    await user.keyboard("{Escape}");
    expect(screen.queryByTestId("lightbox")).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });
});
