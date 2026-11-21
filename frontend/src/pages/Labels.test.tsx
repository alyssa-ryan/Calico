import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { renderAt } from "../test/render";
import { Labels } from "./Labels";

vi.mock("../api/client", async (original) => {
  const actual = await original<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      labelSets: vi.fn(),
      roots: vi.fn(),
      children: vi.fn(),
      newLabelSet: vi.fn(),
      newTasteSet: vi.fn(),
      newStorySet: vi.fn(),
      newCompareSet: vi.fn(),
      newAnchorSet: vi.fn(),
      job: vi.fn(),
    },
  };
});
const mocked = vi.mocked(api);

beforeEach(() => {
  vi.resetAllMocks();
  mocked.roots.mockResolvedValue([{ name: "archive", path: "/archive", mounted: true }]);
  mocked.children.mockResolvedValue([]);
  mocked.labelSets.mockResolvedValue([
    {
      set_id: "s1",
      created_at: "2026-10-13T10:00:00Z",
      kind: "describe",
      text: "fog over water",
      folders: ["/archive/Mill"],
      total: 60,
      labelled: 15,
    },
  ]);
});

describe("Label searches", () => {
  it("lists labelled searches with their progress", async () => {
    renderAt(<Labels />, { path: "/label", route: "/label" });
    const card = await screen.findByTestId("label-set-card");
    expect(card).toHaveTextContent("“fog over water”");
    expect(card).toHaveTextContent("15 of 60 labelled");
    expect(card).toHaveAttribute("href", "/label/s1");
  });

  it("needs words and a folder, then starts a set and opens it when ready", async () => {
    const user = userEvent.setup();
    mocked.newLabelSet.mockResolvedValue({ job_id: "j1" });
    mocked.job.mockResolvedValue({
      status: "done",
      done: 3,
      total: 3,
      step: "reading",
      run_id: null,
      label_set_id: "s2",
      error: null,
    });
    renderAt(<Labels />, { path: "/label", route: "/label" });
    const search = await screen.findByRole("form", { name: "New search" });
    const start = within(search).getByRole("button", { name: "Start labelling" });
    expect(start).toBeDisabled();
    expect(within(search).getByText("Type what you are looking for.")).toBeInTheDocument();
    await user.type(within(search).getByLabelText("What are you looking for?"), "stairs");
    expect(within(search).getByText("Choose at least one folder.")).toBeInTheDocument();
    await user.click(await within(search).findByRole("checkbox", { name: /archive/ }));
    await user.click(start);
    expect(mocked.newLabelSet).toHaveBeenCalledWith("stairs", ["/archive"]);
    expect(await screen.findByText("navigated away")).toBeInTheDocument();
  });

  it("starts one set however fast the button is pressed twice", async () => {
    const user = userEvent.setup();
    let answer!: (job: { job_id: string }) => void;
    mocked.newLabelSet.mockReturnValue(new Promise((res) => (answer = res)));
    renderAt(<Labels />, { path: "/label", route: "/label" });
    const search = await screen.findByRole("form", { name: "New search" });
    await user.type(within(search).getByLabelText("What are you looking for?"), "stairs");
    await user.click(await within(search).findByRole("checkbox", { name: /archive/ }));
    const start = within(search).getByRole("button", { name: "Start labelling" });
    await user.dblClick(start);
    expect(mocked.newLabelSet).toHaveBeenCalledTimes(1);
    expect(start).toBeDisabled();
    answer({ job_id: "j1" });
  });

  it("starts a taste session from folders alone", async () => {
    const user = userEvent.setup();
    mocked.newTasteSet.mockResolvedValue({ job_id: "j2" });
    mocked.job.mockResolvedValue({
      status: "done",
      done: 3,
      total: 3,
      step: "reading",
      run_id: null,
      label_set_id: "s3",
      error: null,
    });
    renderAt(<Labels />, { path: "/label", route: "/label" });
    const taste = await screen.findByRole("form", { name: "Rate for your taste" });
    const start = within(taste).getByRole("button", { name: "Start rating" });
    expect(start).toBeDisabled();
    await user.click(await within(taste).findByRole("checkbox", { name: /archive/ }));
    await user.click(start);
    expect(mocked.newTasteSet).toHaveBeenCalledWith(["/archive"]);
    expect(mocked.newLabelSet).not.toHaveBeenCalled();
    expect(await screen.findByText("navigated away")).toBeInTheDocument();
  });

  it("starts an anchor session from folders alone", async () => {
    const user = userEvent.setup();
    mocked.newAnchorSet.mockResolvedValue({ job_id: "j5" });
    mocked.job.mockResolvedValue({
      status: "done",
      done: 3,
      total: 3,
      step: "reading",
      run_id: null,
      label_set_id: "s6",
      error: null,
    });
    renderAt(<Labels />, { path: "/label", route: "/label" });
    const anchors = await screen.findByRole("form", { name: "Could it anchor a story?" });
    await user.click(await within(anchors).findByRole("checkbox", { name: /archive/ }));
    await user.click(within(anchors).getByRole("button", { name: "Choose photos to rate" }));
    expect(mocked.newAnchorSet).toHaveBeenCalledWith(["/archive"]);
    expect(await screen.findByText("navigated away")).toBeInTheDocument();
  });

  it("starts a story session from folders alone", async () => {
    const user = userEvent.setup();
    mocked.newStorySet.mockResolvedValue({ job_id: "j3" });
    mocked.job.mockResolvedValue({
      status: "done",
      done: 3,
      total: 3,
      step: "reading",
      run_id: null,
      label_set_id: "s4",
      error: null,
    });
    renderAt(<Labels />, { path: "/label", route: "/label" });
    const form = await screen.findByRole("form", { name: "Rate stories" });
    const start = within(form).getByRole("button", { name: "Tell stories to rate" });
    expect(start).toBeDisabled();
    await user.click(await within(form).findByRole("checkbox", { name: /archive/ }));
    await user.click(start);
    expect(mocked.newStorySet).toHaveBeenCalledWith(["/archive"]);
    expect(mocked.newTasteSet).not.toHaveBeenCalled();
    expect(await screen.findByText("navigated away")).toBeInTheDocument();
  });

  it("starts a Compare Strongest session from folders alone", async () => {
    const user = userEvent.setup();
    mocked.newCompareSet.mockResolvedValue({ job_id: "j4" });
    mocked.job.mockResolvedValue({
      status: "done",
      done: 3,
      total: 3,
      step: "reading",
      run_id: null,
      label_set_id: "s5",
      error: null,
    });
    renderAt(<Labels />, { path: "/label", route: "/label" });
    const form = await screen.findByRole("form", { name: "Compare Strongest" });
    const start = within(form).getByRole("button", { name: "Compare first picks" });
    expect(start).toBeDisabled();
    await user.click(await within(form).findByRole("checkbox", { name: /archive/ }));
    await user.click(start);
    expect(mocked.newCompareSet).toHaveBeenCalledWith(["/archive"]);
    expect(mocked.newTasteSet).not.toHaveBeenCalled();
    expect(mocked.newStorySet).not.toHaveBeenCalled();
    expect(await screen.findByText("navigated away")).toBeInTheDocument();
  });

  it("does not quote the name of a taste session", async () => {
    mocked.labelSets.mockResolvedValue([
      {
        set_id: "t1",
        created_at: "2026-10-22T10:00:00Z",
        kind: "taste",
        text: "Your taste: Mill",
        folders: ["/archive/Mill"],
        total: 60,
        labelled: 0,
      },
    ]);
    renderAt(<Labels />, { path: "/label", route: "/label" });
    expect(await screen.findByTestId("label-set-card")).toHaveTextContent(/^Your taste: Mill/);
  });

  it("shows why a set could not be made", async () => {
    const user = userEvent.setup();
    mocked.newLabelSet.mockResolvedValue({ job_id: "j1" });
    mocked.job.mockResolvedValue({
      status: "error",
      done: 0,
      total: 0,
      step: "reading",
      run_id: null,
      label_set_id: null,
      error: "The 'tiny' embedder cannot read text.",
    });
    renderAt(<Labels />, { path: "/label", route: "/label" });
    const search = await screen.findByRole("form", { name: "New search" });
    await user.type(within(search).getByLabelText("What are you looking for?"), "stairs");
    await user.click(await within(search).findByRole("checkbox", { name: /archive/ }));
    await user.click(within(search).getByRole("button", { name: "Start labelling" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("cannot read text");
  });

  it("says so, and frees the form, when a job finishes with nothing to open", async () => {
    const user = userEvent.setup();
    mocked.newLabelSet.mockResolvedValue({ job_id: "j1" });
    mocked.job.mockResolvedValue({
      status: "done",
      done: 1,
      total: 1,
      step: "reading",
      run_id: null,
      label_set_id: null,
      error: null,
    });
    renderAt(<Labels />, { path: "/label", route: "/label" });
    const search = await screen.findByRole("form", { name: "New search" });
    await user.type(within(search).getByLabelText("What are you looking for?"), "stairs");
    await user.click(await within(search).findByRole("checkbox", { name: /archive/ }));
    await user.click(within(search).getByRole("button", { name: "Start labelling" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("made nothing to open");
    expect(within(search).getByRole("button", { name: "Start labelling" })).toBeEnabled();
  });
});
