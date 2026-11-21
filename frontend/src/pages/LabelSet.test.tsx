import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { LabelSetDetail } from "../api/types";
import { renderAt, testClient } from "../test/render";
import { LabelSet } from "./LabelSet";

vi.mock("../api/client", async (original) => {
  const actual = await original<typeof import("../api/client")>();
  return {
    ...actual,
    api: { labelSet: vi.fn(), setLabel: vi.fn(), setStory: vi.fn(), deleteLabelSet: vi.fn() },
  };
});
const mocked = vi.mocked(api);

const detail = (labels: (0 | 1 | 2 | null)[]): LabelSetDetail => ({
  set_id: "s1",
  created_at: "2026-10-13T10:00:00Z",
  kind: "describe",
  text: "fog over water",
  folders: ["/archive"],
  scorer: "baseline@1",
  items: labels.map((label, i) => ({
    content_key: `k${i}`,
    name: `${i}.jpg`,
    folder: "Mill",
    label,
  })),
  stories: [],
});

const show = (client = testClient()) =>
  renderAt(<LabelSet />, { path: "/label/s1", route: "/label/:setId", client });

const deferred = () => {
  let resolve!: () => void;
  let reject!: (e: Error) => void;
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

beforeEach(() => {
  vi.resetAllMocks();
  mocked.setLabel.mockResolvedValue(undefined);
  mocked.setStory.mockResolvedValue(undefined);
});

describe("Labelling a search", () => {
  it("labels the current photo with a key and moves to the next unlabelled one", async () => {
    const user = userEvent.setup();
    mocked.labelSet.mockResolvedValue(detail([null, 2, null]));
    show();
    expect(await screen.findByText("1 of 3 labelled")).toBeInTheDocument();
    screen.getByRole("button", { name: "Enlarge 0.jpg" }).focus();

    await user.keyboard("1");
    expect(mocked.setLabel).toHaveBeenCalledWith("s1", "k0", 2);
    expect(screen.getByRole("button", { name: "Matches: 0.jpg" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: "Enlarge 2.jpg" })).toHaveFocus(); // skipped 1.jpg

    await user.keyboard("n");
    expect(mocked.setLabel).toHaveBeenLastCalledWith("s1", "k2", 0);
    expect(await screen.findByRole("status")).toHaveTextContent("All 3 labelled");
  });

  it("clicking the chosen label again clears it", async () => {
    const user = userEvent.setup();
    mocked.labelSet.mockResolvedValue(detail([1]));
    show();
    await user.click(await screen.findByRole("button", { name: "Partly: 0.jpg" }));
    expect(mocked.setLabel).toHaveBeenCalledWith("s1", "k0", null);
    expect(screen.getByText("0 of 1 labelled")).toBeInTheDocument();
  });

  it("undoes a label the server refuses and says why", async () => {
    const user = userEvent.setup();
    mocked.labelSet.mockResolvedValue(detail([null]));
    mocked.setLabel.mockRejectedValue(new Error("disk full"));
    show();
    await user.click(await screen.findByRole("button", { name: "No: 0.jpg" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("disk full");
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "No: 0.jpg" })).toHaveAttribute(
        "aria-pressed",
        "false",
      ),
    );
  });

  it("enlarges a photo, labels inside the viewer and closes with Escape", async () => {
    const user = userEvent.setup();
    mocked.labelSet.mockResolvedValue(detail([null, null]));
    show();
    const opener = await screen.findByRole("button", { name: "Enlarge 0.jpg" });
    await user.click(opener);
    const viewer = screen.getByTestId("label-viewer");
    expect(viewer).toHaveAccessibleName("0.jpg");
    await user.keyboard("2");
    expect(mocked.setLabel).toHaveBeenCalledWith("s1", "k0", 1);
    expect(screen.getByTestId("label-viewer")).toHaveAccessibleName("1.jpg");
    await user.click(within(viewer).getByRole("button", { name: "Previous" }));
    expect(screen.getByTestId("label-viewer")).toHaveAccessibleName("0.jpg");
    await user.keyboard("{Escape}");
    expect(screen.queryByTestId("label-viewer")).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  it("explains how to label until the first label", async () => {
    mocked.labelSet.mockResolvedValue(detail([null]));
    show();
    const help = (await screen.findByText("How to label")).closest("details");
    expect(help).toHaveAttribute("open");
  });

  it("shows the reference and asks about the same view for a same-view set", async () => {
    mocked.labelSet.mockResolvedValue({
      ...detail([null, null]),
      kind: "same_view",
      text: "Same view as ref.jpg",
    });
    show();
    expect(await screen.findByAltText("The reference view")).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Same view as ref.jpg");
    expect(screen.getByRole("button", { name: "Same view: 0.jpg" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Nearly: 0.jpg" })).toBeInTheDocument();
  });

  it("asks whether you like each photo in a taste session, with no reference", async () => {
    mocked.labelSet.mockResolvedValue({
      ...detail([null]),
      kind: "taste",
      text: "Your taste: Mill",
    });
    show();
    expect(await screen.findByRole("button", { name: "Like: 0.jpg" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/^Your taste: Mill$/);
    expect(screen.getByText(/would you want Calico to pick it/)).toBeInTheDocument();
    expect(screen.queryByAltText("The reference view")).toBeNull();
  });

  it("asks whether each photo belongs among the first picks when comparing", async () => {
    mocked.labelSet.mockResolvedValue({
      ...detail([2]),
      kind: "compare",
      text: "Compare Strongest: Mill",
    });
    show();
    expect(await screen.findByRole("button", { name: "Like: 0.jpg" })).toBeInTheDocument();
    expect(screen.getByText(/among the first photos Calico shows you/)).toBeInTheDocument();
    expect(screen.getByText(/calico eval compare/)).toBeInTheDocument();
    expect(screen.queryByAltText("The reference view")).toBeNull();
  });

  it("has no reference for a description", async () => {
    mocked.labelSet.mockResolvedValue(detail([null]));
    show();
    expect(await screen.findByRole("button", { name: "Matches: 0.jpg" })).toBeInTheDocument();
    expect(screen.queryByAltText("The reference view")).toBeNull();
  });
});

const storySet = (labels: (0 | 1 | 2 | null)[]): LabelSetDetail => ({
  ...detail([]),
  kind: "stories",
  text: "Stories: Mill",
  scorer: "triptych: general, general without variety",
  stories: labels.map((label, index) => ({
    index,
    label,
    photos: ["a", "b", "c"].map((p) => ({
      content_key: `${index}${p}`,
      name: `${index}${p}.jpg`,
      folder: "Mill",
      fits: true,
    })),
  })),
});

describe("Rating stories", () => {
  it("rates the current story with a key and moves to the next unrated one", async () => {
    const user = userEvent.setup();
    mocked.labelSet.mockResolvedValue(storySet([null, 2, null]));
    show();
    expect(await screen.findByText("1 of 3 labelled")).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/^Stories: Mill$/);
    expect(screen.getByText(/do the three photographs belong together/)).toBeInTheDocument();
    screen.getByRole("group", { name: "Story 1 of 3" }).focus();

    await user.keyboard("1");
    expect(mocked.setStory).toHaveBeenCalledWith("s1", 0, 2, []);
    expect(screen.getByRole("button", { name: "Works: story 1" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("group", { name: "Story 3 of 3" })).toHaveFocus(); // skipped 2

    await user.keyboard("3");
    expect(mocked.setStory).toHaveBeenLastCalledWith("s1", 2, 0, []);
    const done = await screen.findByRole("status");
    expect(done).toHaveTextContent("All 3 labelled");
    expect(done).toHaveTextContent("calico eval stories");
  });

  it("marks a photo that alone does not fit, keeping the story's rating", async () => {
    const user = userEvent.setup();
    mocked.labelSet.mockResolvedValue(storySet([1]));
    show();
    const misfit = await screen.findByRole("button", { name: "Doesn't fit: 0b.jpg" });
    await user.click(misfit);
    expect(mocked.setStory).toHaveBeenCalledWith("s1", 0, 1, ["0b"]);
    expect(misfit).toHaveAttribute("aria-pressed", "true");
    await user.click(misfit);
    expect(mocked.setStory).toHaveBeenLastCalledWith("s1", 0, 1, []);
  });

  it("undoes a rating the server refuses and says why", async () => {
    const user = userEvent.setup();
    mocked.labelSet.mockResolvedValue(storySet([null]));
    mocked.setStory.mockRejectedValue(new Error("disk full"));
    show();
    await user.click(await screen.findByRole("button", { name: "Doesn't: story 1" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("disk full");
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Doesn't: story 1" })).toHaveAttribute(
        "aria-pressed",
        "false",
      ),
    );
  });
});

describe("Coming back to a labelling set", () => {
  it("shows the labels already given, though the page's data was fetched before them", async () => {
    const user = userEvent.setup();
    mocked.labelSet.mockResolvedValue(detail([null, null]));
    const client = testClient();
    const first = show(client);
    await user.click(await screen.findByRole("button", { name: "Matches: 0.jpg" }));
    first.unmount();

    show(client);
    expect(await screen.findByText("1 of 2 labelled")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Matches: 0.jpg" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(mocked.labelSet).toHaveBeenCalledTimes(1); // still fresh: the cache answered
  });

  it("saves two quick changes to one photo in the order they were made", async () => {
    const user = userEvent.setup();
    mocked.labelSet.mockResolvedValue(detail([null]));
    const first = deferred();
    mocked.setLabel.mockReturnValueOnce(first.promise).mockResolvedValueOnce(undefined);
    show();
    await user.click(await screen.findByRole("button", { name: "Matches: 0.jpg" }));
    await user.click(screen.getByRole("button", { name: "No: 0.jpg" }));
    expect(mocked.setLabel).toHaveBeenCalledTimes(1);
    first.resolve();
    await waitFor(() => expect(mocked.setLabel).toHaveBeenCalledTimes(2));
    expect(mocked.setLabel).toHaveBeenLastCalledWith("s1", "k0", 0);
  });

  it("puts a label that could not be saved back, and says why", async () => {
    const user = userEvent.setup();
    mocked.labelSet.mockResolvedValue(detail([null]));
    mocked.setLabel.mockRejectedValueOnce(new Error("Calico isn't answering"));
    show();
    const matches = await screen.findByRole("button", { name: "Matches: 0.jpg" });
    await user.click(matches);
    expect(await screen.findByRole("alert")).toHaveTextContent("Calico isn't answering");
    expect(matches).toHaveAttribute("aria-pressed", "false");
  });

  it("keeps a newer label when an older save fails", async () => {
    const user = userEvent.setup();
    mocked.labelSet.mockResolvedValue(detail([null]));
    const first = deferred();
    mocked.setLabel.mockReturnValueOnce(first.promise).mockResolvedValueOnce(undefined);
    show();
    await user.click(await screen.findByRole("button", { name: "Matches: 0.jpg" }));
    await user.click(screen.getByRole("button", { name: "No: 0.jpg" }));
    first.reject(new Error("Calico isn't answering"));
    await waitFor(() => expect(mocked.setLabel).toHaveBeenCalledTimes(2));
    expect(screen.getByRole("button", { name: "No: 0.jpg" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("does not label every photo while a key is held down", async () => {
    mocked.labelSet.mockResolvedValue(detail([null, null, null]));
    show();
    await screen.findByText("0 of 3 labelled");
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "1" }));
    for (let i = 0; i < 5; i++) {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "1", repeat: true }));
    }
    await waitFor(() => expect(mocked.setLabel).toHaveBeenCalledTimes(1));
  });
});
