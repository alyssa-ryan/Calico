import { describe, expect, it } from "vitest";
import {
  labelChoices,
  labelForKey,
  labelProgress,
  nextUnlabelled,
  storyMisfits,
  toggledFit,
  withLabel,
  withStory,
} from "./labels";

const items = (labels: (0 | 1 | 2 | null)[]) =>
  labels.map((label, i) => ({ content_key: `k${i}`, name: `${i}.jpg`, folder: "F", label }));

describe("labelling", () => {
  it("maps number and letter keys to labels", () => {
    expect([labelForKey("1"), labelForKey("y"), labelForKey("Y")]).toEqual([2, 2, 2]);
    expect([labelForKey("2"), labelForKey("p")]).toEqual([1, 1]);
    expect([labelForKey("3"), labelForKey("n")]).toEqual([0, 0]);
    expect(labelForKey("x")).toBeUndefined();
  });

  it("finds the next unlabelled photo after the current one, wrapping round", () => {
    const set = items([2, null, 0, null]);
    expect(nextUnlabelled(set, 0)).toBe(1);
    expect(nextUnlabelled(set, 1)).toBe(3);
    expect(nextUnlabelled(set, 3)).toBe(1);
    expect(nextUnlabelled(items([2, 0]), 0)).toBe(-1);
  });

  it("updates one photo's label without touching the rest", () => {
    const set = items([null, null]);
    const updated = withLabel(set, "k1", 1);
    expect(updated.map((i) => i.label)).toEqual([null, 1]);
    expect(set[1]?.label).toBeNull();
  });

  it("describes progress", () => {
    expect(labelProgress(items([2, null, 0]))).toEqual({
      labelled: 2,
      total: 3,
      text: "2 of 3 labelled",
      done: false,
    });
    expect(labelProgress(items([2])).done).toBe(true);
  });
});

describe("labelChoices", () => {
  it("names the choices for what is being judged, on the same keys", () => {
    expect(labelChoices("describe").map((c) => c.name)).toEqual(["Matches", "Partly", "No"]);
    expect(labelChoices("same_view").map((c) => [c.name, c.value, c.key])).toEqual([
      ["Same view", 2, "1"],
      ["Nearly", 1, "2"],
      ["No", 0, "3"],
    ]);
    expect(labelChoices("taste").map((c) => [c.name, c.value, c.key])).toEqual([
      ["Like", 2, "1"],
      ["Not sure", 1, "2"],
      ["Don't like", 0, "3"],
    ]);
    expect(labelChoices("stories").map((c) => c.name)).toEqual(["Works", "Not sure", "Doesn't"]);
    expect(labelChoices("compare").map((c) => c.name)).toEqual(["Like", "Not sure", "Don't like"]);
    expect(labelChoices("anchor").map((c) => c.name)).toEqual([
      "Could anchor",
      "Not sure",
      "Could not",
    ]);
  });
});

const stories = (labels: (0 | 1 | 2 | null)[]) =>
  labels.map((label, index) => ({
    index,
    label,
    photos: ["a", "b", "c"].map((p) => ({
      content_key: `${index}${p}`,
      name: `${p}.jpg`,
      folder: "Mill",
      fits: true,
    })),
  }));

describe("rating stories", () => {
  it("counts and finds stories as it does photos", () => {
    expect(labelProgress(stories([2, null])).text).toBe("1 of 2 labelled");
    expect(nextUnlabelled(stories([null, 2, null]), 0)).toBe(2);
  });

  it("marks a photo that does not fit, and unmarks it", () => {
    const [story] = stories([null]);
    if (!story) throw new Error("no story");
    expect(storyMisfits(story)).toEqual([]);
    expect(toggledFit(story, "0b")).toEqual(["0b"]);
    const marked = withStory([story], 0, 2, ["0b"]);
    expect(marked[0]?.label).toBe(2);
    expect(marked[0]?.photos.map((p) => p.fits)).toEqual([true, false, true]);
    expect(story.photos[1]?.fits).toBe(true);
    const [again] = marked;
    if (!again) throw new Error("no story");
    expect(toggledFit(again, "0b")).toEqual([]);
    expect(toggledFit(again, "0c")).toEqual(["0b", "0c"]);
  });
});
