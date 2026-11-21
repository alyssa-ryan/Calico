import { describe, expect, it } from "vitest";
import { pick } from "../test/render";
import { moreLikeThis, readPrefill } from "./prefill";

describe("readPrefill", () => {
  it("accepts a well-formed state", () => {
    const state = {
      task: "triptych",
      folders: ["/a"],
      references: [{ path: "/a/x.jpg", contentKey: "k" }],
    };
    expect(readPrefill(state)).toEqual({
      ...state,
      criteria: "",
      examples: [],
      onlyClearMatches: false,
      describeWeight: null,
      template: null,
      limit: null,
      options: {},
      dropped: [],
    });
  });

  it("rejects anything that is not a prefill", () => {
    expect(readPrefill(null)).toBeNull();
    expect(readPrefill({ task: 1, folders: [] })).toBeNull();
    expect(readPrefill({ task: "best", folders: [3] })).toBeNull();
  });

  it("drops malformed references but keeps the rest", () => {
    const prefill = readPrefill({ task: "best", folders: [], references: [{ path: 1 }] });
    expect(prefill?.references).toEqual([]);
  });

  it("keeps well-formed examples", () => {
    const examples = [{ path: "/a/e.jpg", contentKey: "e" }, { path: 2 }];
    expect(readPrefill({ task: "best", folders: [], examples })?.examples).toEqual([
      { path: "/a/e.jpg", contentKey: "e" },
    ]);
  });
});

describe("moreLikeThis", () => {
  const request = {
    task: "best",
    folders: ["/archive"],
    criteria: "golden hour",
    examples: ["/archive/Old Mill/e.jpg"],
    only_clear_matches: true,
    describe_weight: 0.7,
    template: "yours" as const,
    limit: 12,
    options: { diversity: 0.5 },
    references: [],
  };

  it("starts the same search, every setting kept, with the photo added to its examples", () => {
    expect(moreLikeThis(request, ["e"], pick("p.jpg"))).toEqual({
      task: "best",
      folders: ["/archive"],
      criteria: "golden hour",
      references: [],
      examples: [
        { path: "/archive/Old Mill/e.jpg", contentKey: "e" },
        { path: "/archive/Old Mill/p.jpg", contentKey: "p.jpg" },
      ],
      onlyClearMatches: true,
      describeWeight: 0.7,
      template: "yours",
      limit: 12,
      options: { diversity: 0.5 },
      dropped: [],
    });
  });

  it("says when the search's reference photos cannot come along", () => {
    const anchored = { ...request, task: "triptych", references: ["/uploads/a.jpg"] };
    expect(moreLikeThis(anchored, ["e"], pick("p.jpg")).dropped).toEqual(["references"]);
  });

  it("reads every setting back, and leaves out what is malformed", () => {
    const made = moreLikeThis(request, ["e"], pick("p.jpg"));
    expect(readPrefill(made)).toEqual(made);
    const odd = readPrefill({ ...made, limit: "lots", template: "mine", options: [1] });
    expect(odd).toMatchObject({ limit: null, template: null, options: {} });
  });

  it("does not add a photo that is already an example", () => {
    const again = pick("e.jpg", { content_key: "e" });
    expect(moreLikeThis(request, ["e"], again).examples).toHaveLength(1);
  });
});
