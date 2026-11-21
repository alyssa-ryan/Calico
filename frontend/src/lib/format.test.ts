import { describe, expect, it } from "vitest";
import type { Evidence, Pick } from "../api/types";
import {
  clearMatchBadge,
  connectionChips,
  describeWeightChip,
  examplesChip,
  fileName,
  formatDate,
  groupByMonth,
  moreFrames,
  percent,
  SAME_VIEW,
  signalLabel,
} from "./format";

const pick = (name: string, captured_at: string | null): Pick => ({
  path: `/archive/Old Mill/${name}`,
  content_key: name,
  location: "Old Mill",
  captured_at,
  score: 0.5,
  evidence: [],
  burst_size: 1,
  alternates: [],
});

describe("describeWeightChip", () => {
  it("says how much the description counted", () => {
    expect(describeWeightChip(0.4)).toBe("counts 40%");
    expect(describeWeightChip(1 / 3)).toBe("counts 33%");
  });
});

describe("examplesChip", () => {
  it("counts the example photos", () => {
    expect(examplesChip(1)).toBe("like 1 example");
    expect(examplesChip(3)).toBe("like 3 examples");
  });
});

describe("moreFrames", () => {
  it("counts a pick's other frames, from its burst or its view", () => {
    expect(moreFrames(1)).toBe("+1 more frame");
    expect(moreFrames(3)).toBe("+3 more frames");
  });
});

describe("percent", () => {
  it("rounds a 0-1 value to a whole percentage", () => {
    expect(percent(0.824)).toBe("82%");
    expect(percent(1)).toBe("100%");
    expect(percent(0)).toBe("0%");
  });
});

describe("formatDate", () => {
  it("formats a capture time as day month year", () => {
    expect(formatDate("2021-03-08T09:15:00")).toBe("8 Mar 2021");
  });
  it("says undated for missing times", () => {
    expect(formatDate(null)).toBe("Undated");
  });
});

describe("fileName", () => {
  it("returns the last path segment", () => {
    expect(fileName("/a/b/IMG_0042.JPG")).toBe("IMG_0042.JPG");
  });
});

describe("signalLabel", () => {
  it("turns snake_case signal names into words", () => {
    expect(signalLabel("place_type")).toBe("place type");
    expect(signalLabel("kind_mix")).toBe("kind mix");
  });
});

describe("connectionChips", () => {
  it("lists a story's connections, not its strength, strongest first", () => {
    const evidence: Evidence[] = [
      { signal: "strength", value: 0.9, weight: 0.4, detail: "" },
      { signal: "theme", value: 0.6, weight: 0.3, detail: "" },
      { signal: "color", value: 0.8, weight: 0.3, detail: "similar palette" },
    ];
    expect(connectionChips(evidence)).toEqual([
      { label: "color", value: "80%", detail: "similar palette" },
      { label: "theme", value: "60%", detail: "" },
    ]);
  });
});

describe("groupByMonth", () => {
  it("keeps chronological order and puts undated photos last", () => {
    const groups = groupByMonth([
      pick("a.jpg", "2021-03-08T09:00:00"),
      pick("b.jpg", "2021-03-20T09:00:00"),
      pick("c.jpg", "2024-05-01T09:00:00"),
      pick("d.jpg", null),
    ]);
    expect(groups.map((g) => [g.label, g.picks.map((p) => p.content_key)])).toEqual([
      ["March 2021", ["a.jpg", "b.jpg"]],
      ["May 2024", ["c.jpg"]],
      ["Undated", ["d.jpg"]],
    ]);
  });
});

describe("clearMatchBadge", () => {
  it("labels only the photos the model is confident about", () => {
    expect(clearMatchBadge({ ...pick("a.jpg", null), clear_match: true })).toBe("Clear match");
    expect(clearMatchBadge({ ...pick("a.jpg", null), clear_match: false })).toBeNull();
    expect(clearMatchBadge(pick("a.jpg", null))).toBeNull();
  });
});

describe("same view tiers", () => {
  it("summarises both tiers, or only the one there is", () => {
    expect(SAME_VIEW.summary(27, 26)).toBe(
      "27 photos are the same view, oldest first. 26 more are possibly the same view.",
    );
    expect(SAME_VIEW.summary(1, 0)).toBe("1 photo is the same view, oldest first.");
    expect(SAME_VIEW.summary(0, 3)).toBe(
      "No photo is surely the same view. 3 are possibly the same view.",
    );
  });
});
