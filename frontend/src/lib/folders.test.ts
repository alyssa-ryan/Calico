import {
  allTicked,
  coveredBy,
  folderName,
  isInside,
  tickAll,
  toggleFolder,
  topMost,
  untickAll,
} from "./folders";

describe("folder selection", () => {
  it("detects containment without matching name prefixes", () => {
    expect(isInside("/a/b", "/a")).toBe(true);
    expect(isInside("/ab", "/a")).toBe(false);
    expect(isInside("/a", "/a")).toBe(false);
  });

  it("ticking a parent replaces ticked children", () => {
    expect(toggleFolder(["/a/x", "/a/y", "/b"], "/a")).toEqual(["/b", "/a"]);
  });

  it("ignores toggles on covered folders", () => {
    expect(toggleFolder(["/a"], "/a/x")).toEqual(["/a"]);
    expect(coveredBy("/a/x/y", ["/a"])).toBe("/a");
  });

  it("unticks a ticked folder", () => {
    expect(toggleFolder(["/a", "/b"], "/a")).toEqual(["/b"]);
  });

  it("sends only top-most folders", () => {
    expect(topMost(["/a/x", "/a", "/b", "/b"])).toEqual(["/a", "/b"]);
  });

  it("names a folder by its last segment", () => {
    expect(folderName("/Volumes/Archive/Old Mill/")).toBe("Old Mill");
  });
});

describe("ticking every search match", () => {
  it("adds each match, replacing ticked folders inside one", () => {
    expect(tickAll(["/b", "/a/x"], ["/a", "/c"])).toEqual(["/b", "/a", "/c"]);
  });

  it("leaves matches a ticked folder already covers", () => {
    expect(tickAll(["/a"], ["/a/x", "/c"])).toEqual(["/a", "/c"]);
  });

  it("knows when every match is ticked or covered", () => {
    expect(allTicked(["/a", "/c"], ["/a/x", "/c"])).toBe(true);
    expect(allTicked(["/a"], ["/a/x", "/c"])).toBe(false);
    expect(allTicked(["/a"], [])).toBe(false);
  });

  it("unticks exactly the matches, keeping everything else", () => {
    expect(untickAll(["/a", "/b", "/c"], ["/a", "/c", "/d"])).toEqual(["/b"]);
  });
});
