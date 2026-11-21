import { describe, expect, it } from "vitest";
import { ignoresShortcut } from "./keys";

const key = (over: Partial<KeyboardEventInit> = {}, target?: HTMLElement) => {
  const event = new KeyboardEvent("keydown", { key: "1", ...over });
  if (target) Object.defineProperty(event, "target", { value: target });
  return event;
};

describe("ignoresShortcut", () => {
  it("lets a plain key press through", () => {
    expect(ignoresShortcut(key({}, document.createElement("button")))).toBe(false);
  });

  it("ignores a held key, so holding 1 does not label the whole set", () => {
    expect(ignoresShortcut(key({ repeat: true }))).toBe(true);
  });

  it("ignores typing in a field", () => {
    for (const tag of ["input", "textarea", "select"]) {
      expect(ignoresShortcut(key({}, document.createElement(tag)))).toBe(true);
    }
    const editable = document.createElement("div");
    editable.contentEditable = "true";
    expect(ignoresShortcut(key({}, editable))).toBe(true);
  });

  it("ignores shortcuts with a modifier", () => {
    expect(ignoresShortcut(key({ metaKey: true }))).toBe(true);
  });
});
