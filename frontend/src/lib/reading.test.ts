import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { readingToShow, useDebounced } from "./reading";

describe("the reading of a description", () => {
  it("is shown only when it found more than words to look for", () => {
    expect(readingToShow({ used: true, parts: [{ kind: "looks for", text: "fog" }] })).toBeNull();
    const parts = [
      { kind: "looks for", text: "a sky" },
      { kind: "without", text: "clouds" },
    ];
    expect(readingToShow({ used: true, parts })).toEqual(parts);
    expect(readingToShow({ used: false, parts })).toBeNull();
    expect(readingToShow(undefined)).toBeNull();
  });
});

describe("useDebounced", () => {
  afterEach(() => vi.useRealTimers());

  it("settles on the last value after a pause", () => {
    vi.useFakeTimers();
    const { result, rerender } = renderHook(({ v }) => useDebounced(v, 300), {
      initialProps: { v: "f" },
    });
    rerender({ v: "fo" });
    rerender({ v: "fog" });
    expect(result.current).toBe("f");
    act(() => vi.advanceTimersByTime(300));
    expect(result.current).toBe("fog");
  });
});
