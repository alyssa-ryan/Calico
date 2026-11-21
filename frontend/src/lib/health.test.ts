import { describe, expect, it } from "vitest";
import { healthInterval } from "./health";

describe("healthInterval", () => {
  it("checks often while the models load, rarely once ready", () => {
    expect(healthInterval("starting", false)).toBe(1_500);
    expect(healthInterval("ready", false)).toBe(30_000);
  });

  it("checks rarely when the engine failed: it stays failed until a restart", () => {
    expect(healthInterval("error", false)).toBe(30_000);
  });

  it("checks every few seconds while the server does not answer, to see it come back", () => {
    expect(healthInterval("ready", true)).toBe(5_000);
    expect(healthInterval(undefined, true)).toBe(5_000);
  });
});
