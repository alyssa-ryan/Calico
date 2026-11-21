import { describe, expect, it } from "vitest";
import { serialByKey } from "./serial";

const deferred = () => {
  let resolve!: () => void;
  let reject!: (e: Error) => void;
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

describe("serialByKey", () => {
  it("runs saves to one key one after another, and different keys at once", async () => {
    const queue = serialByKey();
    const first = deferred();
    const started: string[] = [];
    const one = queue("a", () => {
      started.push("a1");
      return first.promise;
    });
    const two = queue("a", async () => {
      started.push("a2");
    });
    const other = queue("b", async () => {
      started.push("b1");
    });
    await other;
    expect(started).toEqual(["a1", "b1"]);
    first.resolve();
    await Promise.all([one, two]);
    expect(started).toEqual(["a1", "b1", "a2"]);
  });

  it("goes on after a save fails, and the failure reaches its caller", async () => {
    const queue = serialByKey();
    const failing = queue("a", () => Promise.reject(new Error("offline")));
    const next = queue("a", async () => "saved");
    await expect(failing).rejects.toThrow("offline");
    await expect(next).resolves.toBe("saved");
  });
});
