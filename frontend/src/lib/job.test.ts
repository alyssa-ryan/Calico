import { describe, expect, it } from "vitest";
import { ApiError } from "../api/client";
import type { JobStatus } from "../api/types";
import { isFinished, jobOutcome, phase, retryJob } from "./job";

const job = (over: Partial<JobStatus>): JobStatus => ({
  status: "queued",
  done: 0,
  total: 0,
  step: "reading",
  run_id: null,
  label_set_id: null,
  error: null,
  ...over,
});

describe("phase", () => {
  it("explains a cold start", () => {
    expect(phase(job({ status: "starting" }))).toEqual({
      label: "Loading models…",
      fraction: null,
    });
  });

  it("shows indexing progress while photos are being read", () => {
    expect(phase(job({ status: "running", done: 30, total: 120 }))).toEqual({
      label: "Reading photos: 30 of 120",
      fraction: 0.25,
    });
  });

  it("counts photos compared part by part, for same view", () => {
    expect(phase(job({ status: "running", done: 150, total: 600, step: "comparing" }))).toEqual({
      label: "Comparing details: 150 of 600",
      fraction: 0.25,
    });
  });

  it("switches to selecting once every photo is read", () => {
    expect(phase(job({ status: "running", done: 120, total: 120 }))).toEqual({
      label: "Choosing photographs…",
      fraction: null,
    });
  });

  it("names the step after reading photos for each kind of job", () => {
    const read = job({ status: "running", done: 5, total: 5 });
    expect(phase(read, "Choosing photos to label…").label).toBe("Choosing photos to label…");
  });

  it("waits politely before the job starts", () => {
    expect(phase(undefined).label).toBe("Sending request…");
    expect(phase(job({ status: "queued" })).label).toBe("Waiting to start…");
  });
});

describe("isFinished", () => {
  it("is true only for done and error", () => {
    expect(isFinished(job({ status: "done" }))).toBe(true);
    expect(isFinished(job({ status: "error" }))).toBe(true);
    expect(isFinished(job({ status: "running" }))).toBe(false);
    expect(isFinished(undefined)).toBe(false);
  });
});

describe("jobOutcome", () => {
  it("opens what a finished job made", () => {
    expect(jobOutcome(job({ status: "done", run_id: "r1" }), "run_id")).toEqual({ open: "r1" });
  });

  it("is a failure when a finished job made nothing to open", () => {
    const outcome = jobOutcome(job({ status: "done" }), "label_set_id");
    expect(outcome).toEqual({ failed: expect.stringContaining("made nothing to open") });
  });

  it("is the job's own error when it failed", () => {
    expect(jobOutcome(job({ status: "error", error: "No photos." }), "run_id")).toEqual({
      failed: "No photos.",
    });
  });

  it("waits while the job runs", () => {
    expect(jobOutcome(job({ status: "running" }), "run_id")).toBeNull();
    expect(jobOutcome(undefined, "run_id")).toBeNull();
  });
});

describe("retryJob", () => {
  it("retries a network failure and a server error, a few times", () => {
    expect(retryJob(0, new TypeError("Failed to fetch"))).toBe(true);
    expect(retryJob(0, new ApiError("Bad gateway", 502))).toBe(true);
    expect(retryJob(5, new TypeError("Failed to fetch"))).toBe(false);
  });

  it("does not retry an answer that will not change, such as a job the server forgot", () => {
    expect(retryJob(0, new ApiError("No such job.", 404))).toBe(false);
  });
});
