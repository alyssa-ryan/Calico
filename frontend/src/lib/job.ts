/** Background jobs (selections, labelling sets): what to say while one runs, and polling it. */
import { useQuery } from "@tanstack/react-query";
import { ApiError, api } from "../api/client";
import type { JobStatus } from "../api/types";
import { JOB_MADE_NOTHING } from "./format";

export type Phase = { label: string; fraction: number | null };

/** What to say while ``job`` runs; ``working`` is the step after reading photos. */
export function phase(job: JobStatus | undefined, working = "Choosing photographs…"): Phase {
  if (!job) return { label: "Sending request…", fraction: null };
  switch (job.status) {
    case "starting":
      return { label: "Loading models…", fraction: null };
    case "queued":
      return { label: "Waiting to start…", fraction: null };
    case "running":
      if (job.total > 0 && job.done < job.total) {
        const doing = job.step === "comparing" ? "Comparing details" : "Reading photos";
        return {
          label: `${doing}: ${job.done} of ${job.total}`,
          fraction: job.done / job.total,
        };
      }
      return { label: working, fraction: null };
    case "done":
      return { label: "Done", fraction: 1 };
    case "error":
      return { label: job.error ?? "Something went wrong.", fraction: null };
  }
}

export const isFinished = (job: JobStatus | undefined): boolean =>
  job?.status === "done" || job?.status === "error";

export type Outcome = { open: string } | { failed: string } | null;

/**
 * What a page waiting on ``job`` does next: open what it made, show why it failed, or keep
 * waiting (null). A job that finished with nothing to open is a failure, or the page would
 * show "Done" at 100% for good (audit A08-6, A08-16).
 */
export function jobOutcome(job: JobStatus | undefined, made: "run_id" | "label_set_id"): Outcome {
  if (job?.status === "error") return { failed: job.error ?? "Something went wrong." };
  if (job?.status !== "done") return null;
  const id = job[made];
  return id ? { open: id } : { failed: JOB_MADE_NOTHING };
}

/**
 * Whether polling a job tries again after ``error``: a network blip or a server error
 * (5xx, as a proxy gives while the server restarts) may pass; any other answer from the
 * API, such as 404 for a job a restarted server forgot, will not.
 */
export const retryJob = (failures: number, error: Error): boolean =>
  failures < 5 && !(error instanceof ApiError && error.status < 500);

/** Poll a job until it finishes, retrying as ``retryJob`` says, with backoff. */
export function useJob(jobId: string | null) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId ?? ""),
    enabled: jobId !== null,
    refetchInterval: (query) => (isFinished(query.state.data) ? false : 500),
    retry: retryJob,
    retryDelay: (attempt) => Math.min(500 * 2 ** attempt, 4000),
  });
}
