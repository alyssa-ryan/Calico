import { useQuery } from "@tanstack/react-query";
import { AlertCircle, Loader2 } from "lucide-react";
import { type FormEvent, useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { RootFolder } from "../api/types";
import { FolderPicker } from "../components/FolderPicker";
import { folderName } from "../lib/folders";
import { ANCHORS, COMPARE, STORIES, TASTE } from "../lib/format";
import { jobOutcome, phase, useJob } from "../lib/job";

type Session = "taste" | "stories" | "compare" | "anchor";

const SESSIONS: Record<Session, typeof TASTE> = {
  taste: TASTE,
  stories: STORIES,
  compare: COMPARE,
  anchor: ANCHORS,
};

const START: Record<Session, (folders: string[]) => Promise<{ job_id: string }>> = {
  taste: (folders) => api.newTasteSet(folders),
  stories: (folders) => api.newStorySet(folders),
  compare: (folders) => api.newCompareSet(folders),
  anchor: (folders) => api.newAnchorSet(folders),
};

/** Labelled searches, and a form to start one. */
export function Labels() {
  const navigate = useNavigate();
  const sets = useQuery({ queryKey: ["label-sets"], queryFn: api.labelSets, staleTime: 0 });
  const roots = useQuery({ queryKey: ["roots"], queryFn: api.roots });
  const [text, setText] = useState("");
  const [folders, setFolders] = useState<string[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  // Set at once, so a second click before the server answers starts nothing.
  const sending = useRef(false);
  const [pending, setPending] = useState(false);
  const [jobFor, setJobFor] = useState<"search" | Session>("search");
  const [error, setError] = useState<string | null>(null);

  const job = useJob(jobId);
  useEffect(() => {
    const outcome = jobOutcome(job.data, "label_set_id");
    if (outcome && "open" in outcome) {
      navigate(`/label/${outcome.open}`);
    } else if (outcome || job.isError) {
      setError(outcome?.failed ?? job.error?.message ?? "The request failed.");
      setJobId(null);
    }
  }, [job.data, job.isError, job.error, navigate]);

  const busy = jobId !== null || pending;
  const blocker = !text.trim()
    ? "Type what you are looking for."
    : folders.length === 0
      ? "Choose at least one folder."
      : null;

  /** Start a job once, however fast it is asked for twice. */
  const start = async (what: "search" | Session, make: () => Promise<{ job_id: string }>) => {
    if (busy || sending.current) return;
    sending.current = true;
    setPending(true);
    setError(null);
    try {
      setJobFor(what);
      setJobId((await make()).job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      sending.current = false;
      setPending(false);
    }
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (blocker) return;
    await start("search", () => api.newLabelSet(text.trim(), folders));
  };

  const startSession = (session: Session, chosen: string[]) =>
    start(session, () => START[session](chosen));

  const progress = phase(
    job.data,
    jobFor === "search" ? "Choosing photos to label…" : SESSIONS[jobFor].choosing,
  );
  const searchBusy = busy && jobFor === "search";
  return (
    <main className="mx-auto max-w-3xl space-y-10 px-4 py-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Label searches</h1>
        <p className="text-zinc-600 dark:text-zinc-400">
          Labelling teaches Calico what a match looks like to you. Each search shows about 60 photos
          from the folders you choose; mark each one. Calico then measures every way of reading a
          description against your answers (<code className="text-sm">calico eval criteria</code>
          ). To calibrate “Same view over time”, use <em>Label for calibration</em> on a same-view
          run.
        </p>
      </header>

      <form onSubmit={submit} aria-label="New search" className="space-y-4" aria-busy={searchBusy}>
        <h2 className="text-lg font-semibold">New search</h2>
        <label className="block space-y-1.5">
          <span className="text-sm font-medium">What are you looking for?</span>
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="fog over water, a staircase seen from below…"
            className="w-full rounded-xl border border-zinc-300 bg-white px-3 py-2.5 shadow-sm focus-visible:outline-2 focus-visible:outline-accent-500 dark:border-zinc-700 dark:bg-zinc-900"
          />
        </label>
        <div className="space-y-1.5">
          <p className="text-sm font-medium">Where to look</p>
          {roots.data ? (
            <FolderPicker roots={roots.data} ticked={folders} onChange={setFolders} />
          ) : (
            <Loader2 className="size-5 animate-spin text-zinc-500" />
          )}
        </div>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <button
            type="submit"
            disabled={blocker !== null || busy}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-accent-600 px-6 font-medium text-white shadow-sm transition hover:bg-accent-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-500 disabled:cursor-not-allowed disabled:bg-zinc-300 dark:disabled:bg-zinc-700"
          >
            {searchBusy && <Loader2 className="size-4 animate-spin" />}
            Start labelling
          </button>
          <p className="text-sm text-zinc-500 dark:text-zinc-400" role="status">
            {searchBusy ? progress.label : blocker}
          </p>
        </div>
      </form>

      {(["taste", "stories", "compare", "anchor"] as const).map((session) => (
        <SessionForm
          key={session}
          strings={SESSIONS[session]}
          roots={roots.data}
          busy={busy}
          running={busy && jobFor === session ? progress.label : null}
          onStart={(chosen) => startSession(session, chosen)}
        />
      ))}
      {error && (
        <p role="alert" className="flex items-start gap-2 text-red-600 dark:text-red-400">
          <AlertCircle className="mt-0.5 size-4 shrink-0" /> {error}
        </p>
      )}

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Your labelled searches</h2>
        {sets.isError && (
          <p role="alert" className="text-red-600">
            {sets.error.message}
          </p>
        )}
        {sets.data?.length === 0 && (
          <p className="rounded-xl border border-dashed border-zinc-300 p-8 text-center text-zinc-500 dark:border-zinc-700">
            None yet.
          </p>
        )}
        <ul className="space-y-2">
          {sets.data?.map((s) => (
            <li key={s.set_id}>
              <Link
                to={`/label/${s.set_id}`}
                data-testid="label-set-card"
                className="block space-y-2 rounded-xl border border-zinc-200 bg-white p-4 transition hover:shadow-md focus-visible:outline-2 focus-visible:outline-accent-500 dark:border-zinc-800 dark:bg-zinc-900"
              >
                <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                  <p className="font-medium">{s.kind === "describe" ? `“${s.text}”` : s.text}</p>
                  <p className="text-sm text-zinc-500">
                    {s.labelled} of {s.total} labelled
                  </p>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
                  <div
                    className="h-full rounded-full bg-accent-500"
                    style={{ width: `${s.total ? Math.round((100 * s.labelled) / s.total) : 0}%` }}
                  />
                </div>
                <p className="truncate text-xs text-zinc-500">
                  {s.folders.map(folderName).join(" · ")} ·{" "}
                  {new Date(s.created_at).toLocaleDateString()}
                </p>
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}

/** A rating session started from folders alone: taste, stories, Compare Strongest or anchors. */
function SessionForm({
  strings,
  roots,
  busy,
  running,
  onStart,
}: {
  strings: typeof TASTE;
  roots: readonly RootFolder[] | undefined;
  busy: boolean;
  /** What this session's job is doing, while it runs. */
  running: string | null;
  onStart: (folders: string[]) => void;
}) {
  const [folders, setFolders] = useState<string[]>([]);
  const blocker = folders.length === 0 ? "Choose at least one folder." : null;
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!blocker) onStart(folders);
  };
  return (
    <form
      onSubmit={submit}
      aria-label={strings.heading}
      className="space-y-4"
      aria-busy={running !== null}
    >
      <h2 className="text-lg font-semibold">{strings.heading}</h2>
      <p className="text-sm text-zinc-600 dark:text-zinc-400">{strings.help}</p>
      {roots ? (
        <FolderPicker roots={roots} ticked={folders} onChange={setFolders} />
      ) : (
        <Loader2 className="size-5 animate-spin text-zinc-500" />
      )}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <button
          type="submit"
          disabled={blocker !== null || busy}
          className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-accent-600 px-6 font-medium text-white shadow-sm transition hover:bg-accent-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-500 disabled:cursor-not-allowed disabled:bg-zinc-300 dark:disabled:bg-zinc-700"
        >
          {running !== null && <Loader2 className="size-4 animate-spin" />}
          {strings.start}
        </button>
        <p className="text-sm text-zinc-500 dark:text-zinc-400" role="status">
          {running ?? blocker}
        </p>
      </div>
    </form>
  );
}
