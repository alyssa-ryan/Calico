import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, ChevronDown, Folder, Loader2, Tags } from "lucide-react";
import { type ReactElement, useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, referenceUrl, thumbUrl } from "../api/client";
import type { Group, Pick, Rating, RunDetail } from "../api/types";
import { Lightbox } from "../components/Lightbox";
import { type CardActions, PhotoCard } from "../components/PhotoCard";
import { Progress } from "../components/Progress";
import { folderName } from "../lib/folders";
import {
  CALIBRATE,
  connectionChips,
  describeWeightChip,
  EXAMPLES,
  fileName,
  groupByMonth,
  MARKER,
  NO_MATCHES,
  ONLY_CLEAR_MATCHES,
  POSSIBLY_SAME,
  SAME_VIEW,
  TEMPLATE,
} from "../lib/format";
import { jobOutcome, phase, useJob } from "../lib/job";
import { moreLikeThis } from "../lib/prefill";
import { serialByKey } from "../lib/serial";

export function Run() {
  const { runId = "" } = useParams();
  const detail = useQuery({ queryKey: ["run", runId], queryFn: () => api.run(runId) });

  if (detail.isError) {
    return (
      <main className="mx-auto max-w-3xl px-4 py-16">
        <p role="alert" className="text-red-600">
          {detail.error.message}
        </p>
        <Link to="/runs" className="mt-4 inline-block underline">
          Back to past runs
        </Link>
      </main>
    );
  }
  if (!detail.data) {
    return (
      <main className="mx-auto max-w-7xl px-4 py-16 text-zinc-500">
        <Loader2 className="size-5 animate-spin" />
      </main>
    );
  }
  return <RunView detail={detail.data} />;
}

function RunView({ detail }: { detail: RunDetail }) {
  const { result } = detail;
  const navigate = useNavigate();
  const [ratings, rate, rateError] = useRatings(detail);
  const picks = result.groups.flatMap((g) => g.picks);
  const [viewer, setViewer] = useState<number | null>(null);
  const opener = useRef<HTMLElement | null>(null);

  const open = (pick: Pick, from: HTMLElement) => {
    opener.current = from;
    setViewer(picks.indexOf(pick));
  };
  const close = () => {
    setViewer(null);
    opener.current?.focus();
  };
  const startFrom = (task: string) => (pick: Pick) =>
    navigate("/", {
      state: {
        task,
        folders: result.request.folders,
        references: [{ path: pick.path, contentKey: pick.content_key }],
      },
    });
  const actions: CardActions = {
    story: startFrom("triptych"),
    sameView: startFrom("same_view"),
    ...(result.request.task !== "same_view" && {
      moreLike: (pick: Pick) =>
        navigate("/", { state: moreLikeThis(result.request, result.example_keys, pick) }),
    }),
  };
  const card = (pick: Pick, extra: Partial<Parameters<typeof PhotoCard>[0]> = {}) => (
    <PhotoCard
      key={pick.content_key}
      pick={pick}
      rating={ratings[pick.content_key]}
      onRate={(r) => rate(pick.content_key, r)}
      onOpen={(el) => open(pick, el)}
      actions={actions}
      {...extra}
    />
  );

  const task = result.request.task;
  return (
    <main className="mx-auto max-w-7xl space-y-8 px-4 py-8">
      <RunHeader detail={detail} />
      {rateError && (
        <p role="alert" className="flex items-start gap-2 text-red-600 dark:text-red-400">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" /> {rateError}
        </p>
      )}
      {task === "same_view" ? (
        <Timeline
          runId={result.run_id}
          sure={result.groups[0]}
          possible={result.groups[1]}
          hasReference={result.request.references.length > 0}
          label={detail.reference_label}
          card={card}
          calibrate={
            result.request.references[0] ? (
              <CalibrateView
                folders={result.request.folders}
                reference={result.request.references[0]}
              />
            ) : null
          }
        />
      ) : result.groups.length === 0 || picks.length === 0 ? (
        <NoMatches />
      ) : task === "triptych" ? (
        <Stories groups={result.groups} card={card} />
      ) : (
        result.groups.map((g) => (
          <section key={g.title} className="space-y-4">
            {result.groups.length > 1 && <h2 className="text-lg font-semibold">{g.title}</h2>}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {g.picks.flatMap((p, i) => [
                ...(g.marker && g.marker.before === i
                  ? [<MarkerLine key="marker" probability={g.marker.probability} />]
                  : []),
                card(p),
              ])}
            </div>
            {g.marker && g.marker.before >= g.picks.length && (
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                {MARKER.allAbove(g.marker.probability)}
              </p>
            )}
          </section>
        ))
      )}
      <Diagnostics detail={detail} />
      {viewer !== null && (
        <Lightbox
          runId={result.run_id}
          picks={picks}
          index={viewer}
          onIndex={setViewer}
          onClose={close}
        />
      )}
    </main>
  );
}

type CardFn = (pick: Pick, extra?: Record<string, unknown>) => ReactElement;

function Stories({ groups, card }: { groups: Group[]; card: CardFn }) {
  return (
    <div className="space-y-10">
      {groups.map((group) => (
        <section key={group.title} data-testid="story" className="space-y-4">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            <h2 className="text-lg font-semibold tracking-tight">{group.title}</h2>
            <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-xs text-zinc-600 tabular-nums dark:bg-zinc-800 dark:text-zinc-300">
              {Math.round(group.score * 100)}
            </span>
            <ul className="flex flex-wrap gap-1.5" aria-label="How these photos connect">
              {connectionChips(group.evidence).map((chip) => (
                <li
                  key={chip.label}
                  title={chip.detail || undefined}
                  className="rounded-full border border-accent-500/40 bg-accent-50 px-2.5 py-0.5 text-xs text-accent-700 dark:bg-accent-500/10 dark:text-accent-400"
                >
                  {chip.label} {chip.value}
                </li>
              ))}
            </ul>
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {group.picks.map((pick, i) =>
              card(pick, {
                badge:
                  group.picks.length === 3 && i === 1 ? (
                    <span
                      data-testid="anchor-badge"
                      className="rounded-full bg-accent-600 px-2 py-0.5 text-xs font-medium text-white shadow"
                    >
                      Anchor
                    </span>
                  ) : undefined,
              }),
            )}
          </div>
        </section>
      ))}
    </div>
  );
}

/** Where Yours's list stops being good: it spans the grid, between the picks around it. */
function MarkerLine({ probability }: { probability: number }) {
  return (
    <div
      data-testid="marker"
      role="note"
      className="col-span-full flex items-center gap-3 text-sm font-medium text-amber-700 dark:text-amber-400"
    >
      <span className="h-px flex-1 bg-amber-500/60" aria-hidden />
      <span>{MARKER.line(probability)}</span>
      <span className="h-px flex-1 bg-amber-500/60" aria-hidden />
    </div>
  );
}

function NoMatches() {
  return (
    <p className="rounded-xl border border-dashed border-zinc-300 p-10 text-center text-zinc-500 dark:border-zinc-700">
      {NO_MATCHES}
    </p>
  );
}

const POSSIBLE_ID = "possibly-the-same-view";

/** Same view: the reference, its sure matches oldest first, then the possible ones. */
function Timeline(props: {
  runId: string;
  sure: Group | undefined;
  possible: Group | undefined;
  hasReference: boolean;
  label: string;
  card: CardFn;
  calibrate: ReactElement | null;
}) {
  const { runId, sure, possible, hasReference, label, card, calibrate } = props;
  const surePicks = sure?.picks ?? [];
  const possiblePicks = possible?.picks ?? [];
  return (
    <div className="space-y-8">
      {hasReference && (
        <figure className="flex items-center gap-4 rounded-xl border border-zinc-200 p-3 dark:border-zinc-800">
          <img
            src={referenceUrl(runId, 0)}
            alt="Your reference"
            className="h-24 w-32 rounded-lg object-cover"
          />
          <figcaption>
            <p className="font-medium">Your {label.toLowerCase() || "reference"}</p>
            <p className="text-sm text-zinc-500">
              {SAME_VIEW.summary(surePicks.length, possiblePicks.length)}{" "}
              {possiblePicks.length > 0 && (
                <a
                  href={`#${POSSIBLE_ID}`}
                  className="text-accent-700 underline dark:text-accent-400"
                >
                  {SAME_VIEW.jump}
                </a>
              )}
            </p>
            {calibrate}
          </figcaption>
        </figure>
      )}
      {surePicks.length === 0 && possiblePicks.length === 0 && <NoMatches />}
      {surePicks.length > 0 && (
        <h2 className="text-lg font-semibold tracking-tight">
          {SAME_VIEW.sureHeading(surePicks.length)}
        </h2>
      )}
      {surePicks.length > 0 && (
        <ol className="space-y-8 border-l border-zinc-200 pl-5 dark:border-zinc-800">
          {groupByMonth(surePicks).map((month) => (
            <li key={month.label} data-testid="timeline-month" className="relative space-y-3">
              <span className="absolute top-1.5 -left-[1.6rem] size-2.5 rounded-full bg-accent-500 ring-4 ring-white dark:ring-zinc-950" />
              <h2 className="font-semibold">{month.label}</h2>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                {month.picks.map((p) => card(p, { emphasiseDate: true }))}
              </div>
            </li>
          ))}
        </ol>
      )}
      {possible && possiblePicks.length > 0 && (
        <section id={POSSIBLE_ID} aria-label={possible.title} className="scroll-mt-4 space-y-3">
          <div>
            <h2 className="text-lg font-semibold tracking-tight">
              {SAME_VIEW.possibleHeading(possible.title, possiblePicks.length)}
            </h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">{POSSIBLY_SAME.hint}</p>
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {possible.picks.map((p) => card(p, { emphasiseDate: true }))}
          </div>
        </section>
      )}
    </div>
  );
}

function RunHeader({ detail }: { detail: RunDetail }) {
  const { result } = detail;
  const created = new Date(result.created_at).toLocaleString();
  const photos = result.groups.reduce((n, g) => n + g.picks.length, 0);
  return (
    <header className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">{detail.task_title}</h1>
        <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
          {created} · {photos} photo{photos === 1 ? "" : "s"}
          {result.request.task === "triptych" && ` in ${result.groups.length} stories`}
          {result.request.template === "yours" && ` · ${TEMPLATE.chip}`}
        </p>
      </div>
      {result.request.criteria && (
        <p className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="text-lg text-zinc-700 italic dark:text-zinc-300">
            “{result.request.criteria}”
          </span>
          {result.request.only_clear_matches && (
            <span className="rounded-full bg-emerald-50 px-2.5 py-0.5 text-xs text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-300">
              {ONLY_CLEAR_MATCHES}
            </span>
          )}
          {result.request.describe_weight != null && (
            <span className="rounded-full bg-zinc-100 px-2.5 py-0.5 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
              {describeWeightChip(result.request.describe_weight)}
            </span>
          )}
        </p>
      )}
      {result.request.examples.length > 0 && (
        <div>
          <p className="flex flex-wrap items-baseline gap-x-3 text-sm text-zinc-500 dark:text-zinc-400">
            {EXAMPLES.label}
            {!result.request.criteria && result.request.describe_weight != null && (
              <span className="rounded-full bg-zinc-100 px-2.5 py-0.5 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
                {describeWeightChip(result.request.describe_weight)}
              </span>
            )}
          </p>
          <ul aria-label={EXAMPLES.label} className="mt-2 flex flex-wrap gap-2">
            {result.request.examples.map((path, i) => (
              <li key={path}>
                <img
                  src={thumbUrl(result.example_keys[i] ?? "")}
                  alt={fileName(path)}
                  title={fileName(path)}
                  className="size-16 rounded-lg object-cover ring-1 ring-zinc-200 dark:ring-zinc-800"
                />
              </li>
            ))}
          </ul>
        </div>
      )}
      <ul className="flex flex-wrap gap-2" aria-label="Folders">
        {result.request.folders.map((f) => (
          <li
            key={f}
            title={f}
            className="flex max-w-full items-center gap-1.5 rounded-full bg-zinc-100 px-3 py-1 text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300"
          >
            <Folder className="size-3 shrink-0" /> <span className="truncate">{folderName(f)}</span>
          </li>
        ))}
      </ul>
      {result.notes.map((note) => (
        <p
          key={note}
          role="note"
          className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200"
        >
          <AlertTriangle className="mt-0.5 size-4 shrink-0" /> {note}
        </p>
      ))}
    </header>
  );
}

function Diagnostics({ detail }: { detail: RunDetail }) {
  const { result } = detail;
  return (
    <details className="group rounded-xl border border-zinc-200 text-sm dark:border-zinc-800">
      <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-3 font-medium text-zinc-600 dark:text-zinc-300">
        Diagnostics
        <ChevronDown className="size-4 transition group-open:rotate-180" />
      </summary>
      <div className="space-y-4 overflow-x-auto border-t border-zinc-200 p-4 dark:border-zinc-800">
        <table className="w-full text-left tabular-nums">
          <thead className="text-zinc-500">
            <tr>
              <th className="py-1 font-medium">Stage</th>
              <th className="py-1 text-right font-medium">In</th>
              <th className="py-1 text-right font-medium">Out</th>
              <th className="py-1 text-right font-medium">Seconds</th>
            </tr>
          </thead>
          <tbody>
            {result.stages.map((s) => (
              <tr key={s.name} className="border-t border-zinc-100 dark:border-zinc-800">
                <td className="py-1">{s.name}</td>
                <td className="py-1 text-right">{s.count_in}</td>
                <td className="py-1 text-right">{s.count_out}</td>
                <td className="py-1 text-right">{s.seconds.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
          {Object.entries(result.versions ?? {}).map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-zinc-500">{k}</dt>
              <dd className="font-mono break-all">{v}</dd>
            </div>
          ))}
        </dl>
        <p className="text-xs text-zinc-500">
          Run <code className="font-mono">{result.run_id}</code> · explain any photo with{" "}
          <code className="font-mono">calico runs explain {result.run_id} NAME</code>
        </p>
      </div>
    </details>
  );
}

/** Ratings with optimistic updates: shown at once, reverted if the server refuses. */
/** The run's 👍/👎, as the cached run holds them: a rating shows at once and is still there
 * when you come back. Ratings of one photo are saved in order; one that fails is undone,
 * unless a newer one has replaced it, and says why. */
function useRatings(detail: RunDetail) {
  const client = useQueryClient();
  const runId = detail.result.run_id;
  const [queue] = useState(serialByKey);
  const [error, setError] = useState<string | null>(null);
  const rate = useCallback(
    (key: string, rating: Rating) => {
      const cached = ["run", runId];
      const before = detail.feedback[key];
      const set = (change: (feedback: Record<string, Rating>) => Record<string, Rating>) => {
        void client.cancelQueries({ queryKey: cached });
        client.setQueryData<RunDetail>(cached, (d) => d && { ...d, feedback: change(d.feedback) });
      };
      set((f) => ({ ...f, [key]: rating }));
      queue(key, () => api.feedback(runId, key, rating)).then(
        () => setError(null),
        (err: unknown) => {
          set((f) => {
            if (f[key] !== rating) return f;
            const { [key]: _, ...rest } = f;
            return before === undefined ? rest : { ...rest, [key]: before };
          });
          setError(err instanceof Error ? err.message : String(err));
        },
      );
    },
    [client, runId, detail.feedback, queue],
  );
  return [detail.feedback, rate, error] as const;
}

/** Start a same-view labelling set from this run's reference and folders, then go label it. */
function CalibrateView({ folders, reference }: { folders: string[]; reference: string }) {
  const navigate = useNavigate();
  const [jobId, setJobId] = useState<string | null>(null);
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
  const start = async () => {
    setError(null);
    try {
      setJobId((await api.newLabelSet("", folders, reference)).job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <div className="mt-2 space-y-1">
      <button
        type="button"
        onClick={start}
        disabled={jobId !== null}
        className="flex items-center gap-1.5 rounded-lg border border-zinc-300 px-3 py-1.5 text-sm hover:bg-zinc-100 disabled:opacity-60 dark:border-zinc-700 dark:hover:bg-zinc-800"
      >
        {jobId !== null ? (
          <Loader2 className="size-3.5 animate-spin" aria-hidden />
        ) : (
          <Tags className="size-3.5" aria-hidden />
        )}
        {CALIBRATE.button}
      </button>
      {jobId !== null ? (
        <div className="text-sm">
          <Progress {...phase(job.data, CALIBRATE.choosing)} />
        </div>
      ) : (
        <p className="text-xs text-zinc-500">{CALIBRATE.help}</p>
      )}
      {error && (
        <p role="alert" className="text-sm text-red-600 dark:text-red-400">
          {error}
        </p>
      )}
    </div>
  );
}
