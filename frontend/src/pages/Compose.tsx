import { useQuery } from "@tanstack/react-query";
import { AlertCircle, ChevronDown, Loader2, RotateCcw } from "lucide-react";
import { type FormEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { api, thumbUrl } from "../api/client";
import type { Template } from "../api/types";
import { FolderPicker } from "../components/FolderPicker";
import { OptionsForm } from "../components/OptionsForm";
import { Progress } from "../components/Progress";
import { ReferenceDrop } from "../components/ReferenceDrop";
import { TaskCards } from "../components/TaskCards";
import { TasteChoice } from "../components/TasteChoice";
import { Thumb } from "../components/Thumb";
import {
  DESCRIBE_WEIGHT,
  describeWeightChip,
  EXAMPLES,
  examplesChip,
  fileName,
  ONLY_CLEAR_MATCHES,
  PREFILL,
  percent,
  TEMPLATE,
} from "../lib/format";
import { jobOutcome, phase, useJob } from "../lib/job";
import {
  changedOptions,
  defaultValues,
  parseOptionsSchema,
  type Values,
  validateOptions,
} from "../lib/options";
import { type ArchiveRef, readPrefill } from "../lib/prefill";
import { readingToShow, useDebounced } from "../lib/reading";
import { buildFormData, buildRequest, submitBlocker } from "../lib/request";

const SUGGESTIONS = [
  "light through broken windows",
  "empty corridor",
  "autumn colours",
  "portrait by a window",
];

export function Compose() {
  const location = useLocation();
  const navigate = useNavigate();
  const prefill = useMemo(() => readPrefill(location.state), [location.state]);

  const tasks = useQuery({ queryKey: ["tasks"], queryFn: api.tasks });
  const roots = useQuery({ queryKey: ["roots"], queryFn: api.roots });
  const taste = useQuery({ queryKey: ["taste"], queryFn: api.taste });

  const [taskName, setTaskName] = useState(prefill?.task ?? "");
  const [folders, setFolders] = useState<string[]>(prefill?.folders ?? []);
  const [criteria, setCriteria] = useState(prefill?.criteria ?? "");
  const [examples, setExamples] = useState<ArchiveRef[]>(prefill?.examples ?? []);
  const [onlyClearMatches, setOnlyClearMatches] = useState(prefill?.onlyClearMatches ?? false);
  // null = the task's default
  const [describeWeight, setDescribeWeight] = useState<number | null>(
    prefill?.describeWeight ?? null,
  );
  const [template, setTemplate] = useState<Template | null>(prefill?.template ?? null); // null = the saved default
  const [uploads, setUploads] = useState<File[]>([]);
  const [archiveRefs, setArchiveRefs] = useState<ArchiveRef[]>(prefill?.references ?? []);
  const [limit, setLimit] = useState<number | null>(prefill?.limit ?? null); // null = the task's default
  const [valuesByTask, setValuesByTask] = useState<Record<string, Values>>({});
  const [jobId, setJobId] = useState<string | null>(null);
  // Set at once, so a second click before the server answers starts nothing.
  const sending = useRef(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const settledCriteria = useDebounced(criteria.trim(), 300);
  const reading = useQuery({
    queryKey: ["reading", settledCriteria, folders],
    queryFn: () => api.reading(settledCriteria, folders),
    enabled: settledCriteria !== "",
    staleTime: 60_000,
  });
  const shownReading = settledCriteria === criteria.trim() ? readingToShow(reading.data) : null;

  const task = tasks.data?.find((t) => t.name === taskName) ?? tasks.data?.[0];
  const fields = useMemo(() => (task ? parseOptionsSchema(task.options_schema) : []), [task]);
  // A search started from a result keeps that search's options, over the task's defaults.
  const values = (task && valuesByTask[task.name]) ?? {
    ...defaultValues(fields),
    ...(task && task.name === prefill?.task ? prefill.options : {}),
  };
  const optionErrors = validateOptions(fields, values);
  const usesReference = task !== undefined && task.reference_use !== "unused";
  const described = criteria.trim() !== "" || examples.length > 0;
  const referenceCount = usesReference ? uploads.length + archiveRefs.length : 0;
  const limitValid = limit === null || (Number.isInteger(limit) && limit >= 1 && limit <= 500);

  const job = useJob(jobId);
  useEffect(() => {
    const outcome = jobOutcome(job.data, "run_id");
    if (outcome && "open" in outcome) {
      navigate(`/runs/${outcome.open}`);
    } else if (outcome || job.isError) {
      setError(outcome?.failed ?? job.error?.message ?? "The request failed.");
      setJobId(null);
    }
  }, [job.data, job.isError, job.error, navigate]);

  if (tasks.isError || roots.isError) {
    return (
      <Shell>
        <Problem message={(tasks.error ?? roots.error)?.message ?? "Cannot reach Calico."} />
      </Shell>
    );
  }
  if (!tasks.data || !roots.data || !task) {
    return (
      <Shell>
        <p className="flex items-center gap-2 text-zinc-500">
          <Loader2 className="size-4 animate-spin" /> Loading…
        </p>
      </Shell>
    );
  }
  if (roots.data.length === 0)
    return (
      <Shell>
        <NoCollections />
      </Shell>
    );

  const blocker = submitBlocker({
    folders,
    referenceUse: task.reference_use,
    referenceLabel: task.reference_label,
    referenceCount,
    optionErrors,
    limitValid,
  });
  const busy = jobId !== null || pending;

  const setOption = (key: string, value: unknown) =>
    setValuesByTask((all) => ({ ...all, [task.name]: { ...values, [key]: value } }));

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (blocker || busy || sending.current) return;
    sending.current = true;
    setPending(true);
    setError(null);
    const request = buildRequest({
      task: task.name,
      folders,
      criteria,
      onlyClearMatches,
      describeWeight,
      template: task.uses_taste ? template : null,
      examples: examples.map((e) => e.path),
      limit,
      options: changedOptions(fields, values),
      archiveReferences: usesReference ? archiveRefs.map((r) => r.path) : [],
    });
    try {
      const { job_id } = await api.select(buildFormData(request, usesReference ? uploads : []));
      setJobId(job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      sending.current = false;
      setPending(false);
    }
  };

  // Until Yours is trained the server falls back to General, so that is what is shown.
  const shownTemplate: Template =
    template ?? (taste.data?.trained ? taste.data.default : "general");
  const summary = [
    task.title,
    folders.length ? `${folders.length} folder${folders.length > 1 ? "s" : ""}` : null,
    criteria.trim() ? `“${criteria.trim()}”` : null,
    examples.length ? examplesChip(examples.length) : null,
    described && describeWeight !== null ? describeWeightChip(describeWeight) : null,
    criteria.trim() && onlyClearMatches ? ONLY_CLEAR_MATCHES : null,
    task.uses_taste && shownTemplate === "yours" ? TEMPLATE.chip : null,
    referenceCount
      ? `${task.reference_label.toLowerCase()}: ${
          archiveRefs[0] ? fileName(archiveRefs[0].path) : (uploads[0]?.name ?? "")
        }${referenceCount > 1 ? ` +${referenceCount - 1}` : ""}`
      : null,
  ].filter(Boolean);

  return (
    <Shell>
      {/* Calico checks the options itself (validateOptions) and says why; the browser's
          own check would refuse a value silently inside the closed Fine-tune section. */}
      {prefill?.dropped.length ? (
        <p
          role="note"
          className="mb-6 rounded-xl bg-amber-50 p-4 text-sm text-amber-900 dark:bg-amber-950/40 dark:text-amber-100"
        >
          {PREFILL.dropped(prefill.dropped)}
        </p>
      ) : null}
      <form onSubmit={submit} noValidate className="space-y-10" aria-busy={busy}>
        <header>
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            Choose your photographs
          </h1>
          <p className="mt-1 text-zinc-500 dark:text-zinc-400">
            Pick what you need, where to look, and what you’re after.
          </p>
        </header>

        <Section n={1} title="What do you need?">
          <TaskCards tasks={tasks.data} value={task.name} onChange={setTaskName} />
          {task.uses_taste && taste.data && (
            <TasteChoice
              value={shownTemplate}
              trained={taste.data.trained}
              onChange={setTemplate}
            />
          )}
        </Section>

        <Section n={2} title="Where should Calico look?" hint="Tick one or more folders.">
          <FolderPicker roots={roots.data} ticked={folders} onChange={setFolders} />
        </Section>

        {usesReference && (
          <Section n={3} title={task.reference_label}>
            <ReferenceDrop
              label={task.reference_label}
              help={task.reference_help}
              required={task.reference_use === "required"}
              uploads={uploads}
              onUploads={setUploads}
              archive={archiveRefs}
              onArchive={setArchiveRefs}
            />
          </Section>
        )}

        <Section n={usesReference ? 4 : 3} title="Describe it" hint="Optional — in your own words.">
          <label htmlFor="criteria" className="sr-only">
            What are you looking for?
          </label>
          <textarea
            id="criteria"
            value={criteria}
            onChange={(e) => setCriteria(e.target.value)}
            rows={3}
            placeholder="light through broken windows, autumn, an empty corridor…"
            className="w-full rounded-xl border border-zinc-300 bg-white px-4 py-3 outline-none placeholder:text-zinc-500 focus:border-accent-500 focus:ring-2 focus:ring-accent-500/20 dark:border-zinc-700 dark:bg-zinc-900"
          />
          <div aria-live="polite">
            {shownReading && (
              <p
                data-testid="reading"
                className="mt-2 flex flex-wrap items-center gap-1.5 text-sm text-zinc-600 dark:text-zinc-400"
              >
                <span>Calico reads this as</span>
                {shownReading.map((part) => (
                  <span
                    key={part.kind}
                    className="rounded-full bg-accent-50 px-2.5 py-0.5 text-accent-700 dark:bg-accent-500/15 dark:text-accent-400"
                  >
                    <span className="opacity-70">{part.kind}</span> {part.text}
                  </span>
                ))}
              </p>
            )}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {SUGGESTIONS.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => setCriteria((c) => (c.trim() ? `${c.trim()}, ${example}` : example))}
                className="rounded-full bg-zinc-100 px-3 py-1 text-xs text-zinc-600 hover:bg-zinc-200 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
              >
                + {example}
              </button>
            ))}
          </div>
          {examples.length > 0 && (
            <div className="mt-4">
              <p className="font-medium text-sm">{EXAMPLES.label}</p>
              <ul aria-label={EXAMPLES.label} className="mt-2 flex flex-wrap gap-3">
                {examples.map((example) => (
                  <Thumb
                    key={example.path}
                    src={thumbUrl(example.contentKey)}
                    name={fileName(example.path)}
                    onRemove={() =>
                      setExamples((all) => all.filter((e) => e.path !== example.path))
                    }
                  />
                ))}
              </ul>
              <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">{EXAMPLES.help}</p>
            </div>
          )}
          {described && (
            <div className="mt-4">
              <div className="flex flex-wrap items-baseline justify-between gap-x-3 text-sm">
                <label htmlFor="describe-weight" className="font-medium">
                  {DESCRIBE_WEIGHT.label}
                </label>
                <span className="text-zinc-600 dark:text-zinc-300">
                  <span data-testid="describe-weight" className="tabular-nums">
                    {percent(describeWeight ?? task.default_describe_weight)}
                  </span>
                  {describeWeight !== null && (
                    <button
                      type="button"
                      onClick={() => setDescribeWeight(null)}
                      className="ml-2 text-zinc-500 underline"
                    >
                      default ({percent(task.default_describe_weight)})
                    </button>
                  )}
                </span>
              </div>
              <input
                id="describe-weight"
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={describeWeight ?? task.default_describe_weight}
                onChange={(e) => setDescribeWeight(Number(e.target.value))}
                aria-describedby="describe-weight-help"
                className="mt-2 w-full accent-accent-600"
              />
              <p id="describe-weight-help" className="text-zinc-500 text-xs dark:text-zinc-400">
                {DESCRIBE_WEIGHT.help}
              </p>
            </div>
          )}
          {criteria.trim() && (
            <label className="mt-4 flex cursor-pointer items-start gap-3 text-sm">
              <input
                type="checkbox"
                checked={onlyClearMatches}
                onChange={(e) => setOnlyClearMatches(e.target.checked)}
                className="mt-0.5 size-4 shrink-0 accent-accent-600"
              />
              <span>
                <span className="font-medium">Only clear matches</span>
                <span className="block text-zinc-500 dark:text-zinc-400">
                  Show only photos Calico is confident match your description. If none are, you’ll
                  see the closest, with a note saying so.
                </span>
              </span>
            </label>
          )}
        </Section>

        <Section n={usesReference ? 5 : 4} title="How many?">
          <div className="flex items-center gap-3">
            <label htmlFor="limit" className="sr-only">
              How many results
            </label>
            <input
              id="limit"
              type="number"
              min={1}
              max={500}
              value={limit ?? task.default_limit}
              onChange={(e) =>
                setLimit(e.target.value === "" ? Number.NaN : e.target.valueAsNumber)
              }
              className="w-28 rounded-lg border border-zinc-300 bg-white px-3 py-2 outline-none focus:border-accent-500 focus:ring-2 focus:ring-accent-500/20 dark:border-zinc-700 dark:bg-zinc-900"
            />
            <span className="text-sm text-zinc-500">
              {task.name === "triptych" ? "stories" : "photos at most"}
              {limit !== null && (
                <button type="button" onClick={() => setLimit(null)} className="ml-2 underline">
                  default ({task.default_limit})
                </button>
              )}
            </span>
          </div>
        </Section>

        {fields.length > 0 && (
          <details
            open={Object.keys(optionErrors).length > 0 || undefined}
            className="group rounded-xl border border-zinc-200 dark:border-zinc-800"
          >
            <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-4 py-3 font-medium">
              Fine-tune
              <ChevronDown className="size-4 transition group-open:rotate-180" aria-hidden />
            </summary>
            <div className="space-y-4 border-t border-zinc-200 p-4 dark:border-zinc-800">
              <OptionsForm
                fields={fields}
                values={values}
                errors={optionErrors}
                onChange={setOption}
              />
              <button
                type="button"
                onClick={() =>
                  setValuesByTask((all) => ({ ...all, [task.name]: defaultValues(fields) }))
                }
                className="flex items-center gap-1.5 text-sm text-zinc-500 hover:text-zinc-900 dark:hover:text-white"
              >
                <RotateCcw className="size-3.5" /> Reset to defaults
              </button>
            </div>
          </details>
        )}

        <footer className="sticky bottom-3 z-20 mt-8 rounded-2xl border border-zinc-200 bg-white/90 shadow-lg backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/90">
          <div className="flex flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center">
            <div className="min-w-0 flex-1 text-sm">
              {busy ? (
                <Progress label={phase(job.data).label} fraction={phase(job.data).fraction} />
              ) : error ? (
                <p role="alert" className="flex items-start gap-2 text-red-600 dark:text-red-400">
                  <AlertCircle className="mt-0.5 size-4 shrink-0" /> {error}
                </p>
              ) : (
                <p className="truncate text-zinc-500 dark:text-zinc-400" data-testid="summary">
                  {blocker ?? summary.join(" · ")}
                </p>
              )}
            </div>
            <button
              type="submit"
              disabled={blocker !== null || busy}
              className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-accent-600 px-6 font-medium text-white shadow-sm transition hover:bg-accent-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-500 disabled:cursor-not-allowed disabled:bg-zinc-300 dark:disabled:bg-zinc-700"
            >
              {busy && <Loader2 className="size-4 animate-spin" />}
              Find photographs
            </button>
          </div>
        </footer>
      </form>
    </Shell>
  );
}

function Shell({ children }: { children: ReactNode }) {
  return <main className="mx-auto max-w-3xl px-4 pt-8 pb-8">{children}</main>;
}

function Section({
  n,
  title,
  hint,
  children,
}: {
  n: number;
  title: string;
  hint?: string;
  children: ReactNode;
}) {
  const id = `section-${n}`;
  return (
    <section aria-labelledby={id} className="space-y-4">
      <div className="flex items-baseline gap-3">
        <span className="grid size-6 shrink-0 place-items-center rounded-full bg-zinc-900 text-xs font-semibold text-white dark:bg-zinc-100 dark:text-zinc-900">
          {n}
        </span>
        <h2 id={id} className="text-lg font-semibold tracking-tight">
          {title}
        </h2>
        {hint && <span className="text-sm text-zinc-500 dark:text-zinc-400">{hint}</span>}
      </div>
      {children}
    </section>
  );
}

function Problem({ message }: { message: string }) {
  return (
    <p
      role="alert"
      className="flex items-start gap-2 rounded-xl bg-red-50 p-4 text-red-700 dark:bg-red-950/50 dark:text-red-300"
    >
      <AlertCircle className="mt-0.5 size-4 shrink-0" /> {message}
    </p>
  );
}

function NoCollections() {
  return (
    <div className="rounded-2xl border border-dashed border-zinc-300 p-10 text-center dark:border-zinc-700">
      <h1 className="text-xl font-semibold">Add your photos first</h1>
      <p className="mt-2 text-zinc-500 dark:text-zinc-400">
        Tell Calico where your photographs live, then reload this page:
      </p>
      <pre className="mt-4 overflow-x-auto rounded-lg bg-zinc-100 p-3 text-left text-sm dark:bg-zinc-900">
        uv run calico collections add NAME /path/to/photos
      </pre>
    </div>
  );
}
