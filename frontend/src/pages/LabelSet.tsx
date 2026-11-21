import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Maximize2,
  Trash2,
  X,
} from "lucide-react";
import {
  type KeyboardEvent as ReactKeyboardEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, labelImageUrl, labelReferenceUrl, thumbUrl } from "../api/client";
import type { Label, LabelPhoto, LabelSetDetail, LabelStory } from "../api/types";
import { Preview } from "../components/Preview";
import { ANCHORS, COMPARE, STORIES, TASTE } from "../lib/format";
import { ignoresShortcut } from "../lib/keys";
import {
  type LabelChoice,
  labelChoices,
  labelForKey,
  labelProgress,
  nextUnlabelled,
  storyIs,
  storyMisfits,
  toggledFit,
  withLabel,
  withStory,
} from "../lib/labels";
import { serialByKey } from "../lib/serial";

/** The cached detail of set ``setId``, changed by ``change``: the page shows what the cache
 * holds, so a save shows at once and is still there when you come back. */
function useSetDetail(setId: string) {
  const client = useQueryClient();
  const key = ["label-set", setId];
  return {
    client,
    patch: (change: (detail: LabelSetDetail) => LabelSetDetail) => {
      void client.cancelQueries({ queryKey: key }); // an older fetch must not undo it
      client.setQueryData<LabelSetDetail>(key, (d) => (d ? change(d) : d));
    },
  };
}

export function LabelSet() {
  const { setId = "" } = useParams();
  const detail = useQuery({ queryKey: ["label-set", setId], queryFn: () => api.labelSet(setId) });
  if (detail.isError) {
    return (
      <main className="mx-auto max-w-3xl px-4 py-16">
        <p role="alert" className="text-red-600">
          {detail.error.message}
        </p>
      </main>
    );
  }
  if (!detail.data) {
    return (
      <main className="mx-auto max-w-7xl px-4 py-16">
        <Loader2 className="size-5 animate-spin text-zinc-500" />
      </main>
    );
  }
  return detail.data.kind === "stories" ? (
    <StoryRater detail={detail.data} />
  ) : (
    <Labeller detail={detail.data} />
  );
}

const EVAL_COMMAND: Record<LabelSetDetail["kind"], string> = {
  describe: "calico eval criteria",
  same_view: "calico eval same-view",
  taste: "calico eval taste",
  stories: "calico eval stories",
  compare: "calico eval compare",
  anchor: "calico eval anchors",
};

const TONE: Record<Label, string> = {
  // White text needs 4.5:1: emerald-700 and amber-700 give it, amber-500 gave 2.1.
  2: "bg-emerald-700 text-white border-emerald-700",
  1: "bg-amber-700 text-white border-amber-700",
  0: "bg-zinc-700 text-white border-zinc-700 dark:bg-zinc-200 dark:text-zinc-900 dark:border-zinc-200",
};

function Labeller({ detail }: { detail: LabelSetDetail }) {
  const sameView = detail.kind === "same_view";
  const taste = detail.kind === "taste";
  const compare = detail.kind === "compare";
  const anchor = detail.kind === "anchor";
  const choices = labelChoices(detail.kind);
  const { client, patch } = useSetDetail(detail.set_id);
  const items = detail.items;
  const [queue] = useState(serialByKey);
  const [current, setCurrent] = useState(() => Math.max(0, nextUnlabelled(detail.items, -1)));
  const [viewer, setViewer] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const cards = useRef<(HTMLButtonElement | null)[]>([]);
  const progress = labelProgress(items);

  const save = useCallback(
    (item: LabelPhoto, label: Label | null) => {
      const key = item.content_key;
      patch((d) => ({ ...d, items: withLabel(d.items, key, label) }));
      queue(key, () => api.setLabel(detail.set_id, key, label)).then(
        () => {
          setError(null);
          void client.invalidateQueries({ queryKey: ["label-sets"] });
        },
        (err: unknown) => {
          patch((d) =>
            d.items.find((i) => i.content_key === key)?.label === label
              ? { ...d, items: withLabel(d.items, key, item.label) }
              : d,
          );
          setError(err instanceof Error ? err.message : String(err));
        },
      );
    },
    [detail.set_id, client, patch, queue],
  );

  const moveTo = useCallback(
    (index: number) => {
      setCurrent(index);
      if (!viewer) cards.current[index]?.focus();
    },
    [viewer],
  );

  const labelAndAdvance = useCallback(
    (index: number, label: Label) => {
      const item = items[index];
      if (!item) return;
      save(item, label);
      const next = nextUnlabelled(withLabel(items, item.content_key, label), index);
      if (next >= 0) moveTo(next);
    },
    [items, save, moveTo],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (ignoresShortcut(e)) return;
      const label = labelForKey(e.key);
      if (label !== undefined) {
        e.preventDefault();
        labelAndAdvance(current, label);
      } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const step = e.key === "ArrowRight" ? 1 : -1;
        moveTo((current + step + items.length) % items.length);
      } else if (e.key === "Escape" && viewer) {
        setViewer(false);
        cards.current[current]?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [current, items.length, labelAndAdvance, moveTo, viewer]);

  const choose = (index: number, label: Label) => {
    const item = items[index];
    if (!item) return;
    setCurrent(index);
    save(item, item.label === label ? null : label);
  };

  return (
    <main className="mx-auto max-w-7xl space-y-6 px-4 py-8">
      <SetHeader detail={detail} progress={progress.text} onError={setError} />

      <details
        open={progress.labelled === 0}
        className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 text-sm dark:border-zinc-800 dark:bg-zinc-900"
      >
        <summary className="cursor-pointer font-medium">How to label</summary>
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-zinc-700 dark:text-zinc-300">
          {compare ? (
            <>
              <li>{COMPARE.judge}</li>
              <li>{COMPARE.choices}</li>
            </>
          ) : anchor ? (
            <>
              <li>{ANCHORS.judge}</li>
              <li>{ANCHORS.choices}</li>
            </>
          ) : taste ? (
            <>
              <li>{TASTE.judge}</li>
              <li>{TASTE.choices}</li>
            </>
          ) : sameView ? (
            <>
              <li>
                Judge only whether the photo shows the same view as the reference, whatever the
                light, season or weather.
              </li>
              <li>
                <strong>Same view</strong>: the same spot from about the same place and angle.{" "}
                <strong>Nearly</strong>: the same place, but the camera moved or turned noticeably.{" "}
                <strong>No</strong>: another view.
              </li>
            </>
          ) : (
            <>
              <li>
                Judge only whether the photo shows what the words describe, not whether it is a good
                photo.
              </li>
              <li>
                <strong>Matches</strong>: clearly what you described. <strong>Partly</strong>: some
                of it (fog but no water), or hard to tell. <strong>No</strong>: not what you
                described.
              </li>
            </>
          )}
          <li>
            Keys: <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> (or <kbd>Y</kbd> <kbd>P</kbd> <kbd>N</kbd>)
            label a photo and move to the next; <kbd>←</kbd> <kbd>→</kbd> move; <kbd>Enter</kbd> on
            a photo enlarges it. Every label is saved as you go.
          </li>
        </ul>
      </details>

      {sameView && (
        <figure className="z-20 flex items-center gap-4 rounded-xl border border-zinc-200 bg-white/95 p-3 backdrop-blur sm:sticky sm:top-16 dark:border-zinc-800 dark:bg-zinc-950/95">
          <img
            src={labelReferenceUrl(detail.set_id)}
            alt="The reference view"
            className="size-28 shrink-0 rounded-lg object-cover sm:size-36"
          />
          <figcaption className="text-sm text-zinc-600 dark:text-zinc-400">
            Is each photo below the same view as this one?
          </figcaption>
        </figure>
      )}
      {error && <ErrorLine error={error} />}
      {progress.done && <Done kind={detail.kind} total={progress.total} />}

      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
        {items.map((item, index) => (
          <li
            key={item.content_key}
            data-testid="label-card"
            className={`overflow-hidden rounded-xl border bg-white dark:bg-zinc-900 ${
              index === current
                ? "border-accent-500 ring-2 ring-accent-500/40"
                : "border-zinc-200 dark:border-zinc-800"
            }`}
          >
            <button
              type="button"
              ref={(el) => {
                cards.current[index] = el;
              }}
              onFocus={() => setCurrent(index)}
              onClick={() => {
                setCurrent(index);
                setViewer(true);
              }}
              aria-label={`Enlarge ${item.name}`}
              className="group relative block aspect-[4/3] w-full bg-zinc-100 focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent-500 dark:bg-zinc-800"
            >
              <Preview
                src={thumbUrl(item.content_key)}
                alt=""
                loading="lazy"
                className="size-full object-cover"
              />
              <Maximize2
                className="absolute top-2 right-2 size-4 text-white opacity-0 drop-shadow group-hover:opacity-100"
                aria-hidden
              />
            </button>
            <Choices
              item={item}
              choices={choices}
              onChoose={(label) => choose(index, label)}
              className="grid grid-cols-3 gap-1 p-1.5"
            />
          </li>
        ))}
      </ul>

      {viewer && items[current] && (
        <Viewer
          setId={detail.set_id}
          item={items[current]}
          choices={choices}
          reference={sameView ? labelReferenceUrl(detail.set_id) : null}
          position={`${current + 1} of ${items.length}`}
          onChoose={(label) => choose(current, label)}
          onMove={(step) => moveTo((current + step + items.length) % items.length)}
          onClose={() => {
            setViewer(false);
            cards.current[current]?.focus();
          }}
        />
      )}
    </main>
  );
}

function SetHeader({
  detail,
  progress,
  onError,
}: {
  detail: LabelSetDetail;
  progress: string;
  onError: (message: string) => void;
}) {
  const navigate = useNavigate();
  const client = useQueryClient();
  const remove = async () => {
    if (!window.confirm(`Delete the labels for “${detail.text}”?`)) return;
    try {
      await api.deleteLabelSet(detail.set_id);
      await client.invalidateQueries({ queryKey: ["label-sets"] });
      navigate("/label");
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0 space-y-1">
        <Link to="/label" className="text-sm text-zinc-500 hover:underline">
          ← Label searches
        </Link>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
          {detail.kind === "describe" ? `“${detail.text}”` : detail.text}
        </h1>
        <p className="text-sm text-zinc-500" data-testid="label-progress">
          {progress}
        </p>
      </div>
      <button
        type="button"
        onClick={remove}
        className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm text-zinc-500 hover:bg-zinc-100 hover:text-red-600 dark:hover:bg-zinc-800"
      >
        <Trash2 className="size-4" aria-hidden /> Delete
      </button>
    </header>
  );
}

function ErrorLine({ error }: { error: string }) {
  return (
    <p role="alert" className="flex items-start gap-2 text-red-600 dark:text-red-400">
      <AlertCircle className="mt-0.5 size-4 shrink-0" /> {error}
    </p>
  );
}

function Done({ kind, total }: { kind: LabelSetDetail["kind"]; total: number }) {
  return (
    <p
      role="status"
      className="rounded-xl bg-emerald-50 p-4 text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-200"
    >
      All {total} labelled. Thank you. These labels now count in <code>{EVAL_COMMAND[kind]}</code>.{" "}
      <Link to="/label" className="font-medium underline">
        Label another search
      </Link>
    </p>
  );
}

function StoryRater({ detail }: { detail: LabelSetDetail }) {
  const choices = labelChoices("stories");
  const { client, patch } = useSetDetail(detail.set_id);
  const stories = detail.stories;
  const [queue] = useState(serialByKey);
  const [current, setCurrent] = useState(() => Math.max(0, nextUnlabelled(detail.stories, -1)));
  const [error, setError] = useState<string | null>(null);
  const cards = useRef<(HTMLElement | null)[]>([]);
  const progress = labelProgress(stories);

  const save = useCallback(
    (before: LabelStory, label: Label | null, doesntFit: string[]) => {
      const at = before.index;
      patch((d) => ({ ...d, stories: withStory(d.stories, at, label, doesntFit) }));
      queue(String(at), () => api.setStory(detail.set_id, at, label, doesntFit)).then(
        () => {
          setError(null);
          void client.invalidateQueries({ queryKey: ["label-sets"] });
        },
        (err: unknown) => {
          patch((d) =>
            storyIs(d.stories, at, label, doesntFit)
              ? { ...d, stories: withStory(d.stories, at, before.label, storyMisfits(before)) }
              : d,
          );
          setError(err instanceof Error ? err.message : String(err));
        },
      );
    },
    [detail.set_id, client, patch, queue],
  );

  const moveTo = useCallback((index: number) => {
    setCurrent(index);
    cards.current[index]?.focus();
  }, []);

  const rateAndAdvance = useCallback(
    (index: number, label: Label) => {
      const story = stories[index];
      if (!story) return;
      const misfits = storyMisfits(story);
      save(story, label, misfits);
      const next = nextUnlabelled(withStory(stories, story.index, label, misfits), index);
      if (next >= 0) moveTo(next);
    },
    [stories, save, moveTo],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (ignoresShortcut(e)) return;
      const label = labelForKey(e.key);
      if (label !== undefined) {
        e.preventDefault();
        rateAndAdvance(current, label);
      } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const step = e.key === "ArrowRight" ? 1 : -1;
        moveTo((current + step + stories.length) % stories.length);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [current, stories.length, rateAndAdvance, moveTo]);

  const choose = (index: number, label: Label) => {
    const story = stories[index];
    if (!story) return;
    setCurrent(index);
    save(story, story.label === label ? null : label, storyMisfits(story));
  };

  const toggleFit = (index: number, contentKey: string) => {
    const story = stories[index];
    if (!story) return;
    setCurrent(index);
    save(story, story.label, toggledFit(story, contentKey));
  };

  return (
    <main className="mx-auto max-w-5xl space-y-6 px-4 py-8">
      <SetHeader detail={detail} progress={progress.text} onError={setError} />
      <details
        open={progress.labelled === 0}
        className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 text-sm dark:border-zinc-800 dark:bg-zinc-900"
      >
        <summary className="cursor-pointer font-medium">How to label</summary>
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-zinc-700 dark:text-zinc-300">
          <li>{STORIES.judge}</li>
          <li>{STORIES.choices}</li>
          <li>{STORIES.doesntFit}</li>
          <li>
            Keys: <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> (or <kbd>Y</kbd> <kbd>P</kbd> <kbd>N</kbd>)
            rate a story and move to the next; <kbd>←</kbd> <kbd>→</kbd> move. A photo opens full
            size in a new tab. Every rating is saved as you go.
          </li>
        </ul>
      </details>
      {error && <ErrorLine error={error} />}
      {progress.done && <Done kind={detail.kind} total={progress.total} />}

      <ul className="space-y-4">
        {stories.map((story, index) => (
          <li key={story.index}>
            {/* biome-ignore lint/a11y/useSemanticElements: a fieldset would add a legend we do not want */}
            <section
              ref={(el) => {
                cards.current[index] = el;
              }}
              tabIndex={-1}
              onFocus={() => setCurrent(index)}
              aria-label={STORIES.position(index + 1, stories.length)}
              role="group"
              data-testid="story-card"
              className={`overflow-hidden rounded-xl border bg-white focus-visible:outline-2 focus-visible:outline-accent-500 dark:bg-zinc-900 ${
                index === current
                  ? "border-accent-500 ring-2 ring-accent-500/40"
                  : "border-zinc-200 dark:border-zinc-800"
              }`}
            >
              <div className="grid grid-cols-3 gap-1.5 p-1.5">
                {story.photos.map((photo) => (
                  <figure key={photo.content_key} className="min-w-0 space-y-1">
                    <a
                      href={labelImageUrl(detail.set_id, photo.content_key)}
                      target="_blank"
                      rel="noreferrer"
                      aria-label={`Open ${photo.name}`}
                      className={`block aspect-[4/3] overflow-hidden rounded-lg bg-zinc-100 focus-visible:outline-2 focus-visible:outline-accent-500 dark:bg-zinc-800 ${
                        photo.fits ? "" : "opacity-50"
                      }`}
                    >
                      <Preview
                        src={thumbUrl(photo.content_key)}
                        alt=""
                        loading="lazy"
                        className="size-full object-cover"
                      />
                    </a>
                    <button
                      type="button"
                      aria-pressed={!photo.fits}
                      aria-label={`${STORIES.misfit}: ${photo.name}`}
                      onClick={() => toggleFit(index, photo.content_key)}
                      className={`w-full rounded-lg border px-0.5 py-1 text-xs font-medium tracking-tight transition focus-visible:outline-2 focus-visible:outline-accent-500 ${
                        photo.fits
                          ? "border-zinc-200 text-zinc-500 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800"
                          : "border-red-600 bg-red-600 text-white"
                      }`}
                    >
                      {STORIES.misfit}
                    </button>
                  </figure>
                ))}
              </div>
              {/* biome-ignore lint/a11y/useSemanticElements: a fieldset would add a legend we do not want */}
              <div
                role="group"
                aria-label={`Rate ${STORIES.story(index + 1)}`}
                className="grid grid-cols-3 gap-1 border-t border-zinc-200 p-1.5 dark:border-zinc-800"
              >
                {choices.map((choice) => {
                  const on = story.label === choice.value;
                  const name = `${choice.name}: ${STORIES.story(index + 1)}`;
                  return (
                    <button
                      key={choice.value}
                      type="button"
                      aria-pressed={on}
                      aria-label={name}
                      title={`${choice.name} (${choice.key})`}
                      onClick={() => choose(index, choice.value)}
                      className={`rounded-lg border px-0.5 py-2 text-sm font-medium tracking-tight transition focus-visible:outline-2 focus-visible:outline-accent-500 ${
                        on
                          ? TONE[choice.value]
                          : "border-zinc-200 text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
                      }`}
                    >
                      {choice.name}
                    </button>
                  );
                })}
              </div>
            </section>
          </li>
        ))}
      </ul>
    </main>
  );
}

function Choices({
  item,
  choices,
  onChoose,
  className,
}: {
  item: LabelPhoto;
  choices: LabelChoice[];
  onChoose: (label: Label) => void;
  className: string;
}) {
  return (
    // biome-ignore lint/a11y/useSemanticElements: a fieldset would add a legend we do not want
    <div role="group" aria-label={`Label ${item.name}`} className={className}>
      {choices.map((choice) => {
        const on = item.label === choice.value;
        return (
          <button
            key={choice.value}
            type="button"
            aria-pressed={on}
            aria-label={`${choice.name}: ${item.name}`}
            title={`${choice.name} (${choice.key})`}
            onClick={() => onChoose(choice.value)}
            className={`rounded-lg border px-0.5 py-1.5 text-xs font-medium tracking-tight transition focus-visible:outline-2 focus-visible:outline-accent-500 ${
              on
                ? TONE[choice.value]
                : "border-zinc-200 text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
            }`}
          >
            {choice.name}
          </button>
        );
      })}
    </div>
  );
}

function Viewer({
  setId,
  item,
  choices,
  reference,
  position,
  onChoose,
  onMove,
  onClose,
}: {
  setId: string;
  item: LabelPhoto;
  choices: LabelChoice[];
  reference: string | null;
  position: string;
  onChoose: (label: Label) => void;
  onMove: (step: number) => void;
  onClose: () => void;
}) {
  const close = useRef<HTMLButtonElement>(null);
  const dialog = useRef<HTMLDivElement>(null);
  useEffect(() => close.current?.focus(), []);
  const trapTab = (e: ReactKeyboardEvent) => {
    if (e.key !== "Tab" || !dialog.current) return;
    const focusable = dialog.current.querySelectorAll<HTMLElement>("button");
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last?.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first?.focus();
    }
  };
  const nav =
    "grid size-11 place-items-center rounded-full bg-white/10 text-white hover:bg-white/20 focus-visible:outline-2 focus-visible:outline-white";
  return (
    <div
      ref={dialog}
      role="dialog"
      aria-modal="true"
      aria-label={item.name}
      data-testid="label-viewer"
      onKeyDown={trapTab}
      className="fixed inset-0 z-50 flex flex-col bg-zinc-950/95 text-white"
    >
      <div className="flex items-center justify-between gap-4 p-3">
        <p className="min-w-0 truncate text-sm text-zinc-300">
          {item.folder} / {item.name} · {position}
        </p>
        <button ref={close} type="button" onClick={onClose} aria-label="Close" className={nav}>
          <X className="size-5" />
        </button>
      </div>
      <div className="flex min-h-0 flex-1 items-center gap-2 px-2">
        <button type="button" onClick={() => onMove(-1)} aria-label="Previous" className={nav}>
          <ChevronLeft className="size-5" />
        </button>
        {reference && (
          <figure className="hidden max-h-full w-1/3 shrink-0 flex-col gap-1 md:flex">
            <img
              src={reference}
              alt="The reference view"
              className="max-h-[70vh] w-full object-contain"
            />
            <figcaption className="text-center text-xs text-zinc-300">Reference</figcaption>
          </figure>
        )}
        <Preview
          src={labelImageUrl(setId, item.content_key)}
          alt={item.name}
          className="mx-auto max-h-full min-w-0 flex-1 object-contain"
        />
        <button type="button" onClick={() => onMove(1)} aria-label="Next" className={nav}>
          <ChevronRight className="size-5" />
        </button>
      </div>
      <Choices
        item={item}
        choices={choices}
        onChoose={onChoose}
        className="mx-auto grid w-full max-w-md grid-cols-3 gap-2 p-4"
      />
    </div>
  );
}
