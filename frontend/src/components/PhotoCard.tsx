import {
  CircleCheck,
  GalleryHorizontal,
  History,
  Images,
  Layers,
  MoreHorizontal,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";

import { type ReactNode, useEffect, useRef, useState } from "react";
import { thumbUrl } from "../api/client";
import type { Pick, Rating } from "../api/types";
import {
  clearMatchBadge,
  EXAMPLES,
  fileName,
  formatDate,
  LIKELY,
  likelyBadge,
  moreFrames,
  percent,
  signalLabel,
} from "../lib/format";
import { Preview } from "./Preview";

export type CardActions = {
  story: (pick: Pick) => void;
  sameView: (pick: Pick) => void;
  /** Absent where it would not help (a same-view timeline already shows every match). */
  moreLike?: (pick: Pick) => void;
};

type Props = {
  pick: Pick;
  rating: Rating | undefined;
  onRate: (rating: Rating) => void;
  onOpen: (opener: HTMLElement) => void;
  actions: CardActions;
  badge?: ReactNode;
  emphasiseDate?: boolean;
};

export function PhotoCard({ pick, rating, onRate, onOpen, actions, badge, emphasiseDate }: Props) {
  const name = fileName(pick.path);
  const clear = clearMatchBadge(pick);
  const likely = likelyBadge(pick.probability);
  return (
    <article
      data-testid="pick-card"
      className="group flex min-w-0 flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900"
    >
      <button
        type="button"
        onClick={(e) => onOpen(e.currentTarget)}
        aria-label={`Open ${name}`}
        className="relative block overflow-hidden bg-zinc-100 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent-500 dark:bg-zinc-800"
      >
        <Preview
          src={thumbUrl(pick.content_key)}
          alt={`${name} — ${pick.location || "unknown location"}, ${formatDate(pick.captured_at)}`}
          loading="lazy"
          fade
          className="aspect-[4/3] w-full object-cover transition duration-500 group-hover:scale-[1.02]"
        />
        {badge && <span className="absolute top-2 left-2">{badge}</span>}
        <span className="absolute top-2 right-2 rounded-full bg-black/60 px-2 py-0.5 text-xs font-medium text-white tabular-nums backdrop-blur">
          {Math.round(pick.score * 100)}
        </span>
        {pick.burst_size > 1 && (
          <span className="absolute bottom-2 left-2 flex items-center gap-1 rounded-full bg-black/60 px-2 py-0.5 text-xs text-white backdrop-blur">
            <Layers className="size-3" /> {moreFrames(pick.burst_size - 1)}
          </span>
        )}
        {clear && (
          <span className="absolute right-2 bottom-2 flex items-center gap-1 rounded-full bg-emerald-600/90 px-2 py-0.5 text-xs font-medium text-white backdrop-blur">
            <CircleCheck className="size-3" aria-hidden /> {clear}
          </span>
        )}
      </button>
      <div className="flex flex-1 flex-col gap-3 p-3">
        <div className="min-w-0">
          {emphasiseDate && <p className="text-sm font-semibold">{formatDate(pick.captured_at)}</p>}
          <p className="truncate text-sm font-medium" title={pick.path}>
            {name}
          </p>
          <p className="truncate text-xs text-zinc-500 dark:text-zinc-400">
            {[pick.location, emphasiseDate ? null : formatDate(pick.captured_at)]
              .filter(Boolean)
              .join(" · ")}
          </p>
          {likely && (
            <p
              className="mt-0.5 text-xs font-medium text-emerald-700 tabular-nums dark:text-emerald-400"
              title={LIKELY.help}
            >
              {likely}
            </p>
          )}
        </div>
        <EvidenceBars evidence={pick.evidence} />
        <div className="mt-auto flex items-center gap-1">
          <RateButton kind="up" active={rating === 1} onClick={() => onRate(1)} name={name} />
          <RateButton kind="down" active={rating === -1} onClick={() => onRate(-1)} name={name} />
          <CardMenu pick={pick} actions={actions} />
        </div>
      </div>
    </article>
  );
}

export function EvidenceBars({ evidence }: { evidence: Pick["evidence"] }) {
  if (!evidence.length) return null;
  return (
    <ul className="space-y-1.5" aria-label="Why it was picked">
      {evidence.map((e) => (
        <li
          key={e.signal}
          className="grid grid-cols-[5.5rem_1fr_2.5rem] items-center gap-2 text-xs"
          title={e.detail || undefined}
        >
          <span className="truncate text-zinc-500 dark:text-zinc-400">{signalLabel(e.signal)}</span>
          <span className="h-1.5 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
            <span
              className="block h-full rounded-full bg-accent-500"
              style={{ width: percent(e.value) }}
            />
          </span>
          <span className="text-right text-zinc-500 tabular-nums dark:text-zinc-400">
            {percent(e.value)}
          </span>
        </li>
      ))}
    </ul>
  );
}

function RateButton({
  kind,
  active,
  onClick,
  name,
}: {
  kind: "up" | "down";
  active: boolean;
  onClick: () => void;
  name: string;
}) {
  const Icon = kind === "up" ? ThumbsUp : ThumbsDown;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      aria-label={`${kind === "up" ? "Keep" : "Reject"} ${name}`}
      className={`grid size-9 place-items-center rounded-lg transition focus-visible:outline-2 focus-visible:outline-accent-500 ${
        active
          ? kind === "up"
            ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-300"
            : "bg-rose-100 text-rose-700 dark:bg-rose-500/20 dark:text-rose-300"
          : "text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
      }`}
    >
      <Icon className="size-4" />
    </button>
  );
}

function CardMenu({ pick, actions }: { pick: Pick; actions: CardActions }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (
        e instanceof KeyboardEvent ? e.key === "Escape" : !ref.current?.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);

  const item =
    "flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm hover:bg-zinc-100 dark:hover:bg-zinc-800";
  return (
    <div ref={ref} className="relative ml-auto">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`More actions for ${fileName(pick.path)}`}
        className="grid size-9 place-items-center rounded-lg text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
      >
        <MoreHorizontal className="size-4" />
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 bottom-full z-10 mb-1 w-60 rounded-xl border border-zinc-200 bg-white p-1 shadow-lg dark:border-zinc-800 dark:bg-zinc-900"
        >
          <button
            type="button"
            role="menuitem"
            className={item}
            onClick={() => actions.story(pick)}
          >
            <GalleryHorizontal className="size-4 text-zinc-500" /> Build a story around this
          </button>
          <button
            type="button"
            role="menuitem"
            className={item}
            onClick={() => actions.sameView(pick)}
          >
            <History className="size-4 text-zinc-500" /> Find this view over time
          </button>
          {actions.moreLike && (
            <button
              type="button"
              role="menuitem"
              className={item}
              onClick={() => actions.moreLike?.(pick)}
            >
              <Images className="size-4 text-zinc-500" /> {EXAMPLES.moreLikeThis}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
