import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useEffect, useRef } from "react";
import { imageUrl } from "../api/client";
import type { Pick } from "../api/types";
import { fileName, formatDate } from "../lib/format";
import { EvidenceBars } from "./PhotoCard";
import { Preview } from "./Preview";

type Props = {
  runId: string;
  picks: readonly Pick[];
  index: number;
  onIndex: (index: number) => void;
  onClose: () => void;
};

/** Full-screen viewer: ←/→ move through the run, Esc closes, focus stays inside while open. */
export function Lightbox({ runId, picks, index, onIndex, onClose }: Props) {
  const dialog = useRef<HTMLDivElement>(null);
  const pick = picks[index];
  const count = picks.length;

  useEffect(() => {
    dialog.current?.querySelector<HTMLElement>("[data-autofocus]")?.focus();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight") onIndex((index + 1) % count);
      else if (e.key === "ArrowLeft") onIndex((index - 1 + count) % count);
      else if (e.key === "Tab" && dialog.current) {
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
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [index, count, onIndex, onClose]);

  if (!pick) return null;
  const nav =
    "grid size-11 place-items-center rounded-full bg-white/10 text-white hover:bg-white/20 focus-visible:outline-2 focus-visible:outline-white";
  return (
    <div
      ref={dialog}
      role="dialog"
      aria-modal="true"
      aria-label={fileName(pick.path)}
      data-testid="lightbox"
      className="fixed inset-0 z-50 flex flex-col bg-zinc-950/95 text-white lg:flex-row"
    >
      <div className="relative flex min-h-0 flex-1 items-center justify-center p-4">
        <Preview
          src={imageUrl(runId, pick.content_key)}
          alt={`${fileName(pick.path)} — ${pick.location}, ${formatDate(pick.captured_at)}`}
          className="max-h-full max-w-full rounded-lg object-contain"
        />
        {count > 1 && (
          <>
            <button
              type="button"
              aria-label="Previous photo"
              onClick={() => onIndex((index - 1 + count) % count)}
              className={`${nav} absolute left-4`}
            >
              <ChevronLeft className="size-5" />
            </button>
            <button
              type="button"
              aria-label="Next photo"
              onClick={() => onIndex((index + 1) % count)}
              className={`${nav} absolute right-4`}
            >
              <ChevronRight className="size-5" />
            </button>
          </>
        )}
      </div>
      <aside className="w-full shrink-0 space-y-4 overflow-y-auto border-t border-white/10 p-5 lg:w-80 lg:border-t-0 lg:border-l">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate font-medium" title={pick.path}>
              {fileName(pick.path)}
            </p>
            <p className="text-sm text-zinc-300">
              {pick.location} · {formatDate(pick.captured_at)}
            </p>
            <p className="mt-1 text-xs text-zinc-300">
              {index + 1} of {count} · score {Math.round(pick.score * 100)}
            </p>
          </div>
          <button type="button" data-autofocus onClick={onClose} aria-label="Close" className={nav}>
            <X className="size-5" />
          </button>
        </div>
        <div className="[&_*]:text-zinc-300">
          <EvidenceBars evidence={pick.evidence} />
        </div>
        <p className="break-all text-xs text-zinc-300">{pick.path}</p>
      </aside>
    </div>
  );
}
