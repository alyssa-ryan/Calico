import { useState } from "react";
import { PREVIEW } from "../lib/format";

type Props = {
  src: string;
  alt: string;
  className?: string;
  loading?: "lazy" | "eager";
  /** Fade in once loaded, so a card does not flash an empty frame. */
  fade?: boolean;
};

/**
 * A photo, or, when it cannot be loaded (a thumbnail not made yet, a drive not mounted), a
 * tile saying so: an image that never loads would leave a card blank, or invisible while
 * it fades in. Each ``src`` is tried afresh, as when the lightbox moves to the next photo.
 */
export function Preview({ src, alt, className = "", loading, fade = false }: Props) {
  const [loaded, setLoaded] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  if (failed === src) {
    return (
      <span
        role="img"
        aria-label={alt || PREVIEW.unavailable}
        className={`${className} flex items-center justify-center bg-zinc-100 p-2 text-center text-xs text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300`}
      >
        {PREVIEW.unavailable}
      </span>
    );
  }
  const shown = !fade || loaded === src;
  return (
    <img
      src={src}
      alt={alt}
      loading={loading}
      onLoad={() => setLoaded(src)}
      onError={() => setFailed(src)}
      className={`${className} ${fade ? (shown ? "opacity-100" : "opacity-0") : ""}`}
    />
  );
}
