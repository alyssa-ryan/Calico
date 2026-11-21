/** A job's progress: what it is doing, and a bar when it can tell how far it has got. */
export function Progress({ label, fraction }: { label: string; fraction: number | null }) {
  return (
    <div role="status" className="space-y-1.5">
      <p className="text-zinc-700 dark:text-zinc-300">{label}</p>
      <div className="h-1.5 overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
        <div
          className={`h-full rounded-full bg-accent-500 transition-[width] ${fraction === null ? "w-1/3 animate-pulse" : ""}`}
          style={fraction === null ? undefined : { width: `${Math.round(fraction * 100)}%` }}
        />
      </div>
    </div>
  );
}
