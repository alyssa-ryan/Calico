import { X } from "lucide-react";

/** A chosen photo (a reference, an anchor, an example) with a button to take it back out. */
export function Thumb({
  src,
  name,
  onRemove,
}: {
  src: string;
  name: string;
  onRemove: () => void;
}) {
  return (
    <li className="relative">
      <img
        src={src}
        alt={name}
        className="size-24 rounded-lg object-cover ring-1 ring-zinc-200 dark:ring-zinc-800"
      />
      <button
        type="button"
        onClick={onRemove}
        aria-label={`Remove ${name}`}
        className="absolute -top-2 -right-2 grid size-6 place-items-center rounded-full bg-zinc-900 text-white shadow ring-2 ring-white dark:bg-zinc-100 dark:text-zinc-900 dark:ring-zinc-950"
      >
        <X className="size-3.5" />
      </button>
      <p className="mt-1 max-w-24 truncate text-xs text-zinc-500" title={name}>
        {name}
      </p>
    </li>
  );
}
