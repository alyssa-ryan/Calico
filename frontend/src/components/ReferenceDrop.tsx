import { ImagePlus } from "lucide-react";
import { useEffect, useId, useMemo, useState } from "react";
import { thumbUrl } from "../api/client";
import { fileName } from "../lib/format";
import type { ArchiveRef } from "../lib/prefill";
import { Thumb } from "./Thumb";

type Props = {
  label: string;
  help: string;
  required: boolean;
  uploads: readonly File[];
  onUploads: (files: File[]) => void;
  archive: readonly ArchiveRef[];
  onArchive: (refs: ArchiveRef[]) => void;
};

/** Reference / anchor photos: dropped or browsed uploads, plus archive photos chosen from results. */
export function ReferenceDrop({
  label,
  help,
  required,
  uploads,
  onUploads,
  archive,
  onArchive,
}: Props) {
  const inputId = useId();
  const [dragging, setDragging] = useState(false);
  const previews = useMemo(() => uploads.map((f) => URL.createObjectURL(f)), [uploads]);
  useEffect(
    () => () => {
      for (const url of previews) URL.revokeObjectURL(url);
    },
    [previews],
  );

  const add = (files: FileList | null) => {
    const images = Array.from(files ?? []).filter(
      (f) => f.type.startsWith("image/") || f.type === "",
    );
    if (images.length) onUploads([...uploads, ...images]);
  };

  const count = uploads.length + archive.length;
  return (
    <div className="space-y-3">
      <p className="flex flex-wrap items-baseline gap-2 text-sm text-zinc-500 dark:text-zinc-400">
        <span
          className={`rounded-full px-2 py-0.5 text-xs font-medium ${
            required
              ? "bg-amber-100 text-amber-800 dark:bg-amber-500/15 dark:text-amber-300"
              : "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400"
          }`}
        >
          {required ? "required" : "optional"}
        </span>
        {help}
      </p>
      {count > 0 && (
        <ul aria-label={`${label}s`} className="flex flex-wrap gap-3">
          {archive.map((ref) => (
            <Thumb
              key={ref.path}
              src={thumbUrl(ref.contentKey)}
              name={fileName(ref.path)}
              onRemove={() => onArchive(archive.filter((r) => r.path !== ref.path))}
            />
          ))}
          {uploads.map((file, i) => (
            <Thumb
              key={`${file.name}:${file.size}:${file.lastModified}`}
              src={previews[i] ?? ""}
              name={file.name}
              onRemove={() => onUploads(uploads.filter((_, j) => j !== i))}
            />
          ))}
        </ul>
      )}
      <label
        htmlFor={inputId}
        data-testid="reference-drop"
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          add(e.dataTransfer.files);
        }}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-8 text-center text-sm transition has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent-500 ${
          dragging
            ? "border-accent-500 bg-accent-50 dark:bg-accent-500/10"
            : "border-zinc-300 text-zinc-500 hover:border-zinc-400 dark:border-zinc-700 dark:text-zinc-400"
        }`}
      >
        <ImagePlus className="size-6 text-zinc-500" aria-hidden />
        <span>
          <span className="font-medium text-accent-600 dark:text-accent-400">Choose a photo</span>{" "}
          or drop it here
        </span>
        <input
          id={inputId}
          type="file"
          accept="image/*,.heic,.heif"
          multiple
          className="sr-only"
          aria-label={`Add ${label.toLowerCase()}`}
          onChange={(e) => {
            add(e.target.files);
            e.target.value = "";
          }}
        />
      </label>
    </div>
  );
}
