import { useQuery } from "@tanstack/react-query";
import { ChevronRight, Folder, HardDrive, Loader2, Search, X } from "lucide-react";
import { useState } from "react";
import { api } from "../api/client";
import type { RootFolder } from "../api/types";
import {
  allTicked,
  coveredBy,
  folderName,
  isInside,
  tickAll,
  toggleFolder,
  untickAll,
} from "../lib/folders";
import { FOLDER_SEARCH } from "../lib/format";
import { useDebounced } from "../lib/reading";

type Props = {
  roots: readonly RootFolder[];
  ticked: readonly string[];
  onChange: (ticked: string[]) => void;
};

/**
 * The folder picker: collection roots, lazily expanded. Ticking a folder covers
 * everything below it, so its loaded children are shown ticked and locked.
 * Typing searches folder names in every collection instead, and the matches can
 * be ticked one by one or all at once ("state hospital" ticks every hospital).
 */
export function FolderPicker({ roots, ticked, onChange }: Props) {
  const [filter, setFilter] = useState("");
  const toggle = (path: string) => onChange(toggleFolder(ticked, path));
  const query = useDebounced(filter.trim(), 250);

  return (
    <div className="space-y-3">
      {ticked.length > 0 && (
        <ul aria-label="Chosen folders" className="flex flex-wrap gap-2">
          {ticked.map((path) => (
            <li
              key={path}
              className="flex max-w-full items-center gap-1 rounded-full bg-accent-50 py-1 pr-1 pl-3 text-sm text-accent-700 dark:bg-accent-500/15 dark:text-accent-400"
            >
              <span className="truncate" title={path}>
                {folderName(path)}
              </span>
              <button
                type="button"
                onClick={() => toggle(path)}
                aria-label={`Remove ${folderName(path)}`}
                className="grid size-6 place-items-center rounded-full hover:bg-accent-100 dark:hover:bg-accent-500/20"
              >
                <X className="size-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="rounded-xl border border-zinc-200 dark:border-zinc-800">
        <label className="flex items-center gap-2 border-b border-zinc-200 px-3 dark:border-zinc-800">
          <Search className="size-4 text-zinc-500" aria-hidden />
          <span className="sr-only">Filter folders</span>
          <input
            type="search"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            // Enter confirms the filter; inside the search form it would start the search.
            onKeyDown={(e) => {
              if (e.key === "Enter") e.preventDefault();
            }}
            placeholder={FOLDER_SEARCH.placeholder}
            className="h-10 w-full min-w-0 bg-transparent text-sm outline-none placeholder:text-zinc-500"
          />
        </label>
        {filter.trim() ? (
          <SearchResults
            query={query}
            roots={roots}
            ticked={ticked}
            onToggle={toggle}
            onChange={onChange}
          />
        ) : (
          <ul aria-label="Folders" className="max-h-80 overflow-y-auto p-1.5">
            {roots.map((root) => (
              <FolderNode
                key={root.path}
                name={root.name}
                path={root.path}
                hasChildren={root.mounted}
                disabled={!root.mounted}
                isRoot
                depth={0}
                ticked={ticked}
                onToggle={toggle}
              />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

type SearchProps = {
  query: string;
  roots: readonly RootFolder[];
  ticked: readonly string[];
  onToggle: (path: string) => void;
  onChange: (ticked: string[]) => void;
};

function SearchResults({ query, roots, ticked, onToggle, onChange }: SearchProps) {
  const matches = useQuery({
    queryKey: ["folder-search", query],
    queryFn: () => api.searchFolders(query),
    enabled: query !== "",
    retry: false,
  });
  const found = matches.data ?? [];
  const paths = found.map((m) => m.path);
  const everyTicked = allTicked(ticked, paths);
  // Where a match sits: its collection and the folders between, when there are any.
  const where = (path: string) => {
    const root = roots.find((r) => isInside(path, r.path));
    if (!root) return "";
    const between = path.slice(root.path.length).split("/").filter(Boolean).slice(0, -1);
    return [root.name, ...between].join(" › ");
  };

  if (query === "" || matches.isPending) {
    return (
      <p className="flex items-center gap-2 px-3 py-3 text-sm text-zinc-500">
        <Loader2 className="size-4 animate-spin" aria-hidden />
        {FOLDER_SEARCH.searching}
      </p>
    );
  }
  if (matches.isError) {
    return (
      <p role="alert" className="px-3 py-3 text-sm text-red-600">
        {matches.error.message}
      </p>
    );
  }
  if (found.length === 0) {
    return <p className="px-3 py-3 text-sm text-zinc-500">{FOLDER_SEARCH.none(query)}</p>;
  }
  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-200 px-3 py-2 dark:border-zinc-800">
        <p aria-live="polite" className="text-sm text-zinc-600 dark:text-zinc-400">
          {FOLDER_SEARCH.count(found.length, query)}
        </p>
        <button
          type="button"
          onClick={() => onChange(everyTicked ? untickAll(ticked, paths) : tickAll(ticked, paths))}
          className="rounded-lg px-3 py-1.5 text-sm font-medium text-accent-700 hover:bg-accent-50 dark:text-accent-400 dark:hover:bg-accent-500/15"
        >
          {everyTicked
            ? FOLDER_SEARCH.untickAll(found.length)
            : FOLDER_SEARCH.selectAll(found.length)}
        </button>
      </div>
      <ul aria-label="Matching folders" className="max-h-80 overflow-y-auto p-1.5">
        {found.map((match) => {
          const lockedBy = coveredBy(match.path, ticked);
          return (
            <li key={match.path}>
              <label className="flex min-w-0 cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-sm hover:bg-zinc-50 dark:hover:bg-zinc-900">
                <input
                  type="checkbox"
                  checked={ticked.includes(match.path) || lockedBy !== undefined}
                  disabled={lockedBy !== undefined}
                  onChange={() => onToggle(match.path)}
                  className="size-4 shrink-0 accent-teal-600"
                />
                <Folder className="size-4 shrink-0 text-zinc-500" aria-hidden />
                <span className="min-w-0">
                  <span className="block truncate" title={match.path}>
                    {match.name}
                  </span>
                  <span className="block truncate text-xs text-zinc-500">{where(match.path)}</span>
                </span>
              </label>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

type NodeProps = {
  name: string;
  path: string;
  hasChildren: boolean;
  disabled?: boolean;
  isRoot?: boolean;
  depth: number;
  ticked: readonly string[];
  onToggle: (path: string) => void;
};

function FolderNode(props: NodeProps) {
  const { name, path, hasChildren, disabled, isRoot, depth, ticked, onToggle } = props;
  const [open, setOpen] = useState(false);
  const children = useQuery({
    queryKey: ["folders", path],
    queryFn: () => api.children(path),
    enabled: open,
    retry: false,
  });
  const lockedBy = coveredBy(path, ticked);
  const checked = ticked.includes(path) || lockedBy !== undefined;
  const visibleChildren = children.data ?? [];

  const Icon = isRoot ? HardDrive : Folder;
  return (
    <li>
      <div
        data-testid="folder-node"
        className="flex items-center gap-1 rounded-lg py-0.5 pr-2 hover:bg-zinc-50 dark:hover:bg-zinc-900"
        style={{ paddingLeft: `${depth * 1.25}rem` }}
      >
        {hasChildren ? (
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-label={`${open ? "Hide" : "Show"} subfolders of ${name}`}
            aria-expanded={open}
            className="grid size-8 shrink-0 place-items-center rounded-md text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-200"
          >
            {children.isFetching ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <ChevronRight className={`size-4 transition-transform ${open ? "rotate-90" : ""}`} />
            )}
          </button>
        ) : (
          <span className="size-8 shrink-0" />
        )}
        <label
          className={`flex min-w-0 flex-1 items-center gap-2 py-1.5 text-sm ${
            disabled ? "opacity-50" : "cursor-pointer"
          }`}
        >
          <input
            type="checkbox"
            checked={checked}
            disabled={disabled || lockedBy !== undefined}
            onChange={() => onToggle(path)}
            className="size-4 shrink-0 accent-teal-600"
          />
          <Icon className="size-4 shrink-0 text-zinc-500" aria-hidden />
          <span className="truncate" title={path}>
            {name}
          </span>
          {disabled && <span className="text-xs text-zinc-500">(not mounted)</span>}
        </label>
      </div>
      {open && children.isError && (
        <p
          role="alert"
          className="py-1 text-sm text-red-600"
          style={{ paddingLeft: `${depth * 1.25 + 2.5}rem` }}
        >
          {children.error.message}{" "}
          <button type="button" className="underline" onClick={() => void children.refetch()}>
            Retry
          </button>
        </p>
      )}
      {open && visibleChildren.length > 0 && (
        <ul>
          {visibleChildren.map((child) => (
            <FolderNode
              key={child.path}
              name={child.name}
              path={child.path}
              hasChildren={child.has_children}
              depth={depth + 1}
              ticked={ticked}
              onToggle={onToggle}
            />
          ))}
        </ul>
      )}
    </li>
  );
}
