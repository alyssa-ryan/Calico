/**
 * Starting a new request from a result ("Build a story around this", "Find this
 * view over time", "More like this"): the composer is opened pre-filled via router state.
 */
import type { Pick as PickT, SelectionRequest } from "../api/types";

type Search = Readonly<
  Pick<
    SelectionRequest,
    | "task"
    | "folders"
    | "criteria"
    | "examples"
    | "only_clear_matches"
    | "describe_weight"
    | "template"
    | "limit"
    | "options"
    | "references"
  >
>;

/** A photo already in the archive, used as a reference/anchor by path. */
export type ArchiveRef = { path: string; contentKey: string };

export type Prefill = {
  task: string;
  folders: string[];
  references: ArchiveRef[];
  criteria: string;
  /** Photos that are part of the description. */
  examples: ArchiveRef[];
  onlyClearMatches: boolean;
  /** null: the task's default. */
  describeWeight: number | null;
  template: "general" | "yours" | null;
  limit: number | null;
  options: Record<string, unknown>;
  /** What the search started from could not bring along ("references"), to say so. */
  dropped: string[];
};

const isString = (v: unknown): v is string => typeof v === "string";

/** Router state is untyped (and survives reloads), so validate it before use. */
export function readPrefill(state: unknown): Prefill | null {
  if (typeof state !== "object" || state === null) return null;
  const s = state as Record<string, unknown>;
  if (!isString(s.task) || !Array.isArray(s.folders) || !s.folders.every(isString)) return null;
  const number = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  return {
    task: s.task,
    folders: s.folders,
    references: archiveRefs(s.references),
    criteria: isString(s.criteria) ? s.criteria : "",
    examples: archiveRefs(s.examples),
    onlyClearMatches: s.onlyClearMatches === true,
    describeWeight: number(s.describeWeight),
    template: s.template === "general" || s.template === "yours" ? s.template : null,
    limit: number(s.limit),
    options:
      typeof s.options === "object" && s.options !== null && !Array.isArray(s.options)
        ? (s.options as Record<string, unknown>)
        : {},
    dropped: Array.isArray(s.dropped) ? s.dropped.filter(isString) : [],
  };
}

function archiveRefs(value: unknown): ArchiveRef[] {
  return Array.isArray(value)
    ? value.filter(
        (r): r is ArchiveRef =>
          typeof r === "object" && r !== null && isString(r.path) && isString(r.contentKey),
      )
    : [];
}

/** The same search again, every setting kept, with `pick` added to what it was described
 * by. Its reference photos stay behind (an upload is not in the archive, and Compose has
 * no preview of it), and ``dropped`` says so. */
export function moreLikeThis(
  request: Search,
  exampleKeys: readonly string[],
  pick: PickT,
): Prefill {
  const examples = request.examples.map((path, i) => ({ path, contentKey: exampleKeys[i] ?? "" }));
  if (!examples.some((e) => e.contentKey === pick.content_key)) {
    examples.push({ path: pick.path, contentKey: pick.content_key });
  }
  return {
    task: request.task,
    folders: request.folders,
    criteria: request.criteria,
    references: [],
    examples,
    onlyClearMatches: request.only_clear_matches,
    describeWeight: request.describe_weight ?? null,
    template: request.template ?? null,
    limit: request.limit ?? null,
    options: { ...request.options },
    dropped: request.references.length ? ["references"] : [],
  };
}
