import type { SelectBody, Template } from "../api/types";
import { topMost } from "./folders";

export type Draft = {
  task: string;
  folders: readonly string[];
  criteria: string;
  onlyClearMatches: boolean;
  /** The description's share of each photo's score; null = the task's default. */
  describeWeight: number | null;
  /** Whose taste ranks photos; null = the template saved in calico.toml. */
  template: Template | null;
  /** Archive photos that are part of the description. */
  examples: readonly string[];
  limit: number | null;
  options: Record<string, unknown>;
  archiveReferences: readonly string[];
};

export function buildRequest(draft: Draft): SelectBody {
  const described = draft.criteria.trim() !== "" || draft.examples.length > 0;
  return {
    task: draft.task,
    folders: topMost(draft.folders),
    criteria: draft.criteria.trim(),
    only_clear_matches: draft.onlyClearMatches && draft.criteria.trim() !== "",
    describe_weight: described ? draft.describeWeight : null,
    template: draft.template,
    examples: [...draft.examples],
    limit: draft.limit,
    options: draft.options,
    references: [...draft.archiveReferences],
  };
}

export function buildFormData(request: SelectBody, uploads: readonly File[]): FormData {
  const form = new FormData();
  form.append("request", JSON.stringify(request));
  for (const file of uploads) form.append("references", file, file.name);
  return form;
}

export type Blocker = string | null;

/** Why the form cannot be submitted yet, in words for the user. */
export function submitBlocker(input: {
  folders: readonly string[];
  referenceUse: "required" | "optional" | "unused";
  referenceLabel: string;
  referenceCount: number;
  optionErrors: Record<string, string>;
  limitValid: boolean;
}): Blocker {
  if (input.folders.length === 0) return "Choose at least one folder.";
  if (input.referenceUse === "required" && input.referenceCount === 0) {
    const noun = input.referenceLabel.toLowerCase() || "reference photo";
    return `Add ${/^[aeiou]/.test(noun) ? "an" : "a"} ${noun}.`;
  }
  if (!input.limitValid) return "Enter how many results you want.";
  const first = Object.values(input.optionErrors)[0];
  if (first) return `Fine-tune: ${first}`;
  return null;
}
