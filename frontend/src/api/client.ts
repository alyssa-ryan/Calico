import type {
  ChildFolder,
  FolderMatch,
  Health,
  JobStatus,
  Label,
  LabelSetDetail,
  LabelSetSummary,
  Rating,
  ReadingResponse,
  RootFolder,
  RunDetail,
  RunListItem,
  TaskInfo,
  TasteInfo,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function fail(response: Response): Promise<never> {
  let message = `${response.status} ${response.statusText}`.trim();
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") message = body.detail;
    else if (Array.isArray(body.detail)) {
      message = body.detail
        .map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : String(d)))
        .join("; ");
    }
  } catch {
    /* keep the status line */
  }
  throw new ApiError(message, response.status);
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) return fail(response);
  return (await response.json()) as T;
}

export const api = {
  health: () => getJson<Health>("/api/health"),
  tasks: () => getJson<TaskInfo[]>("/api/tasks"),
  roots: () => getJson<RootFolder[]>("/api/folders"),
  children: (path: string) =>
    getJson<ChildFolder[]>(`/api/folders?path=${encodeURIComponent(path)}`),
  searchFolders: (query: string) =>
    getJson<FolderMatch[]>(`/api/folders/search?q=${encodeURIComponent(query)}`),
  job: (id: string) => getJson<JobStatus>(`/api/jobs/${encodeURIComponent(id)}`),
  runs: (limit = 50) => getJson<RunListItem[]>(`/api/runs?limit=${limit}`),
  run: (id: string) => getJson<RunDetail>(`/api/runs/${encodeURIComponent(id)}`),

  async select(form: FormData): Promise<{ job_id: string }> {
    const response = await fetch("/api/select", { method: "POST", body: form });
    if (!response.ok) return fail(response);
    return (await response.json()) as { job_id: string };
  },

  reading: (text: string, folders: readonly string[]) => {
    const query = new URLSearchParams({ text });
    for (const folder of folders) query.append("folders", folder);
    return getJson<ReadingResponse>(`/api/reading?${query}`);
  },

  labelSets: () => getJson<LabelSetSummary[]>("/api/label-sets"),
  labelSet: (id: string) => getJson<LabelSetDetail>(`/api/label-sets/${encodeURIComponent(id)}`),

  async newLabelSet(
    text: string,
    folders: string[],
    reference?: string,
  ): Promise<{ job_id: string }> {
    const response = await fetch("/api/label-sets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reference ? { text, folders, reference } : { text, folders }),
    });
    if (!response.ok) return fail(response);
    return (await response.json()) as { job_id: string };
  },

  async taste(): Promise<TasteInfo> {
    const response = await fetch("/api/taste");
    if (!response.ok) return fail(response);
    return (await response.json()) as TasteInfo;
  },

  async newTasteSet(folders: string[]): Promise<{ job_id: string }> {
    const response = await fetch("/api/label-sets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folders, taste: true }),
    });
    if (!response.ok) return fail(response);
    return (await response.json()) as { job_id: string };
  },

  async newStorySet(folders: string[]): Promise<{ job_id: string }> {
    const response = await fetch("/api/label-sets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folders, stories: true }),
    });
    if (!response.ok) return fail(response);
    return (await response.json()) as { job_id: string };
  },

  async newAnchorSet(folders: string[]): Promise<{ job_id: string }> {
    const response = await fetch("/api/label-sets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folders, anchor: true }),
    });
    if (!response.ok) return fail(response);
    return (await response.json()) as { job_id: string };
  },

  async newCompareSet(folders: string[]): Promise<{ job_id: string }> {
    const response = await fetch("/api/label-sets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folders, compare: true }),
    });
    if (!response.ok) return fail(response);
    return (await response.json()) as { job_id: string };
  },

  async setStory(
    setId: string,
    index: number,
    label: Label | null,
    doesntFit: string[],
  ): Promise<void> {
    const url = `/api/label-sets/${encodeURIComponent(setId)}/stories/${index}`;
    const response = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, doesnt_fit: doesntFit }),
    });
    if (!response.ok) return fail(response);
  },

  async setLabel(setId: string, contentKey: string, label: Label | null): Promise<void> {
    const url = `/api/label-sets/${encodeURIComponent(setId)}/labels/${encodeURIComponent(contentKey)}`;
    const response = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label }),
    });
    if (!response.ok) return fail(response);
  },

  async deleteLabelSet(setId: string): Promise<void> {
    const response = await fetch(`/api/label-sets/${encodeURIComponent(setId)}`, {
      method: "DELETE",
    });
    if (!response.ok) return fail(response);
  },

  async feedback(runId: string, contentKey: string, rating: Rating): Promise<void> {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runId, content_key: contentKey, rating }),
    });
    if (!response.ok) return fail(response);
  },
};

export const thumbUrl = (contentKey: string) => `/thumbs/${contentKey}`;
export const imageUrl = (runId: string, contentKey: string) => `/runs/${runId}/image/${contentKey}`;
export const labelImageUrl = (setId: string, contentKey: string) =>
  `/api/label-sets/${setId}/image/${contentKey}`;
export const referenceUrl = (runId: string, index: number) => `/runs/${runId}/reference/${index}`;
export const labelReferenceUrl = (setId: string) => `/api/label-sets/${setId}/reference`;
