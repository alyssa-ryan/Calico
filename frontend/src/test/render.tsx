import { type QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import type { Pick, RunDetail, TaskInfo } from "../api/types";
import { makeQueryClient } from "../lib/queryClient";

/** A query client like the app's (its caching), but one that never retries. */
export const testClient = () => makeQueryClient({ retry: false });

/**
 * Render a page inside the same providers the app uses, at `path` (matched by `route`).
 * Pass one `client` to two renders to come back to a page, as navigating away and back
 * does: the app's cache keeps what the first visit fetched.
 */
export function renderAt(
  ui: ReactElement,
  {
    path = "/",
    route = "/",
    state,
    client = testClient(),
  }: { path?: string; route?: string; state?: unknown; client?: QueryClient } = {},
) {
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[{ pathname: path, state }]}>
        <Routes>
          <Route path={route} element={ui} />
          <Route path="*" element={<p>navigated away</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const schema = (properties: Record<string, unknown>) => ({ type: "object", properties });

export const TASKS: TaskInfo[] = [
  {
    name: "best",
    title: "Strongest photographs",
    description: "The strongest photographs.",
    reference_use: "unused",
    reference_label: "",
    reference_help: "",
    default_limit: 24,
    default_describe_weight: 0.5,
    uses_taste: true,
    options_schema: schema({
      diversity: { type: "number", minimum: 0, maximum: 1, default: 0.3, title: "Diversity" },
    }),
  },
  {
    name: "same_view",
    title: "Same view over time",
    description: "Every photo of the same view.",
    reference_use: "required",
    reference_label: "Reference photo",
    reference_help: "The view you want to see again.",
    default_limit: 200,
    default_describe_weight: 0.3,
    uses_taste: false,
    options_schema: schema({}),
  },
  {
    name: "triptych",
    title: "Three-photo stories",
    description: "Sets of three.",
    reference_use: "optional",
    reference_label: "Anchor photo",
    reference_help: "Your photo goes in the middle.",
    default_limit: 6,
    default_describe_weight: 1 / 3,
    uses_taste: true,
    options_schema: schema({}),
  },
];

export const pick = (name: string, over: Partial<Pick> = {}): Pick => ({
  path: `/archive/Old Mill/${name}`,
  content_key: name,
  location: "Old Mill",
  captured_at: "2021-03-08T09:00:00",
  score: 0.8,
  evidence: [{ signal: "quality", value: 0.8, weight: 1, detail: "sharp" }],
  burst_size: 1,
  alternates: [],
  ...over,
});

export function runDetail(
  task: string,
  groups: RunDetail["result"]["groups"],
  over: Partial<RunDetail> = {},
): RunDetail {
  return {
    task_title: TASKS.find((t) => t.name === task)?.title ?? task,
    reference_label: "",
    feedback: {},
    ...over,
    result: {
      run_id: "r1",
      created_at: "2026-10-13T10:00:00Z",
      request: {
        task,
        folders: ["/archive"],
        criteria: "",
        only_clear_matches: false,
        describe_weight: null,
        template: null,
        examples: [],
        references: [],
        limit: 6,
        options: {},
      },
      groups,
      example_keys: [],
      notes: [],
      stages: [{ name: "index", count_in: 3, count_out: 3, seconds: 0.1 }],
      versions: { calico: "0.2.0" },
    },
  };
}
