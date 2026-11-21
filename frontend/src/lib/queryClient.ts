/** The query client the app runs with, so tests run with the same caching. */
import { type DefaultOptions, QueryClient } from "@tanstack/react-query";

export const QUERY_DEFAULTS: DefaultOptions = {
  queries: { staleTime: 30_000, refetchOnWindowFocus: false },
};

/** The app's query client; ``queries`` overrides only what it names (tests: no retries). */
export function makeQueryClient(queries: DefaultOptions["queries"] = {}): QueryClient {
  return new QueryClient({
    defaultOptions: { queries: { ...QUERY_DEFAULTS.queries, ...queries } },
  });
}
