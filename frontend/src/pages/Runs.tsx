import { useQuery } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { Link } from "react-router-dom";
import { api, thumbUrl } from "../api/client";

export function Runs() {
  const runs = useQuery({ queryKey: ["runs"], queryFn: () => api.runs(), staleTime: 0 });
  return (
    <main className="mx-auto max-w-7xl space-y-6 px-4 py-8">
      <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Past runs</h1>
      {runs.isError && (
        <p role="alert" className="text-red-600">
          {runs.error.message}
        </p>
      )}
      {!runs.data && !runs.isError && <Loader2 className="size-5 animate-spin text-zinc-500" />}
      {runs.data?.length === 0 && (
        <p className="rounded-xl border border-dashed border-zinc-300 p-10 text-center text-zinc-500 dark:border-zinc-700">
          Nothing yet.{" "}
          <Link to="/" className="underline">
            Make your first selection.
          </Link>
        </p>
      )}
      <ul className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {runs.data?.map((run) => (
          <li key={run.run_id}>
            <Link
              to={`/runs/${run.run_id}`}
              data-testid="run-card"
              className="group block overflow-hidden rounded-xl border border-zinc-200 bg-white transition hover:shadow-md focus-visible:outline-2 focus-visible:outline-accent-500 dark:border-zinc-800 dark:bg-zinc-900"
            >
              <div className="grid aspect-[4/3] grid-cols-2 grid-rows-2 gap-0.5 bg-zinc-100 dark:bg-zinc-800">
                {run.cover_keys.slice(0, 4).map((key) => (
                  <img
                    key={key}
                    src={thumbUrl(key)}
                    alt=""
                    loading="lazy"
                    className="size-full object-cover"
                  />
                ))}
              </div>
              <div className="space-y-1 p-3">
                <p className="font-medium">{run.task_title}</p>
                {run.criteria && (
                  <p className="truncate text-sm text-zinc-600 italic dark:text-zinc-300">
                    “{run.criteria}”
                  </p>
                )}
                <p className="text-xs text-zinc-500">
                  {new Date(run.created_at).toLocaleString()} · {run.picks} photo
                  {run.picks === 1 ? "" : "s"}
                </p>
              </div>
            </Link>
          </li>
        ))}
      </ul>
    </main>
  );
}
