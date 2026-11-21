import { Component, type ReactNode } from "react";
import { APP_ERROR } from "../lib/format";

type State = { error: Error | null };

/**
 * A page that fails while drawing (a reply it did not expect) shows what went wrong and a
 * way back, instead of unmounting the whole app to a blank page. Keyed by route in the app,
 * so going to another page starts afresh.
 */
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <main className="mx-auto max-w-xl px-4 py-24">
        <div role="alert" className="space-y-3">
          <h1 className="text-xl font-semibold">{APP_ERROR.title}</h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-300">{error.message}</p>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="rounded-lg bg-accent-600 px-3 py-1.5 text-sm font-medium text-white"
          >
            {APP_ERROR.reload}
          </button>
        </div>
      </main>
    );
  }
}
