import { useQuery } from "@tanstack/react-query";
import { Aperture, Loader2, Moon, RefreshCw, Sun } from "lucide-react";
import { NavLink } from "react-router-dom";
import { api } from "../api/client";
import { API_FINGERPRINT } from "../api/fingerprint";
import { DISCONNECTED, STALE_SERVER } from "../lib/format";
import { healthInterval } from "../lib/health";
import { useTheme } from "../lib/theme";

const navClass = ({ isActive }: { isActive: boolean }) =>
  `rounded-lg px-3 py-1.5 text-sm transition-colors ${
    isActive
      ? "bg-zinc-100 font-medium text-zinc-900 dark:bg-zinc-800 dark:text-white"
      : "text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
  }`;

export function Header() {
  const [theme, toggleTheme] = useTheme();
  return (
    <header className="sticky top-0 z-30 border-b border-zinc-200/70 bg-white/80 backdrop-blur dark:border-zinc-800/70 dark:bg-zinc-950/80">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-2 px-4 sm:gap-4">
        <NavLink
          to="/"
          aria-label="Calico"
          className="flex items-center gap-2 font-semibold tracking-tight"
        >
          <Aperture className="size-5 text-accent-600" aria-hidden />
          <span className="hidden sm:inline">Calico</span>
        </NavLink>
        <nav aria-label="Main" className="flex items-center gap-1">
          <NavLink to="/" end className={navClass}>
            New selection
          </NavLink>
          <NavLink to="/runs" className={navClass}>
            Past runs
          </NavLink>
          <NavLink to="/label" className={navClass}>
            Label
          </NavLink>
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <HealthPill />
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            className="grid size-9 place-items-center rounded-lg text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900 focus-visible:outline-2 focus-visible:outline-accent-500 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-white"
          >
            {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </button>
        </div>
      </div>
      <StaleServer />
    </header>
  );
}

const useHealth = () =>
  useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: (q) => healthInterval(q.state.data?.engine, q.state.status === "error"),
  });

/** The server runs another version of Calico than this page was built for: say how to fix it,
 * before a request fails with a validation error that names a field. */
function StaleServer() {
  const { data } = useHealth();
  if (!data || data.api === API_FINGERPRINT) return null;
  return (
    <div
      role="alert"
      className="border-amber-200 border-t bg-amber-50 text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/60 dark:text-amber-100"
    >
      <p className="mx-auto flex max-w-7xl items-start gap-2 px-4 py-2 text-sm">
        <RefreshCw className="mt-0.5 size-4 shrink-0" aria-hidden />
        {STALE_SERVER}
      </p>
    </div>
  );
}

function HealthPill() {
  const { data, isError } = useHealth();
  // A failed check keeps the last answer, so "Ready" would stay up after the server died.
  if (isError) {
    return (
      <span
        role="status"
        title={DISCONNECTED.help}
        className="rounded-full bg-red-50 px-2.5 py-1 text-xs text-red-700 dark:bg-red-950/60 dark:text-red-300"
      >
        {DISCONNECTED.label}
      </span>
    );
  }
  if (!data) return null;
  if (data.engine === "starting") {
    return (
      <span className="hidden items-center gap-1.5 rounded-full bg-amber-50 px-2.5 py-1 text-xs text-amber-800 sm:flex dark:bg-amber-950/60 dark:text-amber-200">
        <Loader2 className="size-3 animate-spin" aria-hidden /> Loading models…
      </span>
    );
  }
  if (data.engine === "error") {
    return (
      <span
        title={data.error ?? undefined}
        className="rounded-full bg-red-50 px-2.5 py-1 text-xs text-red-700 dark:bg-red-950/60 dark:text-red-300"
      >
        Engine error
      </span>
    );
  }
  return (
    <span className="hidden items-center gap-1.5 rounded-full px-2.5 py-1 text-xs text-zinc-500 sm:flex dark:text-zinc-400">
      <span className="size-1.5 rounded-full bg-emerald-500" aria-hidden /> Ready
    </span>
  );
}
