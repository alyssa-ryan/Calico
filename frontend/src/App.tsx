import { Link, Route, Routes, useLocation } from "react-router-dom";
import { Footer, Masthead } from "./components/Banners";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Header } from "./components/Header";
import { RouteFocus } from "./components/RouteFocus";
import { SKIP_TO_CONTENT } from "./lib/format";
import { Compose } from "./pages/Compose";
import { LabelSet } from "./pages/LabelSet";
import { Labels } from "./pages/Labels";
import { Run } from "./pages/Run";
import { Runs } from "./pages/Runs";

export function App() {
  const { pathname } = useLocation();
  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#content"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded-lg focus:bg-white focus:px-3 focus:py-2 focus:text-sm focus:shadow dark:focus:bg-zinc-900"
      >
        {SKIP_TO_CONTENT}
      </a>
      <RouteFocus />
      <Masthead />
      <Header />
      <div id="content" tabIndex={-1} className="flex-1 focus:outline-none">
        <ErrorBoundary key={pathname}>
          <Routes>
            <Route path="/" element={<Compose />} />
            <Route path="/runs" element={<Runs />} />
            <Route path="/runs/:runId" element={<Run />} />
            <Route path="/label" element={<Labels />} />
            <Route path="/label/:setId" element={<LabelSet />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </ErrorBoundary>
      </div>
      <Footer />
    </div>
  );
}

function NotFound() {
  return (
    <main className="mx-auto max-w-xl px-4 py-24 text-center">
      <h1 className="text-2xl font-semibold">Nothing here</h1>
      <p className="mt-2 text-zinc-500">
        <Link to="/" className="text-accent-600 underline-offset-4 hover:underline">
          Start a new selection
        </Link>
      </p>
    </main>
  );
}
