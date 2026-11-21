/** How often the header asks the server whether it is up. */
import type { Health } from "../api/types";

/**
 * Milliseconds until the next health check: often while the models load, rarely once they
 * are ready or have failed (a failed engine stays failed until a restart), and every few
 * seconds while the server does not answer, so the page sees it come back.
 */
export function healthInterval(engine: Health["engine"] | undefined, failed: boolean): number {
  if (failed) return 5_000;
  return engine === "ready" || engine === "error" ? 30_000 : 1_500;
}
