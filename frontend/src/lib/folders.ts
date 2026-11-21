/** Folder selection: ticking a parent covers everything below it. */

export function isInside(path: string, ancestor: string): boolean {
  return path !== ancestor && path.startsWith(ancestor.endsWith("/") ? ancestor : `${ancestor}/`);
}

/** The ticked folder (other than `path` itself) that covers `path`, if any. */
export function coveredBy(path: string, ticked: readonly string[]): string | undefined {
  return ticked.find((t) => isInside(path, t));
}

/** Tick or untick `path`. Covered folders are locked, so toggling them does nothing. */
export function toggleFolder(ticked: readonly string[], path: string): string[] {
  if (ticked.includes(path)) return ticked.filter((t) => t !== path);
  if (coveredBy(path, ticked)) return [...ticked];
  return [...ticked.filter((t) => !isInside(t, path)), path];
}

/** Only the top-most ticked folders: what the server needs. */
export function topMost(ticked: readonly string[]): string[] {
  return [...new Set(ticked)].filter((t) => !coveredBy(t, ticked));
}

export function folderName(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts[parts.length - 1] ?? path;
}

/** True when every path is ticked or covered by a ticked folder. */
export function allTicked(ticked: readonly string[], paths: readonly string[]): boolean {
  return paths.length > 0 && paths.every((p) => ticked.includes(p) || coveredBy(p, ticked));
}

/** Tick every path at once (a search's matches), as ticking each in turn would. */
export function tickAll(ticked: readonly string[], paths: readonly string[]): string[] {
  return paths.reduce<string[]>(
    (acc, path) => (acc.includes(path) ? acc : toggleFolder(acc, path)),
    [...ticked],
  );
}

/** Untick exactly these paths; folders that only cover them stay ticked. */
export function untickAll(ticked: readonly string[], paths: readonly string[]): string[] {
  return ticked.filter((t) => !paths.includes(t));
}
