/** Which key presses a page's shortcuts answer. */

/**
 * Whether a keyboard shortcut should leave ``event`` alone: a key held down (it repeats,
 * and would label every photo in turn), a key typed into a field, or one with a modifier
 * (the browser's own shortcuts).
 */
export function ignoresShortcut(event: KeyboardEvent): boolean {
  if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return true;
  const target = event.target;
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.isContentEditable ||
    target.contentEditable === "true" ||
    ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)
  );
}
