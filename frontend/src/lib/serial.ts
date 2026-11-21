/** Saves that reach the server in the order they were made. */

/**
 * A queue per key: a task starts only after the previous task for the same key has
 * finished (resolved or not); tasks for different keys run at once. Two quick changes to
 * one photo then reach the server in order, so the last one shown is the one stored.
 */
export function serialByKey() {
  const tails = new Map<string, Promise<unknown>>();
  return <T>(key: string, task: () => Promise<T>): Promise<T> => {
    const run = (tails.get(key) ?? Promise.resolve()).then(task, task);
    tails.set(
      key,
      run.catch(() => undefined),
    );
    return run;
  };
}
