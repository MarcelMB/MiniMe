import { HttpError, type Reply } from "./router";

const tasks = new Map<string, { began: number; error?: string; done: boolean }>();

/** Main's "202 while computing": start `fn` once per key and answer the seconds so far. A
 * finished task is forgotten, so the next request reads what `fn` cached. */
export function computing(key: string, fn: () => Promise<void>): Reply {
  const t = tasks.get(key);
  if (t?.error !== undefined) {
    tasks.delete(key);
    throw new HttpError(500, t.error);
  }
  if (!t || t.done) {
    const task = { began: performance.now(), done: false } as { began: number; error?: string; done: boolean };
    tasks.set(key, task);
    fn().then(
      () => {
        task.done = true;
        tasks.delete(key);
      },
      (e: unknown) => {
        task.error = e instanceof Error ? e.message : String(e);
      },
    );
    return { computing: 0 };
  }
  return { computing: (performance.now() - t.began) / 1e3 };
}

export function resetComputing(): void {
  tasks.clear();
}
