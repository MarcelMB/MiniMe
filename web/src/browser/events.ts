import type { EventHandlers } from "../transport";

/** Takes the event stream of `path` (e.g. `/jobs/<id>/events`) or returns null. */
export type EventProvider = (path: string, emit: (name: string, data: unknown) => void) => (() => void) | null;
const providers: EventProvider[] = [];

export function addEventProvider(p: EventProvider): void {
  providers.push(p);
}

export function openEvents(url: string, on: EventHandlers, onError: () => void): () => void {
  const path = new URL(url, "http://local").pathname.replace(/^\/api/, "");
  const emit = (name: string, data: unknown) => on[name]?.(data);
  for (const p of providers) {
    const close = p(path, emit);
    if (close) return close;
  }
  queueMicrotask(onError);
  return () => {};
}
