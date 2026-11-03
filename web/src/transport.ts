/** How `api.ts` reaches the back end: `minime serve` over HTTP. The browser build swaps this
 * module for `browser/transport.ts` (see `vite.config.ts`). */
export type EventHandlers = Record<string, (data: unknown) => void>;

export const transportFetch = (input: string, init?: RequestInit): Promise<Response> => fetch(input, init);

/** Server-sent events of `url`, each named event's JSON data to its handler; on an error the
 * stream is closed first, then `onError` runs. Returns close. */
export function openEvents(url: string, on: EventHandlers, onError: () => void): () => void {
  const es = new EventSource(url);
  for (const [name, fn] of Object.entries(on))
    es.addEventListener(name, (e) => fn(JSON.parse((e as MessageEvent).data)));
  es.onerror = () => {
    es.close();
    onError();
  };
  return () => es.close();
}
