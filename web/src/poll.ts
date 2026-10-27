import { useEffect, useState } from "react";
import { Computing } from "./api";

export type Resource<T> = { data?: T; error?: string; computing: boolean; elapsed: number };

/** Call ``load`` until it stops throwing ``Computing``; report every state. Returns cancel. */
export function pollResource<T>(
  load: () => Promise<T>,
  onState: (s: Resource<T>) => void,
  delayMs = 1500,
): () => void {
  let alive = true;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const attempt = () => {
    load().then(
      (data) => alive && onState({ data, computing: false, elapsed: 0 }),
      (e: unknown) => {
        if (!alive) return;
        if (e instanceof Computing) {
          onState({ computing: true, elapsed: e.elapsed });
          timer = setTimeout(attempt, delayMs);
        } else {
          onState({ error: e instanceof Error ? e.message : String(e), computing: false, elapsed: 0 });
        }
      },
    );
  };
  attempt();
  return () => {
    alive = false;
    if (timer !== undefined) clearTimeout(timer);
  };
}

/** ``load`` once per ``key`` (null = nothing to load), retrying while the back end computes. */
export function useResource<T>(key: string | null, load: () => Promise<T>): Resource<T> {
  const [state, setState] = useState<Resource<T>>({ computing: false, elapsed: 0 });
  useEffect(() => {
    if (key === null) return;
    setState({ computing: false, elapsed: 0 });
    return pollResource(load, setState);
    // ``key`` identifies the request; ``load`` is recreated each render
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}
