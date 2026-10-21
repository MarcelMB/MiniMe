import type { NDArray } from "./api";

export class FrameCache {
  private lru = new Map<string, Promise<NDArray>>();

  constructor(
    private fetcher: (source: string, t: number) => Promise<NDArray>,
    private capacity = 160,
  ) {}

  private key(source: string, t: number) {
    return `${source}:${t}`;
  }

  has(source: string, t: number): boolean {
    return this.lru.has(this.key(source, t));
  }

  get(source: string, t: number): Promise<NDArray> {
    const k = this.key(source, t);
    const hit = this.lru.get(k);
    if (hit) {
      this.lru.delete(k);
      this.lru.set(k, hit);
      return hit;
    }
    const p = this.fetcher(source, t);
    p.catch(() => {
      if (this.lru.get(k) === p) this.lru.delete(k);
    });
    this.lru.set(k, p);
    while (this.lru.size > this.capacity) {
      this.lru.delete(this.lru.keys().next().value as string);
    }
    return p;
  }

  /** Start loading the frames around ``t`` that are not cached yet. */
  prefetch(source: string, t: number, T: number, ahead = 12, behind = 3): void {
    for (let d = -behind; d <= ahead; d++) {
      const u = t + d;
      if (d !== 0 && u >= 0 && u < T && !this.has(source, u)) {
        this.get(source, u).catch(() => {});
      }
    }
  }
}

export const STALE = Symbol("stale");

/** Wrap promises so that only the most recently wrapped one resolves to its value. */
export function latest(): <T>(p: Promise<T>) => Promise<T | typeof STALE> {
  let n = 0;
  return <T>(p: Promise<T>) => {
    const mine = ++n;
    return p.then(
      (v) => (mine === n ? v : STALE),
      (e: unknown) => {
        if (mine === n) throw e;
        return STALE;
      },
    );
  };
}
