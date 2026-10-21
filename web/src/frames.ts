import type { NDArray } from "./api";

/** Most prefetch requests allowed in flight at once; ``get()`` is never capped. */
export const MAX_PREFETCH = 4;

export class FrameCache {
  private lru = new Map<string, Promise<NDArray>>();
  private pending = new Set<string>();
  private prefetching = 0;

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
    this.pending.add(k);
    p.then(
      () => this.pending.delete(k),
      () => {
        this.pending.delete(k);
        if (this.lru.get(k) === p) this.lru.delete(k);
      },
    );
    this.lru.set(k, p);
    this.evict();
    return p;
  }

  /** Drop the oldest settled entries; pending ones stay, so the size may exceed the capacity. */
  private evict(): void {
    if (this.lru.size <= this.capacity) return;
    for (const k of this.lru.keys()) {
      if (this.lru.size <= this.capacity) break;
      if (!this.pending.has(k)) this.lru.delete(k);
    }
  }

  /** Start loading the frames around ``t`` that are not cached yet (at most 4 in flight). */
  prefetch(source: string, t: number, T: number, ahead = 12, behind = 3): void {
    for (let d = -behind; d <= ahead; d++) {
      if (this.prefetching >= MAX_PREFETCH) return;
      const u = t + d;
      if (d !== 0 && u >= 0 && u < T && !this.has(source, u)) {
        this.prefetching++;
        const release = () => void this.prefetching--;
        this.get(source, u).then(release, release);
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
