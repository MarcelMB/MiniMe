import type { FromTuneWorker, ToTuneWorker, TrialOut } from "../../stream/tuneProtocol";

/** A stand-in for a tune worker: each loaded window answers with `score(i, session text)`. */
export class FakeTuneWorker {
  onmessage: ((e: MessageEvent<FromTuneWorker>) => void) | null = null;
  onerror: ((e: ErrorEvent) => void) | null = null;
  held: number[] = [];
  terminated = false;
  constructor(private score: (i: number, session: string) => TrialOut) {}
  postMessage(m: ToTuneWorker) {
    const reply = (r: FromTuneWorker) =>
      setTimeout(() => this.onmessage?.({ data: r } as MessageEvent<FromTuneWorker>), 0);
    if (m.type === "load") {
      this.held = m.windows.map((w) => w.i);
      reply({ type: "ack", heapMb: 10 * this.held.length, bytes: 0 });
    } else if (m.type === "trial") {
      const s = new TextDecoder().decode(m.session);
      reply({
        type: "trial",
        id: m.id,
        results: this.held.map((i) => ({ i, out: this.score(i, s) })),
        heapMb: 20,
      });
    } else if (m.type === "units") {
      reply({
        type: "units",
        id: m.id,
        windows: this.held.map((i) => ({
          i,
          t0: i * 10,
          t1: i * 10 + 10,
          bytes: new Uint8Array([i]),
        })),
      });
    } else if (m.type === "hist") reply({ type: "hist", id: m.id, hist: [this.held.length] });
  }
  terminate() {
    this.terminated = true;
  }
}

export const out = (good: number, flagged: number, lost = 0, error: string | null = null): TrialOut => ({
  counts: { seeds_pnr: 10, seeds_merged: 5, units_final: good + flagged },
  score: error
    ? null
    : {
        good,
        flagged,
        lost,
        score: good - flagged - lost,
        low_pnr: flagged,
        in_pieces: 0,
        bad_size: 0,
        duplicate: 0,
      },
  error,
  size10: error ? [] : [20, 30],
  seconds: [0.1, 0.2, 0.3],
});
