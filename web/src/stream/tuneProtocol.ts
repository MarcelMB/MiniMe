// Messages between the main thread's `TuneWorkers` and a tune worker (`tune.worker.ts`).

export type WindowLoad = { i: number; t0: number; t1: number; files: Map<string, Uint8Array> };

export type Summary = {
  good: number;
  flagged: number;
  lost: number;
  score: number;
  low_pnr: number;
  in_pieces: number;
  bad_size: number;
  duplicate: number;
};

export type TrialOut = {
  counts: Record<string, number>;
  score: Summary | null;
  error: string | null;
  size10: number[];
  seconds: [number, number, number];
};

export type ToTuneWorker =
  /** Replaces what the worker holds. */
  | { type: "load"; runSession: Uint8Array; windows: WindowLoad[] }
  /** `caps` is "" or a JSON object `{ "<window i>": { seeds_pnr, seeds_merged } }`; the worker picks its window's entry. */
  | {
      type: "trial";
      id: number;
      session: Uint8Array;
      checks: string;
      reference: boolean;
      caps: string;
    }
  | { type: "units"; id: number }
  | {
      type: "hist";
      id: number;
      seeds_wnd: number;
      seeds_stp: number;
      seeds_max_wnd: number;
      n_windows: number;
    };

export type FromTuneWorker =
  | { type: "ack"; heapMb: number; bytes: number }
  | { type: "trial"; id: number; results: { i: number; out: TrialOut }[]; heapMb: number }
  | {
      type: "units";
      id: number;
      windows: { i: number; t0: number; t1: number; bytes: Uint8Array }[];
    }
  | { type: "hist"; id: number; hist: number[] }
  | { type: "error"; id: number; message: string };
