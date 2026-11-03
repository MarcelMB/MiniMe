import type { ProgressEvent } from "../launch/params";

/** Messages between the stream page and its worker. */
export type ToWorker = { avis: File[]; inputs: File[] };

export type Stage = "stream" | "motion" | "pass1" | "pass2" | "seeds" | "init" | "cnmf" | "stitch";

export type FromWorker =
  | {
      type: "start";
      stage: Stage;
      /** Where the stage sits in the run ("window 2/3"), empty for a lone stage. */
      label: string;
      frames: number;
      units: number;
      height: number;
      width: number;
    }
  /** A stage began or ended (`progress.jsonl`'s events, as `minime serve` shows them); a
   * stage that runs once per window ends with the seconds of all its windows so far. */
  | ({ type: "stage"; note?: string } & ProgressEvent)
  | {
      type: "progress";
      stage: Stage;
      done: number;
      frames: number;
      seconds: number;
      stateMb: number;
      heapMb: number;
    }
  // large outputs (the traces) come as an OPFS file instead of bytes
  | { type: "done"; files: Map<string, Uint8Array | Blob>; seconds: number }
  | { type: "log"; message: string }
  | { type: "error"; message: string };
