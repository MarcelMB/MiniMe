/** Messages between the stream page and its worker. */
export type ToWorker = { avis: File[]; inputs: File[] };

export type FromWorker =
  | {
      type: "start";
      stage: "stream" | "motion" | "pass1" | "pass2" | "seeds" | "init" | "cnmf" | "stitch";
      frames: number;
      units: number;
      height: number;
      width: number;
    }
  | { type: "progress"; done: number; frames: number; seconds: number; stateMb: number }
  // large outputs (the traces) come as an OPFS file instead of bytes
  | { type: "done"; files: Map<string, Uint8Array | Blob>; seconds: number }
  | { type: "log"; message: string }
  | { type: "error"; message: string };
