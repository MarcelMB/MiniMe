/** Messages between the stream page and its worker. */
export type ToWorker = { avis: File[]; inputs: File[] };

export type FromWorker =
  | {
      type: "start";
      stage: "stream" | "motion" | "pass1" | "pass2";
      frames: number;
      units: number;
      height: number;
      width: number;
    }
  | { type: "progress"; done: number; frames: number; seconds: number; stateMb: number }
  | { type: "done"; files: Map<string, Uint8Array>; seconds: number }
  | { type: "log"; message: string }
  | { type: "error"; message: string };
