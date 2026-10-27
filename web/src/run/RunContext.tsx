import { createContext, useContext } from "react";
import type { SegOption } from "../controls";
import type { Img } from "../draw";
import type { FrameCache } from "../frames";
import type { Available, Summary, Unit } from "../types";

export type Source = "max" | "raw" | "pre" | "recon";

export type RunState = {
  summary: Summary;
  units: Unit[];
  maxProj: Img;
  t: number;
  setT: (t: number) => void;
  sel: number | null;
  setSel: (id: number | null) => void;
  cache: FrameCache;
};

export const RunCtx = createContext<RunState | null>(null);

export function useRun(): RunState {
  const s = useContext(RunCtx);
  if (!s) throw new Error("useRun outside a run page");
  return s;
}

export function sourceOptions(a: Available): SegOption<Source>[] {
  return [
    { value: "max", label: "Max projection" },
    { value: "raw", label: "Raw", disabled: !a.raw, title: a.raw ? undefined : "raw videos not found (moved or unplugged)" },
    {
      value: "pre",
      label: "Preprocessed",
      disabled: !a.pre,
      title: a.pre ? undefined : "tile store not kept (run with --keep-store)",
    },
    { value: "recon", label: "Reconstruction" },
  ];
}
