import { createContext, useContext } from "react";
import type { SegOption } from "../controls";
import type { Img } from "../draw";
import type { FrameCache } from "../frames";
import type { Available, Rerun, Summary, Unit } from "../types";

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
  /** The window-2 rerun whose seeds, units and traces the pages show (null: the run). */
  rerun: string | null;
  setRerun: (name: string | null) => void;
  reruns: Rerun[];
  refreshReruns: () => void;
};

/** Query string that points a data route at ``rerun`` (empty for the run itself). */
export function rerunQuery(rerun: string | null, first = true): string {
  if (!rerun) return "";
  return `${first ? "?" : "&"}rerun=${encodeURIComponent(rerun)}`;
}

export const RunCtx = createContext<RunState | null>(null);

export function useRun(): RunState {
  const s = useContext(RunCtx);
  if (!s) throw new Error("useRun outside a run page");
  return s;
}

/** The run and every finished rerun, for the switcher in the top area. */
export function rerunOptions(reruns: Rerun[]): SegOption<string>[] {
  return [
    { value: "run", label: "run" },
    ...reruns.map((r) => ({
      value: r.name,
      label: r.name,
      disabled: r.status !== "done",
      title: r.status === "done" ? undefined : `this rerun ${r.status === "failed" ? "failed" : "is still running"}`,
    })),
  ];
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
