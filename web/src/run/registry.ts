import type { JSX } from "react";
import type { Page } from "../router";
import { Frames } from "./pages/Frames";
import { Motion } from "./pages/Motion";
import { Overview } from "./pages/Overview";

export const PAGE_LABELS: Record<Page, string> = {
  overview: "Overview",
  frames: "Frames",
  motion: "Motion",
  compression: "Compression",
  seeds: "Seeds",
  units: "Units",
  traces: "Traces",
};

/** Stage pages by route; each page task adds its entry. Unregistered pages are hidden. */
export const PAGE_VIEWS: Partial<Record<Page, () => JSX.Element>> = {
  overview: Overview,
  frames: Frames,
  motion: Motion,
};
