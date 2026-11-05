import type { ComponentType } from "react";
import type { Device } from "./gpu/plan";

/** New run's speed card (the browser build's): the chosen videos and parameters, and the device. */
export type SpeedPanelProps = {
  raw: string;
  frames: number;
  width: number;
  height: number;
  params: Record<string, unknown>;
  device: Device;
  onDevice: (d: Device) => void;
};

/** What this build offers; the browser build swaps this module for `browser/features.ts`. */
export type Features = {
  tune: boolean;
  RunActions: ComponentType<{ runId: string }> | null;
  SpeedPanel: ComponentType<SpeedPanelProps> | null;
};

export const FEATURES: Features = { tune: true, RunActions: null, SpeedPanel: null };
