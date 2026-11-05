import type { Features } from "../features";
import { DownloadRun } from "./DownloadRun";
import { SpeedPanel } from "./SpeedPanel";

export type { Features } from "../features";
export const FEATURES: Features = { tune: true, RunActions: DownloadRun, SpeedPanel };
