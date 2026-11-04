import type { Features } from "../features";
import { DownloadRun } from "./DownloadRun";

export type { Features } from "../features";
export const FEATURES: Features = { tune: false, RunActions: DownloadRun };
