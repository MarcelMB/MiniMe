import type { Features } from "../features";
import { DownloadRun } from "./DownloadRun";
import { PrivacyNote } from "./PrivacyNote";
import { SpeedPanel } from "./SpeedPanel";

export type { Features } from "../features";
export const FEATURES: Features = { tune: false, RunActions: DownloadRun, SpeedPanel, PrivacyNote };
