import type { Features } from "../features";
import { DemoCard } from "./DemoCard";
import { DownloadRun } from "./DownloadRun";
import { PrivacyNote } from "./PrivacyNote";
import { SpeedPanel } from "./SpeedPanel";

export type { Features } from "../features";

/** Auto-tune stays out of the hosted build until it is released; `VITE_TUNE=1` builds it in
 * (`playwright.tune.config.ts`). */
const tune = import.meta.env.VITE_TUNE === "1";
export const FEATURES: Features = { tune, RunActions: DownloadRun, SpeedPanel, PrivacyNote, DemoCard };
