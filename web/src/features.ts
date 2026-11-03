import type { ComponentType } from "react";

/** What this build offers; the browser build swaps this module for `browser/features.ts`. */
export type Features = { tune: boolean; RunActions: ComponentType<{ runId: string }> | null };

export const FEATURES: Features = { tune: true, RunActions: null };
