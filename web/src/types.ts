export type Stage = { stage: string; seconds: number };
export type Available = { raw: boolean; pre: boolean; seeds_saved: boolean; init_saved: boolean };

export type RunListing = {
  id: string;
  name: string;
  path: string;
  units: number | null;
  T: number;
  fps: number | null;
  wall_total: number | null;
  mtime: number;
  error?: string;
};

export type Summary = RunListing & {
  H: number;
  W: number;
  tile: number;
  ranks: number[];
  raw_dir: string | null;
  params: Record<string, unknown>;
  params_changed: Record<string, { value: unknown; default: unknown }>;
  stages: Stage[];
  counts: Record<string, number>;
  peak_rss: number | null;
  peak_rss_children: number | null;
  bytes: Record<string, number>;
  workers: number | null;
  motion: { quality_median?: number; max_abs_shift_px?: number };
  available: Available;
};

export type Unit = {
  id: number;
  centroid: [number, number];
  size_px: number;
  peak: number | null;
  polygons: number[][][];
};
export type UnitsResponse = { stage: "final" | "init"; units: Unit[] };
export type Seeds = { init: number[][]; pnr: number[][]; merged: number[][] };

export type FsEntry = { name: string; path: string; has_avi: boolean; is_run: boolean };
export type FsListing = { path: string; parent: string; entries: FsEntry[] };

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled" | "interrupted";
export type Job = {
  id: string;
  raw_dir: string;
  name: string;
  params: Record<string, unknown>;
  keep_store: boolean;
  status: JobStatus;
  out_dir: string;
  created: number;
  started: number | null;
  ended: number | null;
  error: string;
};

export type ParamMeta = {
  name: string;
  group: string;
  type: string;
  default: unknown;
  help: string;
  common: boolean;
};

export type RawInfo = {
  path: string;
  segments: number;
  frames: number;
  height: number;
  width: number;
  fps: number | null;
  bytes: number;
};

export type Zooms = { units: number[]; windows: { start: number; stop: number }[] };
