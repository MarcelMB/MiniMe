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
export type Place = { path: string; kind: "demo" | "run"; raw?: string; time: number };

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
  kind: "run" | "rerun" | "autotune";
  run_dir: string;
  tune: boolean;
  tune_args: Record<string, number>;
};

/** A window-2 rerun of a run (``reruns/<name>``). */
export type Rerun = {
  name: string;
  autotune: boolean;
  status: "done" | "failed" | "running";
  counts?: Record<string, number>;
  wall_total?: number | null;
  params?: Record<string, unknown>;
  changed?: Record<string, [unknown, unknown]>;
  error?: string[];
};

export type Checks = { diameter: number; pnr_min: number; piece_frac: number; size_lo: number; size_hi: number; dup_corr: number; keep_dist: number };
export type CurvePoint = { value: unknown; good: number; flagged: number };
export type Score = { good: number; flagged: number; lost?: number; score: number } & Record<string, number>;
/** ``autotune.json`` of ``reruns/auto`` plus the sweep curves. */
export type Proposal = {
  diameter: number;
  diameter_source: string;
  min_dist_frac: number;
  tune_frames: number;
  segment: [number, number] | null;
  frames: number;
  checks: Checks;
  derived: Record<string, unknown>;
  kept: string[];
  proposed: Record<string, unknown>;
  run_values: Record<string, unknown>;
  at_edge: string[];
  score: Score;
  run_score: Score | null;
  better_than_run: boolean;
  search_error: string | null;
  n_trials: number;
  wall_s: number;
  curves: Record<string, CurvePoint[]>;
};
export type Flags = { good: number; flagged: number; reasons: Record<string, number[]>; checks: Checks };
export type Pairs = {
  corr_bins: number[];
  near_corr: number[];
  far_corr: number[];
  nn_bins: number[];
  nn_dist: number[];
  dup_corr: number;
  min_dist_frac: number;
  n_near: number;
  n_near_above: number;
  n_closer: number;
  far_p99: number | null;
};

export type ParamMeta = {
  name: string;
  group: string;
  type: string;
  default: unknown;
  help: string;
  common: boolean;
  window2: boolean;
};

export type RawInfo = {
  path: string;
  segments: number;
  frames: number;
  height: number;
  width: number;
  fps: number | null;
  bytes: number;
  names: string[]; // the videos in the order they are read
  skipped: string[]; // .avi files whose name does not start with a number
  naming_rule: string;
};

export type Zooms = { units: number[]; windows: { start: number; stop: number }[] };
