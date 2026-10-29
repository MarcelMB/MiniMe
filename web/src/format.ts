export function fmtInt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

export function fmtDuration(s: number): string {
  if (s < 60) return `${s.toFixed(0)} s`;
  const total = Math.round(s);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const r = total % 60;
  const mm = h ? String(m).padStart(2, "0") : String(m);
  return `${h ? `${h}:` : ""}${mm}:${String(r).padStart(2, "0")}`;
}

export function fmtBytes(b: number): string {
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(0)} MB`;
  return `${(b / 1e3).toFixed(0)} kB`;
}

export function fmtClock(frame: number, fps: number | null): string {
  if (!fps) return `frame ${fmtInt(frame)}`;
  const s = Math.floor(frame / fps);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}:${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

const STAGES: Record<string, string> = {
  pass1: "Pass 1 · preprocessing + motion",
  pass2: "Pass 2 · compression",
  cnmf: "CNMF",
  initA: "init footprints",
  initC: "init traces",
  tune_run: "auto-tune · the run's values",
  tune_seeds: "auto-tune · seeds",
  tune_merge: "auto-tune · merge",
  tune_inita: "auto-tune · init footprints",
  tune_confirm: "auto-tune · confirm through CNMF",
  tune_sparse: "auto-tune · sparse_snr",
  tune_loop_merge: "auto-tune · loop merge",
  tune_decide: "auto-tune · decide on the whole session",
  tune_rerun: "auto-tune · final rerun",
};

export function stageLabel(name: string): string {
  return STAGES[name] ?? name.replace(/_/g, " ");
}
