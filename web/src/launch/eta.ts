import type { RunPlan, StageProgress } from "./params";

// Time left for the browser run's job page. The rules follow what established tools do (tqdm's and
// indicatif's time-weighted rate average, wget's warm-up, Firefox's damped display) and the
// progress-indicator guidance of NN/g, Microsoft and Apple: an honest rate, no estimate until it
// means something, rough wording, and a number that does not jump.

/** Seconds of a stage's own time after which an old rate sample counts 10% (indicatif: 15 s). */
export const TAU = 20;
/** No estimate before this many seconds of a stage and this share of its units (wget: 3 s). */
const WARM_S = 15;
const WARM_SHARE = 0.03;
/** Past this many seconds the slower of the recent and the overall rate is used (pessimistic). */
const LONG_S = 600;

/** A stage's rate in units per second of its own time, from its run-wide counters. */
export class StageRate {
  private last: StageProgress | null = null;
  private ema = 0;
  private weight = 0;

  feed(p: StageProgress): void {
    const l = this.last;
    this.last = p;
    if (!l) return;
    const [dt, dn] = [p.seconds - l.seconds, p.done - l.done];
    if (dt <= 0 || dn < 0) return;
    // weighted by time, not by sample: what happened TAU s ago counts 10%; `weight` debiases
    const w = 0.1 ** (dt / TAU);
    this.ema = this.ema * w + (dn / dt) * (1 - w);
    this.weight = this.weight * w + (1 - w);
  }

  /** Units per second; null before any unit is done. A stage of steps (one per window) has
   * the mean over its windows. */
  perSecond(): number | null {
    const p = this.last;
    if (!p || p.done <= 0 || p.seconds <= 0) return null;
    const overall = p.done / p.seconds;
    if (p.unit === "steps" || this.weight === 0) return overall;
    const recent = this.ema / this.weight;
    return p.seconds > LONG_S ? Math.min(recent, overall) : recent;
  }

  /** Seconds of the stage's own time it still needs; null while warming up. */
  left(): number | null {
    const p = this.last;
    const r = this.perSecond();
    if (!p || r === null || r <= 0) return null;
    const warm = p.unit === "steps" ? p.done >= 1 : p.seconds >= WARM_S && p.done / p.frames >= WARM_SHARE;
    return warm ? Math.max(0, p.frames - p.done) / r : null;
  }
}

/** A displayed time left that counts down on its own clock and only follows a new estimate
 * when it differs by more than 5 s or 5%: down by 30% of the gap, up by 10% (Firefox's
 * download estimate), straight to it when it is under half the shown value. */
export class Countdown {
  private deadline: number | null = null;

  update(estimate: number, clock: number): void {
    if (this.deadline === null) {
      this.deadline = clock + estimate;
      return;
    }
    const shown = Math.max(0, this.deadline - clock);
    const gap = estimate - shown;
    if (Math.abs(gap) < Math.max(5, 0.05 * shown)) return;
    const next = estimate < shown / 2 ? estimate : shown + gap * (gap < 0 ? 0.3 : 0.1);
    this.deadline = clock + next;
  }

  left(clock: number): number | null {
    return this.deadline === null ? null : Math.max(0, this.deadline - clock);
  }
}

/** Rough wording, rounded up: "a few seconds", 5 s steps under a minute, 10 s steps under ten
 * minutes, whole minutes under an hour, then hours and minutes. */
export function fmtLeft(s: number): string {
  if (s < 10) return "a few seconds";
  const by5 = Math.ceil(s / 5) * 5;
  if (by5 < 60) return `${by5} s`;
  const by10 = Math.ceil(s / 10) * 10;
  if (by10 < 600) return `${Math.floor(by10 / 60)}:${String(by10 % 60).padStart(2, "0")}`;
  const min = Math.ceil(s / 60);
  if (min < 60) return `${min} min`;
  return min % 60 ? `${Math.floor(min / 60)} h ${min % 60} min` : `${min / 60} h`;
}

/** The stages that run once per window: the run's time left needs each one's rate. */
const WINDOW_STAGES = ["pass1", "pass2", "seeds", "init", "cnmf"];

/** Seconds the whole run still needs: every stage's remaining units at its overall rate; the
 * streaming pass, before it starts, at pass 1's time per frame (both read every frame; the motion
 * stage is only the template's first frames now; the streaming pass is the faster one, so this
 * errs long); the stitch, a second or so, as
 * nothing. Null until every per-window stage has finished its first window. */
export function runLeft(stages: ReadonlyMap<string, StageProgress>, plan: RunPlan | null): number | null {
  if (!plan) return null;
  const left = (p: StageProgress) => (p.done > 0 ? (Math.max(0, p.frames - p.done) * p.seconds) / p.done : null);
  let total = 0;
  for (const name of WINDOW_STAGES) {
    const p = stages.get(name);
    const s = p && (p.unit !== "steps" || p.done >= 1) ? left(p) : null;
    if (s === null) return null;
    total += s;
  }
  const motion = stages.get("motion");
  if (motion) {
    const s = left(motion);
    if (s === null) return null;
    total += s;
  }
  const stream = stages.get("stream");
  if (stream) total += left(stream) ?? 0;
  else {
    const proxy = stages.get("pass1") ?? motion;
    if (!proxy || proxy.done <= 0) return null;
    total += (plan.frames * proxy.seconds) / proxy.done;
  }
  const stitch = stages.get("stitch");
  if (stitch) total += left(stitch) ?? 0;
  return total;
}
