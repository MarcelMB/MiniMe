import { Countdown, runLeft, StageRate } from "./eta";
import type { RunPlan, StageProgress } from "./params";

/** No new unit for this long (s) and the stage row says so instead of a time left. */
export const STALL_S = 5;

type Tracked = {
  p: StageProgress;
  rate: StageRate;
  countdown: Countdown;
  /** Wall-clock seconds the stage's own clock last read `p.seconds`: when `p` arrived, or the
   * stage started again in a later window (its clock stood still in between). */
  at: number;
  /** Wall-clock seconds `done` last grew (or the stage started). */
  advanced: number;
};

/** A browser run's live counters for the job page: each stage's rate and displayed time left
 * (on the stage's own clock, which only runs while the stage does), the whole run's time left
 * (on the wall clock), and stalls. `minime serve` sends no counters, so it stays empty there. */
export class LiveRun {
  private stages = new Map<string, Tracked>();
  private plan: RunPlan | null = null;
  private run = new Countdown();

  progress(p: StageProgress, wall: number): void {
    const t = this.stages.get(p.stage);
    const tracked: Tracked = t ?? { p, rate: new StageRate(), countdown: new Countdown(), at: wall, advanced: wall };
    if (t && p.done > t.p.done) tracked.advanced = wall;
    tracked.p = p;
    tracked.at = wall;
    tracked.rate.feed(p);
    const left = tracked.rate.left();
    if (left !== null) tracked.countdown.update(left, p.seconds);
    this.stages.set(p.stage, tracked);
    const run = runLeft(new Map([...this.stages].map(([k, v]) => [k, v.p])), this.plan);
    if (run !== null) this.run.update(run, wall);
  }

  /** A stage (or its next window) began at wall-clock `t`. */
  started(stage: string, t: number): void {
    const tracked = this.stages.get(stage);
    if (!tracked) return;
    tracked.at = Math.max(tracked.at, t);
    tracked.advanced = Math.max(tracked.advanced, t);
  }

  setPlan(plan: RunPlan): void {
    this.plan = plan;
  }

  hasPlan(): boolean {
    return this.plan !== null;
  }

  get(stage: string): StageProgress | null {
    return this.stages.get(stage)?.p ?? null;
  }

  /** Units per second (smoothed) of the stage, null before it is known. */
  perSecond(stage: string): number | null {
    return this.stages.get(stage)?.rate.perSecond() ?? null;
  }

  /** The running stage's displayed seconds left, null while estimating. */
  stageLeft(stage: string, wall: number): number | null {
    const t = this.stages.get(stage);
    return t ? t.countdown.left(t.p.seconds + Math.max(0, wall - t.at)) : null;
  }

  /** Seconds since the running stage last did a unit, when that is a stall (not for steps). */
  stalled(stage: string, wall: number): number | null {
    const t = this.stages.get(stage);
    if (!t || t.p.unit === "steps") return null;
    const idle = wall - t.advanced;
    return idle > STALL_S ? idle : null;
  }

  runLeft(wall: number): number | null {
    return this.run.left(wall);
  }
}
