/** Next frame while playing: ``speed`` x real time at ``fps``; stops at the last frame. */
export function advance(
  t: number,
  T: number,
  speed: number,
  fps: number,
  dtMs: number,
  carry: number,
): { t: number; carry: number; ended: boolean } {
  const step = Math.max(0, (speed * fps * Math.max(0, dtMs)) / 1000 + carry);
  const whole = Math.floor(step);
  const next = Math.max(0, Math.min(T - 1, t + whole));
  const ended = next >= T - 1;
  return { t: next, carry: ended ? 0 : step - whole, ended };
}

/** One playback tick: wait (null) until the frame for ``t`` is drawn, then advance. */
export function playStep(
  t: number,
  drawn: number | null,
  T: number,
  speed: number,
  fps: number,
  dtMs: number,
  carry: number,
): { t: number; carry: number; ended: boolean } | null {
  return drawn === t ? advance(t, T, speed, fps, dtMs, carry) : null;
}

export const PREFETCH_WINDOW = 15;

/**
 * Prefetch neighbours at once while stepping or playing (``t`` moved within the window);
 * after a jump (scrubbing) wait until the position settles.
 */
export function prefetchTiming(prev: number | null, t: number, window = PREFETCH_WINDOW): "now" | "later" {
  return prev !== null && Math.abs(t - prev) <= window ? "now" : "later";
}
