/** Next frame while playing: ``speed`` x real time at ``fps``; stops at the last frame. */
export function advance(
  t: number,
  T: number,
  speed: number,
  fps: number,
  dtMs: number,
  carry: number,
): { t: number; carry: number; ended: boolean } {
  const step = (speed * fps * dtMs) / 1000 + carry;
  const whole = Math.floor(step);
  const next = Math.min(T - 1, t + whole);
  const ended = next >= T - 1;
  return { t: next, carry: ended ? 0 : step - whole, ended };
}
