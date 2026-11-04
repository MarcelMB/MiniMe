/** numpy's `np.round(x, digits)` / `np.around`: multiply by 10**digits, round half to even on the
 * product, divide. A product that lands on an exact .5 counts as a tie even when the decimal
 * `x` is not one in binary (0.15 -> 0.2), so this is NOT Python's builtin `round`: use `pyRound`. */
export function roundHalfEven(x: number, digits = 0): number {
  const m = 10 ** digits;
  const y = x * m;
  const r = Math.round(y);
  const half = Math.abs(y % 1) === 0.5;
  return (half ? (r % 2 === 0 ? r : r - 1) : r) / m;
}

/** Python's builtin `round(x, digits)` for `digits` in 0..100: correctly rounded on the exact
 * binary value of `x` (0.15 -> 0.1, 2.675 -> 2.67), a true tie (1.25, 2.5) goes to the even
 * last digit, and the sign of zero is kept (-0.5 -> -0). */
export function pyRound(x: number, digits = 0): number {
  if (digits < 0 || digits > 100) throw new RangeError("pyRound: digits must be in 0..100");
  if (!Number.isFinite(x) || Math.abs(x) >= 1e21) return x; // toFixed switches to exponent form; such x are integers
  const ax = Math.abs(x);
  // toFixed works on the exact binary value; its extra digits show whether the rest is exactly one half
  const exact = ax.toFixed(Math.min(100, digits + 30));
  const dot = exact.indexOf(".");
  const tie = /^50*$/.test(exact.slice(dot + 1 + digits));
  const lastKept = Number(exact[digits === 0 ? dot - 1 : dot + digits]);
  // toFixed rounds an exact tie up, so a tie with an even last digit is the truncated value instead
  const r = tie && lastKept % 2 === 0 ? Number(exact.slice(0, dot + 1 + digits)) : Number(ax.toFixed(digits));
  return x < 0 || Object.is(x, -0) ? -r : r;
}

/** Python's `str(x)` for a float, as it appears in main's error messages: `-1.0`, `2000.0`,
 * `2.5`, `nan`, `inf`. (Exponent forms, which Python writes from 1e16 and below 1e-4, are not
 * reproduced.) */
export function pyFloat(x: number): string {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x < 0 ? "-inf" : "inf";
  if (Object.is(x, -0)) return "-0.0";
  return Number.isInteger(x) && Math.abs(x) < 1e16 ? `${x}.0` : String(x);
}
