/** Python's `round(x, digits)`: half to even (exact for the decimal representations used here). */
export function roundHalfEven(x: number, digits = 0): number {
  const m = 10 ** digits;
  const y = x * m;
  const r = Math.round(y);
  const half = Math.abs(y % 1) === 0.5;
  return (half ? (r % 2 === 0 ? r : r - 1) : r) / m;
}
