/** `.avi` files whose name starts with a number, in that number's order (as `io.list_segments`). */
export function orderAvis<F extends { name: string }>(files: F[]): F[] {
  const num = (f: F) => /^(\d+)/.exec(f.name)?.[1];
  return files
    .filter((f) => /\.avi$/i.test(f.name) && num(f) !== undefined)
    .sort((a, b) => Number(num(a)) - Number(num(b)));
}

/** Frame rate and time left from frames done since the start. */
export function rate(done: number, total: number, seconds: number): { fps: number; eta: number } {
  const fps = seconds > 0 ? done / seconds : 0;
  return { fps, eta: fps > 0 ? (total - done) / fps : Infinity };
}
