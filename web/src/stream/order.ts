/** `.avi` files whose name starts with a number, in that number's order (as `io.list_segments`). */
export function orderAvis<F extends { name: string }>(files: F[]): F[] {
  const num = (f: F) => /^(\d+)/.exec(f.name)?.[1];
  return files
    .filter((f) => /\.avi$/i.test(f.name) && num(f) !== undefined)
    .sort((a, b) => Number(num(a)) - Number(num(b)));
}
