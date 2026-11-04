function fftPow2(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const [wr, wi] = [Math.cos(ang), Math.sin(ang)];
    for (let i = 0; i < n; i += len) {
      let [cr, ci] = [1, 0];
      for (let k = 0; k < len / 2; k++) {
        const [ar, ai] = [re[i + k], im[i + k]];
        const br = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const bi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ar + br;
        im[i + k] = ai + bi;
        re[i + k + len / 2] = ar - br;
        im[i + k + len / 2] = ai - bi;
        [cr, ci] = [cr * wr - ci * wi, cr * wi + ci * wr];
      }
    }
  }
}

/** Complex DFT of any length (unnormalised; `inverse` flips the sign only). */
function dft(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;
  if (n <= 1) return;
  if ((n & (n - 1)) === 0) {
    fftPow2(re, im, inverse);
    return;
  }
  let m = 1;
  while (m < 2 * n - 1) m <<= 1;
  const s = inverse ? 1 : -1;
  const cr = new Float64Array(n);
  const ci = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const a = (s * Math.PI * ((k * k) % (2 * n))) / n;
    cr[k] = Math.cos(a);
    ci[k] = Math.sin(a);
  }
  const ar = new Float64Array(m);
  const ai = new Float64Array(m);
  for (let k = 0; k < n; k++) {
    ar[k] = re[k] * cr[k] - im[k] * ci[k];
    ai[k] = re[k] * ci[k] + im[k] * cr[k];
  }
  const br = new Float64Array(m);
  const bi = new Float64Array(m);
  br[0] = cr[0];
  bi[0] = -ci[0];
  for (let k = 1; k < n; k++) {
    br[k] = br[m - k] = cr[k];
    bi[k] = bi[m - k] = -ci[k];
  }
  fftPow2(ar, ai, false);
  fftPow2(br, bi, false);
  for (let k = 0; k < m; k++) {
    const r = ar[k] * br[k] - ai[k] * bi[k];
    ai[k] = ar[k] * bi[k] + ai[k] * br[k];
    ar[k] = r;
  }
  fftPow2(ar, ai, true);
  for (let k = 0; k < n; k++) {
    const [xr, xi] = [ar[k] / m, ai[k] / m];
    re[k] = xr * cr[k] - xi * ci[k];
    im[k] = xr * ci[k] + xi * cr[k];
  }
}

export function rfft(x: Float64Array): { re: Float64Array; im: Float64Array } {
  const re = Float64Array.from(x);
  const im = new Float64Array(x.length);
  dft(re, im, false);
  const h = (x.length >> 1) + 1;
  return { re: re.slice(0, h), im: im.slice(0, h) };
}

export function irfft(re: Float64Array, im: Float64Array, n: number): Float64Array {
  const fr = new Float64Array(n);
  const fi = new Float64Array(n);
  for (let k = 0; k < re.length && k < n; k++) {
    fr[k] = re[k];
    fi[k] = im[k];
    if (k > 0 && n - k > k) {
      fr[n - k] = re[k];
      fi[n - k] = -im[k];
    }
  }
  if (n % 2 === 0) fi[n / 2] = 0;
  fi[0] = 0;
  dft(fr, fi, true);
  for (let i = 0; i < n; i++) fr[i] /= n;
  return fr;
}
