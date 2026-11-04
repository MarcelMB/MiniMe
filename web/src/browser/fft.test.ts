import { expect, it } from "vitest";
import { irfft, rfft } from "./fft";

const dft = (x: Float64Array) => {
  const n = x.length;
  const re = new Float64Array((n >> 1) + 1);
  const im = new Float64Array((n >> 1) + 1);
  for (let k = 0; k < re.length; k++)
    for (let t = 0; t < n; t++) {
      re[k] += x[t] * Math.cos((2 * Math.PI * k * t) / n);
      im[k] -= x[t] * Math.sin((2 * Math.PI * k * t) / n);
    }
  return { re, im };
};

it.each([1, 2, 7, 8, 100, 301, 1024])("rfft matches the DFT and irfft inverts it (n=%i)", (n) => {
  const x = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 0.37) + (i % 5));
  const a = rfft(x);
  const b = dft(x);
  for (let k = 0; k < b.re.length; k++) {
    expect(a.re[k]).toBeCloseTo(b.re[k], 6);
    expect(a.im[k]).toBeCloseTo(b.im[k], 6);
  }
  const y = irfft(a.re, a.im, n);
  for (let i = 0; i < n; i++) expect(y[i]).toBeCloseTo(x[i], 8);
});
