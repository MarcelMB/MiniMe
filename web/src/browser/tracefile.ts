import { fileAt } from "./dir";
import { readNpyHeader } from "./npy";
import { HttpError } from "./router";
import type { RunInfo } from "./runs";

/** `cnmf/<kind>.npy` ((K, T) float32, C order), read by row slices. */
export async function tracesShape(r: RunInfo, kind: "C" | "YrA") {
  const file = await fileAt(r.dir, `cnmf/${kind}.npy`).catch(() => {
    throw new HttpError(404, `${kind} traces are not available for this run`);
  });
  const h = readNpyHeader(new Uint8Array(await file.slice(0, 4096).arrayBuffer()));
  if (h.descr !== "<f4" || h.shape.length !== 2) throw new HttpError(500, `cnmf/${kind}.npy is not (K, T) float32`);
  return { K: h.shape[0], T: h.shape[1], file, offset: h.offset };
}

export async function traceRow(r: RunInfo, kind: "C" | "YrA", unit: number): Promise<Float32Array> {
  const { K, T, file, offset } = await tracesShape(r, kind);
  if (!(0 <= unit && unit < K)) throw new HttpError(400, `unit ${unit} out of range 0..${K - 1}`);
  const a = offset + unit * T * 4;
  return new Float32Array(await file.slice(a, a + T * 4).arrayBuffer());
}
