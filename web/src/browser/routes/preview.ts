import { sessionFor } from "../../stream/session";
import type { RawVideo } from "../avi";
import { pyFloat, pyRepr } from "../num";
import defaults from "../params.defaults.json";
import { HttpError, router } from "../router";
import { wasm } from "../wasm";
import { openRaw } from "./raw";

/** `preview.MOTION_FRAMES`. */
const MOTION_FRAMES = 150;

/** `Params` fields, and the keys of a run's params.json that `Params.from_dict` drops. */
const FIELDS = new Set(defaults.params.map((p) => p.name));
const RUN_METADATA = [
  "raw_dir",
  "raw_hash",
  "fps",
  "minime_version",
  "caanfast_version",
  "cafast_version",
  "fastminian_version",
  "source_run",
];

type Body = { raw_dir: string; t: number; params: Record<string, unknown> };

/** `routes_jobs.PreviewBody`: 422 for what its model would not take. */
function previewBody(raw: unknown): Body {
  const b = (raw ?? {}) as Record<string, unknown>;
  const params = b.params ?? {};
  if (
    typeof b.raw_dir !== "string" ||
    !Number.isInteger(b.t) ||
    typeof params !== "object" ||
    params === null ||
    Array.isArray(params)
  )
    throw new HttpError(422, "raw_dir must be text, t a whole number and params an object");
  return { raw_dir: b.raw_dir, t: b.t as number, params: params as Record<string, unknown> };
}

/** Python's `repr` of a JSON value, for `Params.__post_init__`'s message. */
function pyJson(v: unknown): string {
  if (typeof v === "string") return pyRepr(v);
  if (v === null) return "None";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : pyFloat(v);
  return JSON.stringify(v);
}

/** The checks of `Params.from_dict` on the request's parameters (400, main's text). */
function checkParams(params: Record<string, unknown>): Record<string, unknown> {
  const d = { ...params };
  for (const k of RUN_METADATA) delete d[k];
  if ("sparse_penal" in d)
    throw new HttpError(
      400,
      "params: sparse_penal was replaced by sparse_snr = sparse_penal * sqrt(T) (T = frames of the session it was tuned on)",
    );
  const unknown = Object.keys(d)
    .filter((k) => !FIELDS.has(k))
    .sort();
  if (unknown.length) throw new HttpError(400, `unknown parameter(s) in params: ${unknown.join(", ")}`);
  // motion._prep only knows 2x downsampling, but estimate scales shifts by motion_down
  if ("motion_down" in d && d.motion_down !== 1 && d.motion_down !== 2)
    throw new HttpError(400, `motion_down must be 1 or 2, not ${pyJson(d.motion_down)}`);
  return d;
}

/** `routes_jobs._preview_args`: the raw videos, the frame in range, and the pass-1 parameters as
 * JSON for the wasm core (the request's over the defaults). */
async function open(raw: unknown): Promise<{ v: RawVideo; t: number; params: string }> {
  const b = previewBody(raw);
  const { v } = await openRaw(b.raw_dir);
  if (b.t < 0 || b.t >= v.frames) throw new HttpError(400, `frame ${b.t} out of range 0..${v.frames - 1}`);
  const params = checkParams(b.params);
  return { v, t: b.t, params: JSON.stringify(sessionFor(v.height, v.width, params).pass1) };
}

/** Runs a wasm export for the request. What it returns as an error (a `JsError`, a plain `Error`:
 * an odd frame size for the band-pass image, a parameter of the wrong type) is the request's
 * fault, 400 with its message. A trap (`WebAssembly.RuntimeError`, which may leave the instance
 * unusable), an allocation failure (`RangeError`) or a call that went wrong (`TypeError`) is not:
 * it propagates and the router answers 500. */
export function wasmCall<T>(run: () => T): T {
  try {
    return run();
  } catch (e) {
    if (e instanceof Error && Object.getPrototypeOf(e) === Error.prototype) throw new HttpError(400, e.message);
    throw e;
  }
}

/** `preview.clip_start` and `preview.read_clip`'s length: the first frame of an `n`-frame clip
 * around `t` that stays inside `t`'s video (it starts at `t` unless fewer than `n` frames follow),
 * and how many frames that video leaves from there (fewer than `n` for a short video). */
export function clipStart(segmentFrames: number[], t: number, n: number): { t0: number; n: number } {
  let start = 0;
  for (const len of segmentFrames) {
    const end = start + len;
    if (t < end) {
      const t0 = Math.max(start, Math.min(t, end - n));
      return { t0, n: Math.min(n, end - t0) };
    }
    start = end;
  }
  throw new HttpError(400, `frame ${t} is beyond the ${start} raw frames`);
}

/** `POST /raw/preview`: frame `t` through the pass-1 steps, `(4, H, W)` uint8 (`preview.PANELS`). */
router.add("POST", "/raw/preview", async (q) => {
  const { v, t, params } = await open(q.body);
  const frame = await v.frame(t);
  const m = await wasm();
  const out = wasmCall(() => m.preview_panels(frame, v.height, v.width, params));
  return { array: { shape: [4, v.height, v.width], dtype: "uint8", data: out } };
});

/** `POST /raw/motion`: shifts of a clip around `t`; the template comes from the clip. */
router.add("POST", "/raw/motion", async (q) => {
  const { v, t, params } = await open(q.body);
  const { t0, n } = clipStart(v.segmentFrames, t, MOTION_FRAMES);
  const clip = new Uint8Array(n * v.height * v.width);
  try {
    (await v.clip(t0, n)).forEach((f, i) => {
      clip.set(f, i * f.length);
    });
  } catch (e) {
    // main's `read_clip` raises OSError, which the route answers with 400
    throw new HttpError(400, `cannot read frame ${t0} of the videos: ${e instanceof Error ? e.message : e}`);
  }
  const m = await wasm();
  return { json: JSON.parse(wasmCall(() => m.motion_check(clip, n, v.height, v.width, t0, params))) };
});
