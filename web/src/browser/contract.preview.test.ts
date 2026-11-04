import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { beforeAll, describe, expect, it } from "vitest";
import { postArray, postJSON } from "../api";
import { Folders } from "./folders";
import { memoryKV } from "./kv";
import { HttpError } from "./router";
import { clipStart, wasmCall } from "./routes/preview";
import { configure } from "./state";
import { expectClose, loadContract } from "./testing/contract";
import { DATA, setupE2eRun } from "./testing/e2eRun";
import { NodeDir } from "./testing/nodeDir";

const py = loadContract(`${DATA}/contract-python`);
let raw = "";
beforeAll(async () => {
  raw = (await setupE2eRun()).raw;
});

type Motion = { t0: number; frames: number; max_shift_px: number; quality_median: number; shifts: number[][] };

describe("previews match main's", () => {
  it("frame panels", async () => {
    const want = py.find("POST /raw/preview");
    const body = { ...(want.entry.body as object), raw_dir: raw };
    const got = await postArray("/raw/preview", body);
    const w = want.npy();
    expect(got.shape).toEqual(w.shape);
    expect(got.dtype).toBe("uint8");
    let diff = 0;
    for (let i = 0; i < w.data.length; i++) diff = Math.max(diff, Math.abs(got.data[i] - (w.data[i] as number)));
    expect(diff).toBeLessThanOrEqual(1); // one grey level: float rounding in the u8 scaling
  });
  it("motion check", async () => {
    const want = py.find("POST /raw/motion");
    const got = await postJSON("/raw/motion", { ...(want.entry.body as object), raw_dir: raw });
    expectClose(got, want.json(), 1e-3);
  });
});

describe("the preview follows the parameters and the segment of the frame", () => {
  it("passes the preprocessing parameters on: a median size of 1 leaves the denoised panel as the raw one", async () => {
    const got = await postArray("/raw/preview", { raw_dir: raw, t: 20, params: { median_ksize: 1 } });
    const px = 64 * 64;
    expect(got.data.slice(0, px)).toEqual(got.data.slice(px, 2 * px));
    const dflt = await postArray("/raw/preview", { raw_dir: raw, t: 20, params: {} });
    expect(dflt.data.slice(0, px)).toEqual(got.data.slice(0, px));
    expect(dflt.data.slice(px, 2 * px)).not.toEqual(got.data.slice(px, 2 * px));
  });

  it("checks the motion on the frames of t's own segment", async () => {
    // the e2e video is three segments of 100 frames: a clip never crosses into the next one
    const at = async (t: number) => await postJSON<Motion>("/raw/motion", { raw_dir: raw, t, params: {} });
    for (const [t, t0] of [
      [0, 0],
      [99, 0],
      [100, 100],
      [299, 200],
    ]) {
      const got = await at(t);
      expect([t, got.t0, got.frames, got.shifts.length]).toEqual([t, t0, 100, 100]);
    }
  });

  it("starts a clip at t, moved back to fit the segment, and cuts it to a short segment", () => {
    const frames = [300, 50, 200];
    expect(clipStart(frames, 0, 150)).toEqual({ t0: 0, n: 150 });
    expect(clipStart(frames, 120, 150)).toEqual({ t0: 120, n: 150 });
    expect(clipStart(frames, 290, 150)).toEqual({ t0: 150, n: 150 }); // 150..299
    expect(clipStart(frames, 299, 150)).toEqual({ t0: 150, n: 150 });
    expect(clipStart(frames, 300, 150)).toEqual({ t0: 300, n: 50 }); // a 50-frame segment
    expect(clipStart(frames, 349, 150)).toEqual({ t0: 300, n: 50 });
    expect(clipStart(frames, 350, 150)).toEqual({ t0: 350, n: 150 }); // 350..499
    expect(clipStart(frames, 549, 150)).toEqual({ t0: 400, n: 150 }); // 400..549
    expect(clipStart(frames, 549, 20)).toEqual({ t0: 530, n: 20 });
  });
});

describe("the preview answers as main does when the request is wrong", () => {
  const ask = (body: object) => postArray("/raw/preview", { raw_dir: raw, t: 0, params: {}, ...body });

  it("rejects a frame outside the session with 400 and main's text", async () => {
    await expect(ask({ t: 300 })).rejects.toMatchObject({ status: 400, message: "frame 300 out of range 0..299" });
    await expect(ask({ t: -1 })).rejects.toMatchObject({ status: 400, message: "frame -1 out of range 0..299" });
    await expect(postJSON("/raw/motion", { raw_dir: raw, t: 300, params: {} })).rejects.toMatchObject({
      status: 400,
      message: "frame 300 out of range 0..299",
    });
  });

  it("rejects a body of the wrong type with 422, as main's request model does", async () => {
    await expect(ask({ t: 1.5 })).rejects.toMatchObject({ status: 422 });
    await expect(ask({ t: "7" })).rejects.toMatchObject({ status: 422 });
    await expect(ask({ raw_dir: 7 })).rejects.toMatchObject({ status: 422 });
    await expect(ask({ params: [] })).rejects.toMatchObject({ status: 422 });
  });

  it("rejects parameters as Params.from_dict does, with 400", async () => {
    await expect(ask({ params: { nope: 1, also_not: 2 } })).rejects.toMatchObject({
      status: 400,
      message: "unknown parameter(s) in params: also_not, nope",
    });
    await expect(ask({ params: { sparse_penal: 0.01 } })).rejects.toMatchObject({
      status: 400,
      message: expect.stringMatching(/^params: sparse_penal was replaced by sparse_snr/),
    });
    await expect(ask({ params: { motion_down: 3 } })).rejects.toMatchObject({
      status: 400,
      message: "motion_down must be 1 or 2, not 3",
    });
    // run metadata that a params.json carries is ignored, and so are the parameters pass 1 does not use
    await expect(ask({ params: { raw_dir: "x", fps: 20, merge_corr: 0.5, workers: 4 } })).resolves.toMatchObject({
      shape: [4, 64, 64],
    });
  });

  it("rejects an even median window with 400: cv2.medianBlur takes odd sizes only", async () => {
    for (const k of [4, 6, 8]) {
      const want = { status: 400, message: `median_ksize must be odd, got ${k}` };
      await expect(ask({ params: { median_ksize: k } })).rejects.toMatchObject(want);
      await expect(postJSON("/raw/motion", { raw_dir: raw, t: 0, params: { median_ksize: k } })).rejects.toMatchObject(
        want,
      );
    }
    // a parameter of the wrong type is the wasm core's refusal, also the request's fault
    await expect(ask({ params: { median_ksize: "seven" } })).rejects.toMatchObject({
      status: 400,
      message: expect.stringMatching(/^params: invalid type/),
    });
    // 1 skips the median, odd sizes filter
    for (const k of [1, 3, 5]) {
      await expect(ask({ params: { median_ksize: k } })).resolves.toMatchObject({ shape: [4, 64, 64] });
    }
  });

  it("answers a folder without videos with 404, as /raw/info does, and one that is not known too", async () => {
    const top = mkdtempSync(`${tmpdir()}/preview-`);
    const folders = new Folders(memoryKV(), async () => null);
    const name = await folders.add(new NodeDir(top), "picked");
    configure({ folders });
    await expect(postArray("/raw/preview", { raw_dir: name, t: 0, params: {} })).rejects.toMatchObject({
      status: 404,
      message: expect.stringMatching(/^no videos in .*: each video's file name must start with a number/),
    });
    await expect(postJSON("/raw/motion", { raw_dir: "nowhere/raw", t: 0, params: {} })).rejects.toMatchObject({
      status: 404,
    });
  });
});

describe("wasmCall", () => {
  it("answers 400 with the message of an error the export returns (a JsError)", () => {
    expect(() =>
      wasmCall(() => {
        throw new Error("params: missing field `median_ksize`");
      }),
    ).toThrow(expect.objectContaining({ status: 400, message: "params: missing field `median_ksize`" }));
    expect(wasmCall(() => 7)).toBe(7);
  });

  it("lets a trap, an allocation failure and a call that went wrong through, for the router's 500", () => {
    for (const e of [
      new WebAssembly.RuntimeError("unreachable"),
      new RangeError("out of memory"),
      new TypeError("x"),
    ]) {
      let got: unknown;
      try {
        wasmCall(() => {
          throw e;
        });
      } catch (x) {
        got = x;
      }
      expect(got).toBe(e);
      expect(got).not.toBeInstanceOf(HttpError);
    }
  });
});
