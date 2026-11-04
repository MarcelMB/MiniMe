import { orderAvis } from "../stream/order";
import type { Dir } from "./dir";
import { HttpError } from "./router";
import { wasm } from "./wasm";

export type SegIndex = { n: number; height: number; width: number; offsets: number[]; sizes: number[] };
/** Frame offsets and sizes of each AVI (wasm `AviFile.parse`). */
export type Indexer = (files: File[]) => Promise<SegIndex[]>;

const workerIndexer: Indexer = (files) =>
  new Promise((ok, no) => {
    const w = new Worker(new URL("./avi.worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (e: MessageEvent<{ index?: SegIndex[]; error?: string }>) => {
      w.terminate();
      if (e.data.index) ok(e.data.index);
      else no(new HttpError(400, `cannot read the videos: ${e.data.error}`));
    };
    w.onerror = (e) => {
      w.terminate();
      no(new HttpError(400, `cannot read the videos: ${e.message}`));
    };
    w.postMessage(files);
  });

/** Reads each file whole; only for tests (Node has no Worker), where the AVIs are tiny. */
const wholeFileIndexer: Indexer = async (files) => {
  const { AviFile } = await wasm();
  const out: SegIndex[] = [];
  for (const file of files) {
    const all = new Uint8Array(await file.arrayBuffer());
    const avi = AviFile.parse((off: number, len: number) => all.subarray(off, off + len), file.size);
    const n = avi.n_frames();
    out.push({
      n,
      height: avi.height(),
      width: avi.width(),
      offsets: Array.from({ length: n }, (_, i) => avi.offset(i)),
      sizes: Array.from({ length: n }, (_, i) => avi.size(i)),
    });
    avi.free();
  }
  return out;
};
const defaultIndexer: Indexer = (files) => (typeof Worker === "undefined" ? wholeFileIndexer : workerIndexer)(files);

const NAMING_RULE =
  "each video's file name must start with a number (0.avi, 1.avi, ... or 0_session.avi); the videos are read in the order of that number, so 2.avi comes before 10.avi, and .avi files whose name does not start with a number are ignored";

type Seg = { file: File; n: number; offsets: number[]; sizes: number[] };

/** The raw videos of a folder: uncompressed 8-bit AVIs in name-number order (io.list_segments). */
export class RawVideo {
  static readonly NAMING_RULE = NAMING_RULE;
  private constructor(
    private segs: Seg[],
    readonly height: number,
    readonly width: number,
    readonly names: string[],
    readonly skipped: string[],
  ) {}
  get frames(): number {
    return this.segs.reduce((s, g) => s + g.n, 0);
  }
  get bytes(): number {
    return this.segs.reduce((s, g) => s + g.file.size, 0);
  }

  static async open(dir: Dir, index: Indexer = defaultIndexer): Promise<RawVideo> {
    const avis: File[] = [];
    for (const e of await dir.entries())
      if (e.kind === "file" && e.name.toLowerCase().endsWith(".avi") && !e.name.startsWith("."))
        avis.push(await dir.file(e.name));
    const ordered = orderAvis(avis);
    const skipped = avis
      .filter((f) => !ordered.includes(f))
      .map((f) => f.name)
      .sort();
    if (!ordered.length) throw new HttpError(404, `no videos in ${dir.name}: ${NAMING_RULE}`);
    const num = (f: File) => Number(/^(\d+)/.exec(f.name)?.[1]);
    ordered.forEach((f, i) => {
      if (i > 0 && num(f) === num(ordered[i - 1]))
        throw new HttpError(
          400,
          `${ordered[i - 1].name} and ${f.name} in ${dir.name} both start with ${num(f)}; rename one: ${NAMING_RULE}`,
        );
    });
    const idx = await index(ordered);
    const [H, W] = [idx[0].height, idx[0].width];
    idx.forEach((s, i) => {
      if (s.height !== H || s.width !== W)
        throw new HttpError(400, `${ordered[i].name}: ${s.height}x${s.width} frames, ${ordered[0].name} has ${H}x${W}`);
    });
    const segs = idx.map((s, i) => ({ file: ordered[i], n: s.n, offsets: s.offsets, sizes: s.sizes }));
    return new RawVideo(
      segs,
      H,
      W,
      ordered.map((f) => f.name),
      skipped,
    );
  }

  async frame(t: number): Promise<Uint8Array> {
    if (!(Number.isInteger(t) && t >= 0 && t < this.frames))
      throw new HttpError(400, `frame ${t} out of range 0..${this.frames - 1}`);
    let i = 0;
    while (t >= this.segs[i].n) t -= this.segs[i++].n;
    const s = this.segs[i];
    const buf = new Uint8Array(await s.file.slice(s.offsets[t], s.offsets[t] + s.sizes[t]).arrayBuffer());
    const [H, W] = [this.height, this.width];
    if (buf.length === H * W) return buf;
    const stride = (W + 3) & ~3; // rows padded to 4 bytes
    const out = new Uint8Array(H * W);
    for (let y = 0; y < H; y++) out.set(buf.subarray(y * stride, y * stride + W), y * W);
    return out;
  }

  async clip(t0: number, n: number): Promise<Uint8Array[]> {
    return Promise.all(Array.from({ length: n }, (_, k) => this.frame(t0 + k)));
  }
}
