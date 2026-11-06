/** The synthetic demo session (GRIN V4, 30 min, CC0): `src/minime/demo.py`'s "grin" dataset on UCLA
 * Dataverse, and a download of it straight into a folder the user picked or into browser storage.
 *
 * Dataverse's bot check answers a browser's cross-origin request with a challenge page that has no
 * CORS header, so the hosted page cannot fetch from it: the download comes from the same files on
 * Hugging Face (`hfSource`), which allows CORS; Dataverse stays the citable copy and the manual route. */

export const DATAVERSE = "https://dataverse.ucla.edu";
export const DEMO_DOI = "doi:10.25346/S6WHMSXK";
/** The dataset's landing page (the manual download). Not the doi.org link: Dataverse's
 * `/citation` redirect behind it drops a slash (`dataverse.ucla.edudataset.xhtml`), a host that
 * does not exist. */
export const DEMO_PAGE = `${DATAVERSE}/dataset.xhtml?persistentId=${DEMO_DOI}`;
export const LISTING_URL = `${DATAVERSE}/api/datasets/:persistentId/?persistentId=${DEMO_DOI}`;
export const fileUrl = (id: number) => `${DATAVERSE}/api/access/datafile/${id}`;

/** The Hugging Face mirror: the dataset's files in the same layout (`raw/N.avi` and the top level). */
export const HF_REPO = "MarcelMB/minime-demo";
export const HF_PAGE = `https://huggingface.co/datasets/${HF_REPO}`;
export const HF_LISTING_URL = `https://huggingface.co/api/datasets/${HF_REPO}/tree/main?recursive=true`;
export const hfFileUrl = (path: string) =>
  `${HF_PAGE}/resolve/main/${path.split("/").map(encodeURIComponent).join("/")}`;

/** One 1000-frame segment (600 x 600 px, 20 fps), for the size shown before the listing is in. */
export const SEGMENT_BYTES = 360_029_862;

export type DemoKind = "short" | "full";
export type Demo = {
  segments: number;
  minutes: number;
  /** The session folder the files go to (`raw/N.avi` and the top-level files). */
  folder: string;
  /** Top-level files next to `raw/`: `meta.json` gives the frame rate; the truth is for 30 min. */
  extras: readonly string[];
};
export const DEMOS: Record<DemoKind, Demo> = {
  short: { segments: 6, minutes: 5, folder: "minime-demo-5min", extras: ["README.md", "meta.json"] },
  full: { segments: 36, minutes: 30, folder: "minime-demo-30min", extras: ["README.md", "meta.json", "truth.npz"] },
};

/** The bytes a demo is about to take, before the listing says it exactly. */
export const approxBytes = (kind: DemoKind) => DEMOS[kind].segments * SEGMENT_BYTES;

export type DataFile = { id: number; filename: string; filesize: number; dir: string };

/** Where the demo's files come from: their list (`dir` "raw" for the videos, "" for the top level)
 * and each one's URL. */
export type DemoSource = {
  list(fetchFn: typeof fetch, signal?: AbortSignal): Promise<DataFile[]>;
  url(f: DataFile): string;
};
type Listing = {
  data: {
    latestVersion: {
      files: { directoryLabel?: string; dataFile: { id: number; filename: string; filesize: number } }[];
    };
  };
};

/** `demo._files`: the dataset's files with their folder ("" for the top level). */
export function listingFiles(listing: Listing): DataFile[] {
  return listing.data.latestVersion.files.map((f) => ({
    id: f.dataFile.id,
    filename: f.dataFile.filename,
    filesize: f.dataFile.filesize,
    dir: f.directoryLabel ?? "",
  }));
}

/** `demo.raw_files`: the `N.avi` files, by their number. */
export function rawFiles(files: DataFile[]): DataFile[] {
  return files
    .filter((f) => /^\d+\.avi$/.test(f.filename))
    .sort((a, b) => Number.parseInt(a.filename, 10) - Number.parseInt(b.filename, 10));
}

/** A file to fetch and where it goes, relative to the folder the user chose. */
export type Planned = { file: DataFile; path: string[] };

/** The demo's first `segments` videos into `<folder>/raw/` and its extras into `<folder>/`. */
export function plan(files: DataFile[], kind: DemoKind): Planned[] {
  const d = DEMOS[kind];
  const raw = rawFiles(files).slice(0, d.segments);
  if (raw.length < d.segments) throw new Error(`the demo dataset lists ${raw.length} videos, ${d.segments} expected`);
  const extras = d.extras.map((name) => {
    const f = files.find((x) => x.dir === "" && x.filename === name);
    if (!f) throw new Error(`the demo dataset has no ${name}`);
    return f;
  });
  return [
    ...raw.map((file) => ({ file, path: [d.folder, "raw", file.filename] })),
    ...extras.map((file) => ({ file, path: [d.folder, file.filename] })),
  ];
}

/** Where the files go: a picked folder or browser storage (`handleTarget`), or memory in tests. */
export interface Target {
  /** The size of the file at `path`, or null when there is none. */
  size(path: string[]): Promise<number | null>;
  /** A new file at `path`; it appears with its contents only on `close` (`abort` leaves nothing). */
  create(path: string[]): Promise<Sink>;
}
export type Sink = { write(chunk: Uint8Array): Promise<void>; close(): Promise<void>; abort(): Promise<void> };

/** A target in a directory handle. `createWritable` writes to a swap file the browser moves into
 * place on `close`, so a cancelled or broken download never looks like a finished file. */
export function handleTarget(root: FileSystemDirectoryHandle): Target {
  const dirOf = async (path: string[], create: boolean) => {
    let d = root;
    for (const p of path.slice(0, -1)) d = await d.getDirectoryHandle(p, { create });
    return d;
  };
  return {
    async size(path) {
      try {
        const f = await (await (await dirOf(path, false)).getFileHandle(path[path.length - 1])).getFile();
        return f.size;
      } catch {
        return null;
      }
    },
    async create(path) {
      const h = await (await dirOf(path, true)).getFileHandle(path[path.length - 1], { create: true });
      const w = await h.createWritable();
      return {
        write: (chunk) => w.write(chunk),
        close: () => w.close(),
        abort: () => w.abort(),
      };
    },
  };
}

export type Progress = {
  /** Bytes in place: skipped files count as done. */
  done: number;
  total: number;
  /** The file being fetched (1-based) of `count`, and its name. */
  index: number;
  count: number;
  file: string;
  skipped: number;
};

export type DownloadOptions = {
  /** Default: the Hugging Face mirror (a browser page cannot reach Dataverse; see the top of this file). */
  source?: DemoSource;
  fetch?: typeof fetch;
  signal?: AbortSignal;
  onProgress?: (p: Progress) => void;
  /** Called with the bytes still to fetch before the first one is; throw to refuse (no room). */
  beforeStart?: (need: number) => Promise<void> | void;
};

/** The dataset's files, from its Dataverse listing. */
export async function fetchListing(fetchFn: typeof fetch = fetch, signal?: AbortSignal): Promise<DataFile[]> {
  const r = await fetchFn(LISTING_URL, { signal });
  if (!r.ok) throw new Error(`Dataverse answered ${r.status} for the demo's file list`);
  return listingFiles((await r.json()) as Listing);
}

export const dataverseSource: DemoSource = { list: fetchListing, url: (f) => fileUrl(f.id) };

export type HfEntry = { type: "file" | "directory"; path: string; size: number; lfs?: { size: number } };

/** `HF_LISTING_URL`'s files as `DataFile`s (`id` is their place in the listing; Hugging Face has none). */
export function hfFiles(entries: HfEntry[]): DataFile[] {
  return entries
    .filter((e) => e.type === "file")
    .map((e, id) => {
      const i = e.path.lastIndexOf("/");
      return {
        id,
        filename: e.path.slice(i + 1),
        filesize: e.lfs?.size ?? e.size,
        dir: i < 0 ? "" : e.path.slice(0, i),
      };
    });
}

export const hfSource: DemoSource = {
  async list(fetchFn = fetch, signal) {
    const r = await fetchFn(HF_LISTING_URL, { signal });
    if (!r.ok) throw new Error(`Hugging Face answered ${r.status} for the demo's file list`);
    return hfFiles((await r.json()) as HfEntry[]);
  },
  url: (f) => hfFileUrl(f.dir ? `${f.dir}/${f.filename}` : f.filename),
};

/** Fetch one file into `target`, checking it is as long as the listing says. */
async function fetchFile(
  p: Planned,
  target: Target,
  url: string,
  fetchFn: typeof fetch,
  signal: AbortSignal | undefined,
  onBytes: (n: number) => void,
): Promise<void> {
  const r = await fetchFn(url, { signal });
  if (!r.ok || !r.body) throw new Error(`the server answered ${r.status} for ${p.file.filename}`);
  const sink = await target.create(p.path);
  const reader = r.body.getReader();
  let n = 0;
  try {
    for (;;) {
      signal?.throwIfAborted();
      const { done, value } = await reader.read();
      if (done) break;
      await sink.write(value);
      n += value.byteLength;
      onBytes(value.byteLength);
    }
    if (n !== p.file.filesize)
      throw new Error(`${p.file.filename} ended after ${n} of ${p.file.filesize} bytes; download it again`);
    await sink.close();
  } catch (e) {
    await reader.cancel().catch(() => undefined);
    await sink.abort().catch(() => undefined);
    throw e;
  }
}

/** Download demo `kind` into `target`, one file after another; a file already there with the
 * listed size is kept. Returns the files fetched and kept. A cancel (`signal`) throws its reason
 * and leaves the finished files in place, so the next download goes on from there. */
export async function downloadDemo(
  kind: DemoKind,
  target: Target,
  opts: DownloadOptions = {},
): Promise<{ fetched: number; skipped: number; bytes: number }> {
  const fetchFn = opts.fetch ?? fetch;
  const source = opts.source ?? hfSource;
  const todo = plan(await source.list(fetchFn, opts.signal), kind);
  const total = todo.reduce((s, p) => s + p.file.filesize, 0);
  const missing: Planned[] = [];
  let done = 0;
  for (const p of todo)
    if ((await target.size(p.path)) === p.file.filesize) done += p.file.filesize;
    else missing.push(p);
  const skipped = todo.length - missing.length;
  await opts.beforeStart?.(total - done);
  const report = (index: number, file: string) =>
    opts.onProgress?.({ done, total, index, count: missing.length, file, skipped });
  report(0, "");
  for (const [i, p] of missing.entries()) {
    opts.signal?.throwIfAborted();
    report(i + 1, p.file.filename);
    await fetchFile(p, target, source.url(p.file), fetchFn, opts.signal, (n) => {
      done += n;
      report(i + 1, p.file.filename);
    });
  }
  return { fetched: missing.length, skipped, bytes: total };
}
