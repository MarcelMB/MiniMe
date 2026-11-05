import { describe, expect, it } from "vitest";
import {
  type DataFile,
  DEMOS,
  downloadDemo,
  fetchListing,
  fileUrl,
  LISTING_URL,
  type Progress,
  plan,
  rawFiles,
  type Target,
} from "./demo";

/** The dataset as Dataverse lists it: 36 videos under raw/ (in its own order) and the top-level files. */
function dataset(videoBytes = 10): { listing: unknown; bytes: Map<number, Uint8Array> } {
  const files: { directoryLabel?: string; dataFile: { id: number; filename: string; filesize: number } }[] = [];
  const bytes = new Map<number, Uint8Array>();
  const add = (id: number, filename: string, size: number, dir?: string) => {
    files.push({ ...(dir ? { directoryLabel: dir } : {}), dataFile: { id, filename, filesize: size } });
    bytes.set(
      id,
      Uint8Array.from({ length: size }, (_, i) => (id + i) & 255),
    );
  };
  // Dataverse sorts by name: 0, 10, 11, ..., 1, 20, ...
  for (const i of [...Array(36).keys()].sort((a, b) => String(a).localeCompare(String(b))))
    add(100 + i, `${i}.avi`, videoBytes, "raw");
  add(1, "meta.json", 4);
  add(2, "README.md", 3);
  add(3, "truth.npz", 5);
  return { listing: { data: { latestVersion: { files } } }, bytes };
}

/** `fetch` answering the listing and the files from `ds`, in `chunk`-byte pieces; `log` gets every URL. */
function fakeFetch(
  ds: ReturnType<typeof dataset>,
  opts: { chunk?: number; log?: string[]; short?: number; onChunk?: (id: number) => void } = {},
): typeof fetch {
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    opts.log?.push(url);
    init?.signal?.throwIfAborted();
    if (url === LISTING_URL) return new Response(JSON.stringify(ds.listing));
    const id = Number(url.split("/").pop());
    let data = ds.bytes.get(id);
    if (!data) return new Response("not found", { status: 404 });
    if (id === opts.short) data = data.slice(0, data.length - 1);
    const chunk = opts.chunk ?? 4;
    let at = 0;
    const body = new ReadableStream<Uint8Array>({
      pull(c) {
        if (at >= data.length) return c.close();
        opts.onChunk?.(id);
        c.enqueue(data.slice(at, at + chunk));
        at += chunk;
      },
    });
    return new Response(body);
  }) as typeof fetch;
}

/** A folder in memory: a file appears only on `close`, as a browser's `createWritable` does. */
function memoryTarget(files = new Map<string, Uint8Array>()): Target & { files: Map<string, Uint8Array> } {
  return {
    files,
    async size(path) {
      return files.get(path.join("/"))?.length ?? null;
    },
    async create(path) {
      const parts: Uint8Array[] = [];
      return {
        write: async (c) => {
          parts.push(c.slice());
        },
        close: async () => {
          const all = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
          let o = 0;
          for (const p of parts) {
            all.set(p, o);
            o += p.length;
          }
          files.set(path.join("/"), all);
        },
        abort: async () => {
          parts.length = 0;
        },
      };
    },
  };
}

describe("demo listing", () => {
  it("reads the Dataverse listing and orders the videos by their number", async () => {
    const ds = dataset();
    const files = await fetchListing(fakeFetch(ds));
    expect(files).toHaveLength(39);
    expect(rawFiles(files).map((f) => f.filename)).toEqual([...Array(36).keys()].map((i) => `${i}.avi`));
    expect(files.find((f) => f.filename === "meta.json")).toEqual({
      id: 1,
      filename: "meta.json",
      filesize: 4,
      dir: "",
    });
  });

  it("plans the short demo as the first six videos plus the small files, the full one with the truth", async () => {
    const files = await fetchListing(fakeFetch(dataset()));
    const short = plan(files, "short").map((p) => p.path.join("/"));
    expect(short).toEqual([
      ...[0, 1, 2, 3, 4, 5].map((i) => `minime-demo-5min/raw/${i}.avi`),
      "minime-demo-5min/README.md",
      "minime-demo-5min/meta.json",
    ]);
    const full = plan(files, "full");
    expect(full).toHaveLength(39);
    expect(full.at(-1)?.path).toEqual(["minime-demo-30min", "truth.npz"]);
  });

  it("refuses a listing without enough videos", () => {
    const files: DataFile[] = [{ id: 1, filename: "0.avi", filesize: 1, dir: "raw" }];
    expect(() => plan(files, "short")).toThrow(/lists 1 videos, 6 expected/);
  });

  it("says what Dataverse answered when the listing fails", async () => {
    const f = (async () => new Response("down", { status: 503 })) as unknown as typeof fetch;
    await expect(fetchListing(f)).rejects.toThrow(/503/);
  });
});

describe("downloadDemo", () => {
  it("fetches every file of the short demo into the target, with progress up to the total", async () => {
    const ds = dataset();
    const t = memoryTarget();
    const seen: Progress[] = [];
    const out = await downloadDemo("short", t, { fetch: fakeFetch(ds), onProgress: (p) => seen.push(p) });
    expect(out).toEqual({ fetched: 8, skipped: 0, bytes: 6 * 10 + 4 + 3 });
    expect(t.files.get("minime-demo-5min/raw/3.avi")).toEqual(ds.bytes.get(103));
    expect(t.files.get("minime-demo-5min/meta.json")).toEqual(ds.bytes.get(1));
    expect(t.files.has("minime-demo-5min/truth.npz")).toBe(false);
    const last = seen.at(-1)!;
    expect(last).toMatchObject({ done: 67, total: 67, index: 8, count: 8, file: "meta.json", skipped: 0 });
    expect(seen.every((p, i) => i === 0 || p.done >= seen[i - 1].done)).toBe(true);
  });

  it("keeps files already there with the listed size and fetches the rest (a broken or short one again)", async () => {
    const ds = dataset();
    const t = memoryTarget(
      new Map([
        ["minime-demo-5min/raw/0.avi", ds.bytes.get(100)!],
        ["minime-demo-5min/raw/1.avi", ds.bytes.get(101)!],
        ["minime-demo-5min/raw/2.avi", new Uint8Array(3)], // an older, broken copy
      ]),
    );
    const log: string[] = [];
    const need: number[] = [];
    const out = await downloadDemo("short", t, {
      fetch: fakeFetch(ds, { log }),
      beforeStart: (n) => {
        need.push(n);
      },
    });
    expect(out).toMatchObject({ fetched: 6, skipped: 2 });
    expect(need).toEqual([67 - 20]);
    expect(log).not.toContain(fileUrl(100));
    expect(log).not.toContain(fileUrl(101));
    expect(log).toContain(fileUrl(102));
    expect(t.files.get("minime-demo-5min/raw/2.avi")).toEqual(ds.bytes.get(102));
  });

  it("fetches nothing when every file is there", async () => {
    const ds = dataset();
    const t = memoryTarget();
    await downloadDemo("short", t, { fetch: fakeFetch(ds) });
    const log: string[] = [];
    expect(await downloadDemo("short", t, { fetch: fakeFetch(ds, { log }) })).toMatchObject({
      fetched: 0,
      skipped: 8,
    });
    expect(log).toEqual([LISTING_URL]);
  });

  it("stops before any file when beforeStart refuses (no room)", async () => {
    const ds = dataset();
    const t = memoryTarget();
    const log: string[] = [];
    await expect(
      downloadDemo("full", t, {
        fetch: fakeFetch(ds, { log }),
        beforeStart: () => {
          throw new Error("no room");
        },
      }),
    ).rejects.toThrow("no room");
    expect(log).toEqual([LISTING_URL]);
    expect(t.files.size).toBe(0);
  });

  it("leaves no file behind for a download that ends short, and keeps the ones before it", async () => {
    const ds = dataset();
    const t = memoryTarget();
    await expect(downloadDemo("short", t, { fetch: fakeFetch(ds, { short: 102 }) })).rejects.toThrow(
      /2\.avi ended after 9 of 10 bytes/,
    );
    expect([...t.files.keys()].sort()).toEqual(["minime-demo-5min/raw/0.avi", "minime-demo-5min/raw/1.avi"]);
    // the next try goes on from there
    const log: string[] = [];
    expect(await downloadDemo("short", t, { fetch: fakeFetch(ds, { log }) })).toMatchObject({ fetched: 6, skipped: 2 });
  });

  it("cancels in the middle of a file: that file is dropped, the finished ones stay", async () => {
    const ds = dataset();
    const t = memoryTarget();
    const ac = new AbortController();
    const fetchFn = fakeFetch(ds, {
      onChunk: (id) => {
        if (id === 103) ac.abort(new DOMException("cancelled", "AbortError"));
      },
    });
    await expect(downloadDemo("short", t, { fetch: fetchFn, signal: ac.signal })).rejects.toThrow("cancelled");
    expect([...t.files.keys()].sort()).toEqual([0, 1, 2].map((i) => `minime-demo-5min/raw/${i}.avi`));
  });

  it("cancels before the listing is in", async () => {
    const ac = new AbortController();
    ac.abort(new DOMException("cancelled", "AbortError"));
    await expect(
      downloadDemo("short", memoryTarget(), { fetch: fakeFetch(dataset()), signal: ac.signal }),
    ).rejects.toThrow("cancelled");
  });

  it("downloads from another source (a mirror) by its own list and URLs", async () => {
    const files: DataFile[] = [
      ...[0, 1, 2, 3, 4, 5].map((i) => ({ id: i, filename: `${i}.avi`, filesize: 3, dir: "raw" })),
      { id: 10, filename: "README.md", filesize: 1, dir: "" },
      { id: 11, filename: "meta.json", filesize: 2, dir: "" },
    ];
    const log: string[] = [];
    const fetchFn = (async (u: RequestInfo | URL) => {
      log.push(String(u));
      const f = files.find((x) => String(u).endsWith(`/${x.filename}`))!;
      return new Response(new Uint8Array(f.filesize));
    }) as typeof fetch;
    const t = memoryTarget();
    const source = { list: async () => files, url: (f: DataFile) => `https://mirror.test/demo/${f.filename}` };
    expect(await downloadDemo("short", t, { source, fetch: fetchFn })).toMatchObject({ fetched: 8 });
    expect(log[0]).toBe("https://mirror.test/demo/0.avi");
    expect(t.files.get("minime-demo-5min/meta.json")?.length).toBe(2);
  });

  it("names the demo folders after their length", () => {
    expect(DEMOS.short.folder).toBe("minime-demo-5min");
    expect(DEMOS.full.segments).toBe(36);
  });
});
