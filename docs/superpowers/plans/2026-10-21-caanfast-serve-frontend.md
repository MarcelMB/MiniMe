# `caanfast serve` front end Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The browser app of `caanfast serve`: home (runs, jobs), a run page with the max-projection/video top area and seven stage pages in the bento + glass look, and the run launcher, built into static files that ship inside the Python package.

**Architecture:** Vite + React + TypeScript in `web/`. A tiny API client (`api.ts`) adds the token and decodes the back end's binary arrays; pure helpers (drawing, frame cache, playback, polling, trace mode) carry the logic and are unit-tested with vitest; React components stay thin. Hash routing (`#/run/<id>/<page>`), no router library. `npm run build` writes to `src/caanfast/serve/static/`.

**Tech Stack:** React 18.3.1, TypeScript 5.6.3, Vite 5.4.10, uPlot 1.6.31, vitest 2.1.4 + jsdom, Playwright 1.48.2, Geist / Geist Mono via `@fontsource-variable` (bundled, works offline).

**Spec:** `docs/superpowers/specs/2026-10-21-caanfast-serve-design.md`. Plan 2 of 2; requires the back end of `2026-10-21-caanfast-serve-backend.md` (its routes and JSON shapes are used verbatim here).

## Global Constraints

- Routes, JSON shapes and the binary array format are exactly those of the back-end plan (`X-Shape`, `X-Dtype` ∈ `uint8`, `int32`, `float32`; `202` = computing with `{"status": "computing", "elapsed_s"}`; `500` with `{"status": "failed", "error"}`).
- Token: read `?token=` from the start URL, keep it in `sessionStorage` (`caanfast-token`), send it as `X-Token`; `EventSource` URLs carry `?token=`.
- Look: bento + glass, Geist / Geist Mono, light with a dark variant via `prefers-color-scheme`; max projection with contours and the video scrubber on top of every run page; layout works at 375 px width.
- Low-pass control: on/off plus a cutoff slider in Hz, default 5 Hz, from 0.5 Hz to Nyquist; disabled while the frame rate is unknown.
- Full-resolution traces are fetched only for windows of ≤ 20,000 frames; wider views use the 2,000-bin min/max overview.
- No runtime requests to other hosts (fonts are bundled).
- Deviation from the spec, on purpose: controls are native, accessible HTML elements (`button` with `aria-pressed`, `input type=range`, checkbox with `role="switch"`) styled in `controls.tsx`, instead of Base UI. Base UI renamed its package and changed APIs during its beta; with all controls in one file it can replace them later without touching the pages.
- `npm run typecheck` and `npm test` pass before every commit; one-sentence commit messages, no attribution trailer.

## Review Focus

- A run without a known frame rate: time labels show frames, the low-pass is disabled and the Traces page offers a field to enter the rate (`fmtClock` and `lowpassState` tests in Tasks 1 and 6).
- A run whose raw videos or tile store are missing: those sources are disabled with the reason as a tooltip, and the Frames page shows a greyed tile instead of failing (`sourceOptions` test in Task 3).
- A route still computing (older runs, contrast, overviews): the tile shows "Computing… N s" and fills in by itself when the back end is done; a failed computation shows its error (`pollResource` test in Task 2).
- Fast scrubbing through a 3 h run: frames that arrive late never replace the frame for the current position (`latest` test in Task 3).
- Hundreds of units and 216,000-frame traces: wide views use the overview, only narrow windows load full traces (`traceMode` test in Task 6).

## File Structure

| File | Responsibility |
|---|---|
| `web/package.json`, `web/tsconfig.json`, `web/vite.config.ts`, `web/index.html` | build setup; `build` writes `../src/caanfast/serve/static/` |
| `web/src/main.tsx` | entry: token, fonts, CSS, `<App/>` |
| `web/src/api.ts` | token, `request`, `getJSON`, `postJSON`, `getArray`, `decodeArray`, `Computing`, `ApiError` |
| `web/src/types.ts` | TypeScript shapes of the API's JSON |
| `web/src/format.ts` | number, duration, byte and clock formatting |
| `web/src/poll.ts` | `pollResource` (retry while `202`) + `useResource` hook |
| `web/src/router.ts` | hash routes: `parseHash`, `href`, `useRoute`, `PAGES` |
| `web/src/theme.css` | tokens, bento grid, glass cards, controls |
| `web/src/controls.tsx` | `Toggle`, `Slider`, `Segmented`, `ErrorNote`, `Computing` |
| `web/src/App.tsx` | header + route switch |
| `web/src/home/Home.tsx`, `web/src/home/FolderPicker.tsx` | runs under a folder, jobs list, folder browser |
| `web/src/draw.ts` | gray/heat images, polygons, points, `pickUnit`, `percentile` |
| `web/src/frames.ts` | `FrameCache` (LRU of frame promises + prefetch), `latest` |
| `web/src/playback.ts` | `advance` for play at 1×/4×/16× |
| `web/src/ImageView.tsx` | one canvas for frames, maps, contours, seeds, picking |
| `web/src/Chart.tsx` | uPlot wrapper with playhead and drag-to-zoom |
| `web/src/run/RunContext.tsx` | run state shared by the top area and pages |
| `web/src/run/RunPage.tsx`, `web/src/run/TopArea.tsx`, `web/src/run/registry.ts` | run page shell, top area, page registry |
| `web/src/run/pages/*.tsx` | Overview, Frames, Motion, Compression, Seeds, Units, Traces |
| `web/src/launch/NewRun.tsx`, `web/src/launch/ParamsForm.tsx`, `web/src/launch/JobPage.tsx` | launcher and job progress |
| `web/e2e/make_run.py`, `web/e2e/smoke.spec.ts`, `web/playwright.config.ts` | end-to-end smoke test |
| `pyproject.toml`, `.gitignore`, `README.md` (modify) | ship `static/` in the wheel, ignore build output, docs |

---

### Task 1: Scaffold, API client, types, formatting

**Files:**
- Create: `web/package.json`, `web/tsconfig.json`, `web/vite.config.ts`, `web/index.html`, `web/src/main.tsx`, `web/src/api.ts`, `web/src/types.ts`, `web/src/format.ts`, `web/src/App.tsx` (minimal, replaced in Task 2)
- Modify: `.gitignore`
- Test: `web/src/api.test.ts`, `web/src/format.test.ts`

**Interfaces:**
- Produces (`api.ts`): `type DType`, `type NDArray = { shape: number[]; dtype: DType; data: Uint8Array | Int32Array | Float32Array }`, `class Computing extends Error { elapsed: number }`, `class ApiError extends Error { status: number }`, `readToken(search, store): string`, `initToken(): void`, `tokenQuery(): string`, `decodeArray(buf, shape, dtype): NDArray`, `getJSON<T>(path): Promise<T>`, `postJSON<T>(path, body): Promise<T>`, `getArray(path): Promise<NDArray>`. Paths are relative to `/api`.
- Produces (`types.ts`): `Stage, Available, RunListing, Summary, Unit, UnitsResponse, Seeds, FsEntry, FsListing, Job, ParamMeta, RawInfo, Zooms`.
- Produces (`format.ts`): `fmtInt(n)`, `fmtDuration(seconds)`, `fmtBytes(bytes)`, `fmtClock(frame, fps | null)`, `stageLabel(name)`.

- [ ] **Step 1: Create the project files**

`web/package.json`:

```json
{
  "name": "caanfast-web",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "typecheck": "tsc --noEmit",
    "test": "vitest run",
    "e2e": "playwright test"
  },
  "dependencies": {
    "@fontsource-variable/geist": "^5.1.0",
    "@fontsource-variable/geist-mono": "^5.1.0",
    "react": "18.3.1",
    "react-dom": "18.3.1",
    "uplot": "1.6.31"
  },
  "devDependencies": {
    "@playwright/test": "1.48.2",
    "@types/node": "22.9.0",
    "@types/react": "18.3.12",
    "@types/react-dom": "18.3.1",
    "@vitejs/plugin-react": "4.3.3",
    "jsdom": "25.0.1",
    "typescript": "5.6.3",
    "vite": "5.4.10",
    "vitest": "2.1.4"
  }
}
```

`web/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "resolveJsonModule": true,
    "types": ["vite/client", "node"]
  },
  "include": ["src", "e2e", "vite.config.ts", "playwright.config.ts"]
}
```

`web/vite.config.ts`:

```ts
/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "../src/caanfast/serve/static", emptyOutDir: true },
  server: { proxy: { "/api": process.env.CAANFAST_API ?? "http://127.0.0.1:8765" } },
  test: { environment: "jsdom", include: ["src/**/*.test.ts"] },
});
```

`web/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CaAnFast</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Append to the repository `.gitignore`:

```
web/node_modules/
web/test-results/
web/playwright-report/
web/e2e/.data/
src/caanfast/serve/static/
```

Run: `cd web && npm install`
Expected: installs without errors and writes `web/package-lock.json`.

- [ ] **Step 2: Write the failing tests**

`web/src/api.test.ts`:

```ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, Computing, decodeArray, getArray, getJSON, readToken } from "./api";

function memStore() {
  const m = new Map<string, string>();
  return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v) };
}

describe("token", () => {
  it("takes the token from the URL and remembers it", () => {
    const s = memStore();
    expect(readToken("?token=abc&root=/x", s)).toBe("abc");
    expect(readToken("", s)).toBe("abc");
  });
  it("is empty without one", () => {
    expect(readToken("", memStore())).toBe("");
  });
});

describe("decodeArray", () => {
  it("decodes float32 with its shape", () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const d = decodeArray(a.buffer, "2,3", "float32");
    expect(d.shape).toEqual([2, 3]);
    expect(Array.from(d.data)).toEqual([1, 2, 3, 4, 5, 6]);
  });
  it("handles an empty array and rejects unknown dtypes", () => {
    expect(decodeArray(new ArrayBuffer(0), "0,2", "float32").shape).toEqual([0, 2]);
    expect(() => decodeArray(new ArrayBuffer(8), "1", "float64")).toThrow();
  });
});

describe("requests", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("turns 202 into Computing and errors into ApiError with the detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(new Response(JSON.stringify({ status: "computing", elapsed_s: 3 }), { status: 202 }))
        .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "unknown run x" }), { status: 404 }))
        .mockResolvedValueOnce(new Response(JSON.stringify({ status: "failed", error: "boom" }), { status: 500 })),
    );
    await expect(getJSON("/a")).rejects.toBeInstanceOf(Computing);
    await expect(getJSON("/b")).rejects.toMatchObject({ status: 404, message: "unknown run x" });
    const e = await getJSON("/c").catch((x) => x);
    expect(e).toBeInstanceOf(ApiError);
    expect(e.message).toBe("boom");
  });

  it("decodes array responses from the headers", async () => {
    const body = new Uint8Array([1, 2, 3, 4]);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(body, { status: 200, headers: { "X-Shape": "2,2", "X-Dtype": "uint8" } }),
      ),
    );
    const a = await getArray("/frame");
    expect(a.shape).toEqual([2, 2]);
    expect(Array.from(a.data)).toEqual([1, 2, 3, 4]);
    expect((fetch as unknown as { mock: { calls: unknown[][] } }).mock.calls[0][0]).toBe("/api/frame");
  });
});
```

`web/src/format.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { fmtBytes, fmtClock, fmtDuration, fmtInt, stageLabel } from "./format";

describe("format", () => {
  it("formats durations like the run report", () => {
    expect(fmtDuration(42.4)).toBe("42 s");
    expect(fmtDuration(1185)).toBe("19:45");
    expect(fmtDuration(3 * 3600 + 61)).toBe("3:01:01");
  });
  it("formats clocks with and without a frame rate", () => {
    expect(fmtClock(48200, 20)).toBe("0:40:10");
    expect(fmtClock(48200, null)).toBe("frame 48,200");
  });
  it("formats numbers, bytes and stage names", () => {
    expect(fmtInt(216000)).toBe("216,000");
    expect(fmtBytes(13.9e9)).toBe("13.9 GB");
    expect(fmtBytes(512e6)).toBe("512 MB");
    expect(stageLabel("pass1")).toBe("Pass 1 · preprocessing + motion");
    expect(stageLabel("cnmf")).toBe("CNMF");
    expect(stageLabel("pnr_refine")).toBe("pnr refine");
  });
});
```

- [ ] **Step 3: Run them to make sure they fail**

Run: `cd web && npx vitest run`
Expected: FAIL — `Failed to resolve import "./api"` and `"./format"`.

- [ ] **Step 4: Implement**

`web/src/api.ts`:

```ts
export type DType = "uint8" | "int32" | "float32";
export type NDArray = { shape: number[]; dtype: DType; data: Uint8Array | Int32Array | Float32Array };

/** The back end answered 202: the result is being computed. */
export class Computing extends Error {
  constructor(public elapsed: number) {
    super("computing");
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

const KEY = "caanfast-token";
let token = "";

export function readToken(search: string, store: Pick<Storage, "getItem" | "setItem">): string {
  const t = new URLSearchParams(search).get("token");
  if (t) {
    store.setItem(KEY, t);
    return t;
  }
  return store.getItem(KEY) ?? "";
}

export function initToken(): void {
  token = readToken(window.location.search, window.sessionStorage);
}

export function tokenQuery(): string {
  return `token=${encodeURIComponent(token)}`;
}

const CTORS = { uint8: Uint8Array, int32: Int32Array, float32: Float32Array } as const;

export function decodeArray(buf: ArrayBuffer, shape: string, dtype: string): NDArray {
  const ctor = CTORS[dtype as DType];
  if (!ctor) throw new Error(`unknown dtype ${dtype}`);
  const dims = shape ? shape.split(",").map(Number) : [];
  return { shape: dims, dtype: dtype as DType, data: new ctor(buf) };
}

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = { "X-Token": token, ...(init.headers as Record<string, string> | undefined) };
  const r = await fetch(`/api${path}`, { ...init, headers });
  if (r.status === 202) {
    const b = await r.json();
    throw new Computing(Number(b.elapsed_s ?? 0));
  }
  if (!r.ok) {
    const text = await r.text();
    let msg = text || r.statusText;
    try {
      const j = JSON.parse(text);
      msg = j.detail ?? j.error ?? msg;
    } catch {
      /* plain text body */
    }
    throw new ApiError(r.status, String(msg));
  }
  return r;
}

export async function getJSON<T>(path: string): Promise<T> {
  return (await request(path)).json() as Promise<T>;
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json() as Promise<T>;
}

export async function getArray(path: string): Promise<NDArray> {
  const r = await request(path);
  return decodeArray(await r.arrayBuffer(), r.headers.get("X-Shape") ?? "", r.headers.get("X-Dtype") ?? "");
}
```

`web/src/types.ts`:

```ts
export type Stage = { stage: string; seconds: number };
export type Available = { raw: boolean; pre: boolean; seeds_saved: boolean; init_saved: boolean };

export type RunListing = {
  id: string;
  name: string;
  path: string;
  units: number | null;
  T: number;
  fps: number | null;
  wall_total: number | null;
  mtime: number;
  error?: string;
};

export type Summary = RunListing & {
  H: number;
  W: number;
  tile: number;
  ranks: number[];
  raw_dir: string | null;
  params: Record<string, unknown>;
  params_changed: Record<string, { value: unknown; default: unknown }>;
  stages: Stage[];
  counts: Record<string, number>;
  peak_rss: number | null;
  peak_rss_children: number | null;
  bytes: Record<string, number>;
  workers: number | null;
  motion: { quality_median?: number; max_abs_shift_px?: number };
  available: Available;
};

export type Unit = {
  id: number;
  centroid: [number, number];
  size_px: number;
  peak: number | null;
  polygons: number[][][];
};
export type UnitsResponse = { stage: "final" | "init"; units: Unit[] };
export type Seeds = { init: number[][]; pnr: number[][]; merged: number[][] };

export type FsEntry = { name: string; path: string; has_avi: boolean; is_run: boolean };
export type FsListing = { path: string; parent: string; entries: FsEntry[] };

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled" | "interrupted";
export type Job = {
  id: string;
  raw_dir: string;
  name: string;
  params: Record<string, unknown>;
  keep_store: boolean;
  status: JobStatus;
  out_dir: string;
  created: number;
  started: number | null;
  ended: number | null;
  error: string;
};

export type ParamMeta = {
  name: string;
  group: string;
  type: string;
  default: unknown;
  help: string;
  common: boolean;
};

export type RawInfo = {
  path: string;
  segments: number;
  frames: number;
  height: number;
  width: number;
  fps: number | null;
  bytes: number;
};

export type Zooms = { units: number[]; windows: { start: number; stop: number }[] };
```

`web/src/format.ts`:

```ts
export function fmtInt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

export function fmtDuration(s: number): string {
  if (s < 60) return `${s.toFixed(0)} s`;
  const total = Math.round(s);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const r = total % 60;
  const mm = h ? String(m).padStart(2, "0") : String(m);
  return `${h ? `${h}:` : ""}${mm}:${String(r).padStart(2, "0")}`;
}

export function fmtBytes(b: number): string {
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(0)} MB`;
  return `${(b / 1e3).toFixed(0)} kB`;
}

export function fmtClock(frame: number, fps: number | null): string {
  if (!fps) return `frame ${fmtInt(frame)}`;
  const s = Math.floor(frame / fps);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}:${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

const STAGES: Record<string, string> = {
  pass1: "Pass 1 · preprocessing + motion",
  pass2: "Pass 2 · compression",
  cnmf: "CNMF",
  initA: "init footprints",
  initC: "init traces",
};

export function stageLabel(name: string): string {
  return STAGES[name] ?? name.replace(/_/g, " ");
}
```

`web/src/App.tsx` (minimal; Task 2 replaces it):

```tsx
export function App() {
  return <main>CaAnFast</main>;
}
```

`web/src/main.tsx`:

```tsx
import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { initToken } from "./api";
import { App } from "./App";

initToken();
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

- [ ] **Step 5: Run the checks**

Run: `cd web && npm run typecheck && npm test && npm run build`
Expected: no type errors; all tests pass; `src/caanfast/serve/static/index.html` exists.

- [ ] **Step 6: Commit**

```bash
git add .gitignore web/package.json web/package-lock.json web/tsconfig.json web/vite.config.ts web/index.html web/src
git commit -m "Scaffold the web front end with its API client, types and formatting"
```

---

### Task 2: Theme, controls, routing, polling, home page

**Files:**
- Create: `web/src/theme.css`, `web/src/controls.tsx`, `web/src/router.ts`, `web/src/poll.ts`, `web/src/home/Home.tsx`, `web/src/home/FolderPicker.tsx`
- Modify: `web/src/App.tsx`, `web/src/main.tsx` (import `./theme.css`)
- Test: `web/src/router.test.ts`, `web/src/poll.test.ts`

**Interfaces:**
- Consumes: Task 1.
- Produces (`router.ts`): `PAGES` (`["overview","frames","motion","compression","seeds","units","traces"]`), `type Page`, `type Route = {name:"home"} | {name:"new"} | {name:"job"; id} | {name:"run"; id; page}`, `parseHash(h): Route`, `href(r): string`, `useRoute(): Route`, `go(r): void`.
- Produces (`poll.ts`): `type Resource<T> = { data?: T; error?: string; computing: boolean; elapsed: number }`, `pollResource<T>(load, onState, delayMs = 1500): () => void` (returns cancel), `useResource<T>(key: string | null, load): Resource<T>`.
- Produces (`controls.tsx`): `Toggle({label, checked, onChange, disabled?, title?})`, `Slider({label, min, max, step, value, onChange, disabled?, format?})`, `Segmented<T extends string>({label, options: {value: T; label: string; disabled?: boolean; title?: string}[], value, onChange, glass?})`, `ErrorNote({message})` (`role="alert"`), `ComputingNote({elapsed, what})`, `Loading<T>({res, what, children: (data: T) => ReactNode})`.
- Produces (`FolderPicker.tsx`): `FolderPicker({start, onPick, onClose, pickLabel, accept?: (e: FsEntry) => boolean})`.

- [ ] **Step 1: Write the failing tests**

`web/src/router.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { href, parseHash } from "./router";

describe("router", () => {
  it("parses every route and round-trips", () => {
    for (const r of [
      { name: "home" },
      { name: "new" },
      { name: "job", id: "abc" },
      { name: "run", id: "0123456789ab", page: "traces" },
    ] as const) {
      expect(parseHash(href(r))).toEqual(r);
    }
  });
  it("falls back to home and to the overview page", () => {
    expect(parseHash("")).toEqual({ name: "home" });
    expect(parseHash("#/nowhere")).toEqual({ name: "home" });
    expect(parseHash("#/run/abc")).toEqual({ name: "run", id: "abc", page: "overview" });
    expect(parseHash("#/run/abc/bogus")).toEqual({ name: "run", id: "abc", page: "overview" });
  });
});
```

`web/src/poll.test.ts`:

```ts
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, Computing } from "./api";
import { pollResource, type Resource } from "./poll";

describe("pollResource", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("retries while computing, then delivers the data", async () => {
    const load = vi
      .fn()
      .mockRejectedValueOnce(new Computing(1))
      .mockRejectedValueOnce(new Computing(2.5))
      .mockResolvedValueOnce(42);
    const states: Resource<number>[] = [];
    pollResource(load, (s) => states.push(s), 1000);
    await vi.runAllTimersAsync();
    expect(load).toHaveBeenCalledTimes(3);
    expect(states.map((s) => s.computing)).toEqual([true, true, false]);
    expect(states[1].elapsed).toBe(2.5);
    expect(states.at(-1)!.data).toBe(42);
  });

  it("stops on an error and after cancel", async () => {
    const states: Resource<number>[] = [];
    pollResource(() => Promise.reject(new ApiError(500, "boom")), (s) => states.push(s), 1000);
    await vi.runAllTimersAsync();
    expect(states.at(-1)!.error).toBe("boom");

    const load = vi.fn().mockRejectedValue(new Computing(0));
    const cancel = pollResource(load, () => {}, 1000);
    await vi.advanceTimersByTimeAsync(10);
    cancel();
    await vi.advanceTimersByTimeAsync(5000);
    expect(load).toHaveBeenCalledTimes(1);
  });
});
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `cd web && npx vitest run src/router.test.ts src/poll.test.ts`
Expected: FAIL — cannot resolve `./router` and `./poll`.

- [ ] **Step 3: Implement routing and polling**

`web/src/router.ts`:

```ts
import { useEffect, useState } from "react";

export const PAGES = ["overview", "frames", "motion", "compression", "seeds", "units", "traces"] as const;
export type Page = (typeof PAGES)[number];

export type Route =
  | { name: "home" }
  | { name: "new" }
  | { name: "job"; id: string }
  | { name: "run"; id: string; page: Page };

export function parseHash(h: string): Route {
  const parts = h.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (parts[0] === "new") return { name: "new" };
  if (parts[0] === "job" && parts[1]) return { name: "job", id: parts[1] };
  if (parts[0] === "run" && parts[1]) {
    const page = (PAGES as readonly string[]).includes(parts[2] ?? "") ? (parts[2] as Page) : "overview";
    return { name: "run", id: parts[1], page };
  }
  return { name: "home" };
}

export function href(r: Route): string {
  switch (r.name) {
    case "home":
      return "#/";
    case "new":
      return "#/new";
    case "job":
      return `#/job/${r.id}`;
    case "run":
      return `#/run/${r.id}/${r.page}`;
  }
}

export function go(r: Route): void {
  window.location.hash = href(r);
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}
```

`web/src/poll.ts`:

```ts
import { useEffect, useState } from "react";
import { Computing } from "./api";

export type Resource<T> = { data?: T; error?: string; computing: boolean; elapsed: number };

/** Call ``load`` until it stops throwing ``Computing``; report every state. Returns cancel. */
export function pollResource<T>(
  load: () => Promise<T>,
  onState: (s: Resource<T>) => void,
  delayMs = 1500,
): () => void {
  let alive = true;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const attempt = () => {
    load().then(
      (data) => alive && onState({ data, computing: false, elapsed: 0 }),
      (e: unknown) => {
        if (!alive) return;
        if (e instanceof Computing) {
          onState({ computing: true, elapsed: e.elapsed });
          timer = setTimeout(attempt, delayMs);
        } else {
          onState({ error: e instanceof Error ? e.message : String(e), computing: false, elapsed: 0 });
        }
      },
    );
  };
  attempt();
  return () => {
    alive = false;
    if (timer !== undefined) clearTimeout(timer);
  };
}

/** ``load`` once per ``key`` (null = nothing to load), retrying while the back end computes. */
export function useResource<T>(key: string | null, load: () => Promise<T>): Resource<T> {
  const [state, setState] = useState<Resource<T>>({ computing: false, elapsed: 0 });
  useEffect(() => {
    if (key === null) return;
    setState({ computing: false, elapsed: 0 });
    return pollResource(load, setState);
    // ``key`` identifies the request; ``load`` is recreated each render
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}
```

- [ ] **Step 4: Implement the theme and controls**

`web/src/theme.css`:

```css
:root {
  --ground: #eef1f6;
  --card: rgba(255, 255, 255, 0.55);
  --card-solid: #ffffff;
  --card-border: rgba(255, 255, 255, 0.75);
  --text: #111318;
  --muted: #4b5263;
  --accent: #2563eb;
  --hot: #ea580c;
  --ok: #15803d;
  --bad: #b91c1c;
  --track: rgba(17, 19, 24, 0.08);
  --shadow: 0 10px 30px rgba(30, 41, 59, 0.08);
  --map: #06080d;
  --blob1: #bcdcff;
  --blob2: #ffd8c4;
  --blob3: #c6f1e2;
  --blob4: #e3d9ff;
  color-scheme: light;
  font-family: "Geist Variable", system-ui, sans-serif;
  font-size: 14px;
  color: var(--text);
}

@media (prefers-color-scheme: dark) {
  :root {
    --ground: #0b0d12;
    --card: rgba(28, 32, 42, 0.62);
    --card-solid: #161a22;
    --card-border: rgba(255, 255, 255, 0.08);
    --text: #e8eaef;
    --muted: #a4abb9;
    --accent: #7aa2ff;
    --hot: #fb923c;
    --ok: #4ade80;
    --bad: #f87171;
    --track: rgba(255, 255, 255, 0.09);
    --shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    --blob1: #13284a;
    --blob2: #3a2216;
    --blob3: #0f3328;
    --blob4: #241b45;
    color-scheme: dark;
  }
}

* { box-sizing: border-box; }

body {
  margin: 0;
  min-height: 100vh;
  background:
    radial-gradient(38% 46% at 12% 18%, var(--blob1) 0, transparent 70%),
    radial-gradient(34% 42% at 88% 12%, var(--blob2) 0, transparent 70%),
    radial-gradient(44% 50% at 72% 88%, var(--blob3) 0, transparent 70%),
    radial-gradient(30% 40% at 30% 80%, var(--blob4) 0, transparent 70%),
    var(--ground);
  background-attachment: fixed;
}

a { color: var(--accent); }
.mono { font-family: "Geist Mono Variable", ui-monospace, monospace; }
.muted { color: var(--muted); }

.topbar {
  max-width: 1280px; margin: 0 auto; padding: 16px 16px 0;
  display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
}
.brand { font-weight: 600; letter-spacing: -0.01em; text-decoration: none; color: var(--text); font-size: 16px; }
.topbar nav { display: flex; gap: 8px; }

.page { max-width: 1280px; margin: 0 auto; padding: 20px 16px 48px; display: flex; flex-direction: column; gap: 20px; }
.page h1 { margin: 0; font-size: 28px; font-weight: 600; letter-spacing: -0.03em; }
.row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.spacer { flex: 1 1 0; }

.bento { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }
.span2 { grid-column: span 2; }
.span3 { grid-column: span 3; }
.span4 { grid-column: 1 / -1; }
.row2 { grid-row: span 2; }

.card {
  background: var(--card);
  backdrop-filter: blur(24px) saturate(180%);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
  border: 1px solid var(--card-border);
  border-radius: 24px;
  padding: 20px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.5), var(--shadow);
  display: flex; flex-direction: column; gap: 12px; min-width: 0;
}
.card.dim { opacity: 0.6; }
.label { font-size: 13px; color: var(--muted); font-weight: 500; }
.big { font-size: 40px; font-weight: 600; letter-spacing: -0.04em; line-height: 1; }

.btn {
  font: inherit; font-size: 13px; font-weight: 500; color: var(--text);
  background: var(--card); border: 1px solid var(--card-border); border-radius: 999px;
  min-height: 38px; padding: 0 16px; cursor: pointer; text-decoration: none;
  display: inline-flex; align-items: center; gap: 8px;
  backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
}
.btn:hover { background: var(--card-solid); }
.btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }

.seg { display: inline-flex; flex-wrap: wrap; gap: 2px; padding: 4px; border-radius: 999px;
  background: var(--card); border: 1px solid var(--card-border); }
.seg button { font: inherit; font-size: 13px; font-weight: 500; color: var(--muted); background: transparent;
  border: 0; border-radius: 999px; min-height: 32px; padding: 0 14px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--text); color: var(--ground); }
.seg button:disabled { opacity: 0.45; cursor: not-allowed; }

.glass {
  backdrop-filter: blur(20px) saturate(170%); -webkit-backdrop-filter: blur(20px) saturate(170%);
  background: rgba(255, 255, 255, 0.14); border: 1px solid rgba(255, 255, 255, 0.28);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.35), 0 8px 24px rgba(0, 0, 0, 0.3);
  border-radius: 999px; color: #fff;
}
.glass.seg { background: rgba(255, 255, 255, 0.14); border-color: rgba(255, 255, 255, 0.28); }
.glass.seg button { color: rgba(255, 255, 255, 0.85); }
.glass.seg button[aria-pressed="true"] { background: rgba(255, 255, 255, 0.92); color: #0b1220; }

.map { position: relative; background: var(--map); border-radius: 24px; overflow: hidden; min-height: 320px; }
.map .imageview { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
.overlay { position: absolute; inset: 0; padding: 14px; display: flex; flex-direction: column;
  justify-content: space-between; gap: 12px; pointer-events: none; }
.overlay > * { pointer-events: auto; }
.player { display: flex; align-items: center; gap: 12px; padding: 6px 16px 6px 6px; flex-wrap: wrap; }
.player input[type="range"] { flex: 1 1 160px; accent-color: #fff; }

.imageview { display: block; width: 100%; height: auto; image-rendering: pixelated; border-radius: 14px; background: var(--map); }

.bar { height: 10px; background: var(--track); border-radius: 999px; overflow: hidden; flex: 1; }
.bar > span { display: block; height: 100%; background: var(--accent); border-radius: 999px; }
.barrow { display: flex; align-items: center; gap: 12px; }
.barrow > :first-child { width: 190px; font-size: 13px; }
.barrow > :last-child { width: 72px; text-align: right; font-size: 13px; }

.chip { font: inherit; font-size: 12px; color: var(--text); background: var(--card); border: 1px solid var(--card-border);
  border-radius: 999px; min-height: 32px; padding: 0 12px; cursor: pointer; }
.chip[aria-pressed="true"] { background: var(--text); color: var(--ground); border-color: var(--text); }

.field { display: flex; flex-direction: column; gap: 6px; font-size: 13px; }
.field input[type="text"], .field input[type="number"] {
  font: inherit; color: var(--text); background: var(--card-solid); border: 1px solid var(--track);
  border-radius: 10px; min-height: 36px; padding: 0 10px; }
.switch { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; cursor: pointer; }
.switch input { accent-color: var(--accent); width: 18px; height: 18px; }
.slider { display: flex; align-items: center; gap: 10px; font-size: 13px; }
.slider input { accent-color: var(--accent); flex: 1; }

table.data { width: 100%; border-collapse: collapse; font-size: 13px; }
table.data th { text-align: left; color: var(--muted); font-weight: 500; padding: 6px 8px; }
table.data td { padding: 6px 8px; border-top: 1px solid var(--track); }
table.data tr[aria-selected="true"] td { background: var(--track); }
table.data th button { font: inherit; color: inherit; background: none; border: 0; cursor: pointer; padding: 0; }
.scroll { overflow: auto; max-height: 420px; }

.error { color: var(--bad); margin: 0; }
.computing { color: var(--muted); display: flex; align-items: center; gap: 10px; }
.computing .bar > span { width: 35%; animation: slide 1.4s ease-in-out infinite; }
@keyframes slide { 0% { margin-left: -35%; } 100% { margin-left: 100%; } }

.chart { width: 100%; }
.log { font-family: "Geist Mono Variable", ui-monospace, monospace; font-size: 12px; white-space: pre-wrap;
  max-height: 260px; overflow: auto; margin: 0; }
.picker { display: flex; flex-direction: column; gap: 6px; max-height: 420px; overflow: auto; }
.picker button { text-align: left; }

@media (max-width: 900px) {
  .bento { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .span3 { grid-column: 1 / -1; }
}
@media (max-width: 560px) {
  .bento { grid-template-columns: minmax(0, 1fr); }
  .span2, .span3, .row2 { grid-column: auto; grid-row: auto; }
  .barrow > :first-child { width: 110px; }
}
```

`web/src/controls.tsx`:

```tsx
import type { ReactNode } from "react";
import type { Resource } from "./poll";

export function Toggle(p: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <label className="switch" title={p.title}>
      <input
        type="checkbox"
        role="switch"
        checked={p.checked}
        disabled={p.disabled}
        onChange={(e) => p.onChange(e.target.checked)}
      />
      {p.label}
    </label>
  );
}

export function Slider(p: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  format?: (v: number) => string;
}) {
  return (
    <label className="slider">
      <span>{p.label}</span>
      <input
        type="range"
        min={p.min}
        max={p.max}
        step={p.step}
        value={p.value}
        disabled={p.disabled}
        onChange={(e) => p.onChange(Number(e.target.value))}
      />
      <span className="mono">{p.format ? p.format(p.value) : p.value}</span>
    </label>
  );
}

export type SegOption<T extends string> = { value: T; label: string; disabled?: boolean; title?: string };

export function Segmented<T extends string>(p: {
  label: string;
  options: SegOption<T>[];
  value: T;
  onChange: (v: T) => void;
  glass?: boolean;
}) {
  return (
    <div className={p.glass ? "seg glass" : "seg"} role="group" aria-label={p.label}>
      {p.options.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={o.value === p.value}
          disabled={o.disabled}
          title={o.title}
          onClick={() => p.onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <p role="alert" className="error">
      {message}
    </p>
  );
}

export function ComputingNote({ elapsed, what }: { elapsed: number; what: string }) {
  return (
    <div className="computing" aria-live="polite">
      <span>
        Computing {what}… {elapsed.toFixed(0)} s
      </span>
      <div className="bar">
        <span />
      </div>
    </div>
  );
}

/** Render ``children(data)`` once a resource is loaded; error / computing notes before. */
export function Loading<T>(p: { res: Resource<T>; what: string; children: (data: T) => ReactNode }) {
  if (p.res.error) return <ErrorNote message={p.res.error} />;
  if (p.res.computing) return <ComputingNote elapsed={p.res.elapsed} what={p.what} />;
  if (p.res.data === undefined) return <p className="muted">Loading {p.what}…</p>;
  return <>{p.children(p.res.data)}</>;
}
```

- [ ] **Step 5: Implement the home page, folder picker and app shell**

`web/src/home/FolderPicker.tsx`:

```tsx
import { useState } from "react";
import { getJSON } from "../api";
import { Loading } from "../controls";
import { useResource } from "../poll";
import type { FsEntry, FsListing } from "../types";

export function FolderPicker(p: {
  start: string;
  pickLabel: string;
  onPick: (path: string) => void;
  onClose: () => void;
  accept?: (e: FsEntry) => boolean;
}) {
  const [path, setPath] = useState(p.start);
  const res = useResource(`fs:${path}`, () =>
    getJSON<FsListing>(`/fs${path ? `?path=${encodeURIComponent(path)}` : ""}`),
  );
  return (
    <section className="card span4" aria-label="Choose a folder">
      <div className="row">
        <span className="label">Folder</span>
        <span className="mono">{res.data?.path ?? path}</span>
        <span className="spacer" />
        <button className="btn" type="button" onClick={p.onClose}>
          Close
        </button>
        {res.data && (
          <button className="btn primary" type="button" onClick={() => p.onPick(res.data!.path)}>
            {p.pickLabel}
          </button>
        )}
      </div>
      <Loading res={res} what="folder">
        {(fs) => (
          <div className="picker">
            <button className="btn" type="button" onClick={() => setPath(fs.parent)}>
              ↑ {fs.parent}
            </button>
            {fs.entries.map((e) => (
              <div className="row" key={e.path}>
                <button className="btn" type="button" onClick={() => setPath(e.path)}>
                  {e.name}
                </button>
                {e.has_avi && <span className="label">videos</span>}
                {e.is_run && <span className="label">run</span>}
                {p.accept?.(e) && (
                  <button className="chip" type="button" onClick={() => p.onPick(e.path)}>
                    {p.pickLabel}
                  </button>
                )}
              </div>
            ))}
            {fs.entries.length === 0 && <p className="muted">No subfolders.</p>}
          </div>
        )}
      </Loading>
    </section>
  );
}
```

`web/src/home/Home.tsx`:

```tsx
import { useEffect, useState } from "react";
import { getJSON } from "../api";
import { Loading } from "../controls";
import { fmtDuration, fmtInt } from "../format";
import { pollResource, useResource, type Resource } from "../poll";
import { href } from "../router";
import type { Job, RunListing } from "../types";
import { FolderPicker } from "./FolderPicker";

const ROOT_KEY = "caanfast-root";

function initialRoot(): string {
  const fromUrl = new URLSearchParams(window.location.search).get("root");
  if (fromUrl) return fromUrl;
  try {
    return window.localStorage.getItem(ROOT_KEY) ?? "";
  } catch {
    return "";
  }
}

export function useJobs(): Resource<{ jobs: Job[] }> {
  const [res, setRes] = useState<Resource<{ jobs: Job[] }>>({ computing: false, elapsed: 0 });
  useEffect(() => {
    let cancel = () => {};
    const tick = () => {
      cancel = pollResource(() => getJSON<{ jobs: Job[] }>("/jobs"), setRes);
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => {
      clearInterval(id);
      cancel();
    };
  }, []);
  return res;
}

export function Home() {
  const [root, setRoot] = useState(initialRoot);
  const [picking, setPicking] = useState(false);
  const runs = useResource(`runs:${root}`, () =>
    getJSON<{ root: string; runs: RunListing[] }>(`/runs${root ? `?root=${encodeURIComponent(root)}` : ""}`),
  );
  const jobs = useJobs();
  const choose = (p: string) => {
    setRoot(p);
    setPicking(false);
    try {
      window.localStorage.setItem(ROOT_KEY, p);
    } catch {
      /* private mode */
    }
  };
  const active = (jobs.data?.jobs ?? []).filter((j) => j.status === "queued" || j.status === "running");
  return (
    <div className="page">
      <div className="row">
        <div>
          <span className="label">Runs in</span>
          <h1 className="mono" style={{ fontSize: 18 }}>
            {runs.data?.root ?? (root || "home folder")}
          </h1>
        </div>
        <span className="spacer" />
        <button className="btn" type="button" onClick={() => setPicking(true)}>
          Change folder
        </button>
        <a className="btn primary" href={href({ name: "new" })}>
          New run
        </a>
      </div>
      {picking && (
        <div className="bento">
          <FolderPicker
            start={runs.data?.root ?? root}
            pickLabel="Show runs here"
            onPick={choose}
            onClose={() => setPicking(false)}
          />
        </div>
      )}
      {active.length > 0 && (
        <section className="card" aria-label="Running jobs">
          <span className="label">Jobs</span>
          {active.map((j) => (
            <a key={j.id} className="row" href={href({ name: "job", id: j.id })}>
              <span>{j.name}</span>
              <span className="muted">{j.status}</span>
            </a>
          ))}
        </section>
      )}
      <Loading res={runs} what="runs">
        {(d) =>
          d.runs.length === 0 ? (
            <p className="muted">No runs in this folder. Pick another folder or start a new run.</p>
          ) : (
            <div className="bento">
              {d.runs.map((r) =>
                r.error ? (
                  <section className="card dim" key={r.path}>
                    <span className="label">{r.name}</span>
                    <span className="error">{r.error}</span>
                  </section>
                ) : (
                  <a
                    className="card"
                    key={r.id}
                    aria-label={`Open run ${r.name}`}
                    href={href({ name: "run", id: r.id, page: "overview" })}
                    style={{ textDecoration: "none", color: "inherit" }}
                  >
                    <span className="label">{r.name}</span>
                    <span className="big">{r.units ?? "–"}</span>
                    <span className="muted">
                      units · {fmtInt(r.T)} frames{r.fps ? ` · ${fmtDuration(r.T / r.fps)}` : ""}
                    </span>
                    <span className="muted mono" style={{ fontSize: 12, overflowWrap: "anywhere" }}>
                      {r.path}
                    </span>
                  </a>
                ),
              )}
            </div>
          )
        }
      </Loading>
    </div>
  );
}
```

Replace `web/src/App.tsx`:

```tsx
import { Home } from "./home/Home";
import { href, useRoute } from "./router";

export function App() {
  const route = useRoute();
  return (
    <>
      <header className="topbar">
        <a className="brand" href={href({ name: "home" })}>
          CaAnFast
        </a>
        <nav aria-label="Main">
          <a className="btn" href={href({ name: "home" })}>
            Runs
          </a>
          <a className="btn" href={href({ name: "new" })}>
            New run
          </a>
        </nav>
      </header>
      {route.name === "home" && <Home />}
    </>
  );
}
```

In `web/src/main.tsx`, add `import "./theme.css";` after the font imports.

- [ ] **Step 6: Run the checks and look at it**

Run: `cd web && npm run typecheck && npm test`
Expected: no type errors; all tests pass.

Manual check: in one terminal `caanfast serve <folder with a run> --no-browser --port 8765 --token dev`; in another `cd web && npm run dev`; open `http://localhost:5173/?token=dev&root=<folder>`. Expected: the run appears as a card; "Change folder" lists subfolders.

- [ ] **Step 7: Commit**

```bash
git add web/src
git commit -m "Add the theme, controls, hash routing, polling and the home page"
```

---

### Task 3: Images, frame cache, playback and the run page's top area

**Files:**
- Create: `web/src/draw.ts`, `web/src/frames.ts`, `web/src/playback.ts`, `web/src/ImageView.tsx`, `web/src/run/RunContext.tsx`, `web/src/run/registry.ts`, `web/src/run/RunPage.tsx`, `web/src/run/TopArea.tsx`
- Modify: `web/src/App.tsx` (route `run`)
- Test: `web/src/draw.test.ts`, `web/src/frames.test.ts`, `web/src/playback.test.ts`

**Interfaces:**
- Consumes: Tasks 1–2; routes `/runs/{id}/summary`, `/runs/{id}/units`, `/runs/{id}/image/max_proj`, `/runs/{id}/frame/{t}?source=`.
- Produces (`draw.ts`): `type Img = { data: ArrayLike<number>; W: number; H: number; lo: number; hi: number; cmap?: "gray" | "heat" }`, `percentile(a, q): number`, `imgFrom(arr: NDArray, cmap?, qLo = 0.5, qHi = 99.8): Img`, `fillImageData(img, out: ImageData): ImageData`, `heat(v: number): [number, number, number]`, `pointInPolygon(x, y, poly): boolean`, `pickUnit(units, x, y, maxDist = 8): number | null`, `drawPolygons(ctx, units, scale, sel)`, `drawPoints(ctx, pts, scale, color)`.
- Produces (`frames.ts`): `class FrameCache { constructor(fetcher: (source: string, t: number) => Promise<NDArray>, capacity = 160); get(source, t); has(source, t); prefetch(source, t, T, ahead = 12, behind = 3) }`, `latest(): <T>(p: Promise<T>) => Promise<T | typeof STALE>`, `STALE`.
- Produces (`playback.ts`): `advance(t, T, speed, fps, dtMs, carry): { t: number; carry: number; ended: boolean }`.
- Produces (`ImageView.tsx`): `ImageView({ image: Img | null, units?, contours?, sel?, onPick?, points?, pointColor?, label?, scale? })`.
- Produces (`RunContext.tsx`): `type Source = "max" | "raw" | "pre" | "recon"`, `type RunState = { summary: Summary; units: Unit[]; maxProj: Img; t: number; setT(t: number): void; sel: number | null; setSel(id: number | null): void; cache: FrameCache }`, `RunCtx`, `useRun(): RunState`, `sourceOptions(available: Available): SegOption<Source>[]`.
- Produces (`registry.ts`): `PAGE_VIEWS: Partial<Record<Page, () => JSX.Element>>`, `PAGE_LABELS: Record<Page, string>`; later tasks add their pages here.

- [ ] **Step 1: Write the failing tests**

`web/src/draw.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { fillImageData, percentile, pickUnit, pointInPolygon } from "./draw";
import { sourceOptions } from "./run/RunContext";
import type { Unit } from "./types";

const square = [
  [10, 10],
  [20, 10],
  [20, 20],
  [10, 20],
];
const unit = (id: number, poly: number[][], c: [number, number]): Unit => ({
  id,
  centroid: c,
  size_px: 1,
  peak: null,
  polygons: [poly],
});

describe("draw helpers", () => {
  it("maps lo..hi to 0..255 and clamps", () => {
    const out = fillImageData(
      { data: [0, 5, 10, 20], W: 4, H: 1, lo: 0, hi: 10 },
      { data: new Uint8ClampedArray(16), width: 4, height: 1 } as ImageData,
    );
    expect(Array.from(out.data.filter((_, i) => i % 4 === 0))).toEqual([0, 128, 255, 255]);
    expect(out.data[3]).toBe(255);
  });
  it("computes percentiles", () => {
    expect(percentile([5, 1, 3, 2, 4], 50)).toBe(3);
    expect(percentile([1, 2, 3, 4, 5], 100)).toBe(5);
  });
  it("finds points in polygons and picks units", () => {
    expect(pointInPolygon(15, 15, square)).toBe(true);
    expect(pointInPolygon(25, 15, square)).toBe(false);
    const units = [unit(0, square, [15, 15]), unit(1, [[40, 40]], [40, 40])];
    expect(pickUnit(units, 12, 18)).toBe(0);
    expect(pickUnit(units, 43, 41)).toBe(1); // a one-point outline: nearest centroid
    expect(pickUnit(units, 100, 100)).toBeNull();
  });
  it("disables sources the run does not have, with the reason", () => {
    const opts = sourceOptions({ raw: false, pre: true, seeds_saved: true, init_saved: true });
    const raw = opts.find((o) => o.value === "raw")!;
    expect(raw.disabled).toBe(true);
    expect(raw.title).toMatch(/raw videos/);
    expect(opts.find((o) => o.value === "recon")!.disabled).toBeFalsy();
  });
});
```

`web/src/frames.test.ts`:

```ts
import { describe, expect, it, vi } from "vitest";
import type { NDArray } from "./api";
import { FrameCache, STALE, latest } from "./frames";

const arr = (t: number): NDArray => ({ shape: [1], dtype: "uint8", data: new Uint8Array([t]) });

describe("FrameCache", () => {
  it("fetches each frame once and evicts the oldest", async () => {
    const f = vi.fn((_: string, t: number) => Promise.resolve(arr(t)));
    const c = new FrameCache(f, 2);
    await c.get("recon", 1);
    await c.get("recon", 1);
    expect(f).toHaveBeenCalledTimes(1);
    await c.get("recon", 2);
    await c.get("recon", 1); // touches 1: now 2 is the oldest
    await c.get("recon", 3);
    expect(c.has("recon", 1)).toBe(true);
    expect(c.has("recon", 2)).toBe(false);
  });
  it("forgets failed fetches and prefetches inside the recording", async () => {
    const f = vi.fn().mockRejectedValueOnce(new Error("x")).mockResolvedValue(arr(0));
    const c = new FrameCache(f, 50);
    await expect(c.get("raw", 0)).rejects.toThrow();
    expect(c.has("raw", 0)).toBe(false);
    c.prefetch("raw", 1, 5, 12, 3);
    expect(f.mock.calls.slice(1).map((x) => x[1]).sort()).toEqual([0, 2, 3, 4]);
  });
});

describe("latest", () => {
  it("lets only the most recent request through", async () => {
    const wrap = latest();
    let resolveFirst!: (v: number) => void;
    const first = wrap(new Promise<number>((r) => (resolveFirst = r)));
    const second = wrap(Promise.resolve(2));
    resolveFirst(1);
    expect(await second).toBe(2);
    expect(await first).toBe(STALE);
  });
});
```

`web/src/playback.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { advance } from "./playback";

describe("advance", () => {
  it("moves by speed x fps x dt and keeps the fraction", () => {
    expect(advance(0, 1000, 1, 20, 100, 0)).toEqual({ t: 2, carry: 0, ended: false });
    const a = advance(0, 1000, 1, 20, 30, 0); // 0.6 frames
    expect(a.t).toBe(0);
    expect(advance(a.t, 1000, 1, 20, 30, a.carry).t).toBe(1);
    expect(advance(0, 1000, 16, 20, 100, 0).t).toBe(32);
  });
  it("stops at the last frame", () => {
    expect(advance(995, 1000, 16, 20, 100, 0)).toEqual({ t: 999, carry: 0, ended: true });
  });
});
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `cd web && npx vitest run src/draw.test.ts src/frames.test.ts src/playback.test.ts`
Expected: FAIL — cannot resolve `./draw`, `./frames`, `./playback`, `./run/RunContext`.

- [ ] **Step 3: Implement the helpers**

`web/src/draw.ts`:

```ts
import type { NDArray } from "./api";
import type { Unit } from "./types";

export type Img = { data: ArrayLike<number>; W: number; H: number; lo: number; hi: number; cmap?: "gray" | "heat" };

export function percentile(a: ArrayLike<number>, q: number): number {
  const s = Float64Array.from(a as ArrayLike<number>).sort();
  if (s.length === 0) return 0;
  const i = Math.min(s.length - 1, Math.max(0, Math.round((q / 100) * (s.length - 1))));
  return s[i];
}

export function imgFrom(arr: NDArray, cmap: "gray" | "heat" = "gray", qLo = 0.5, qHi = 99.8): Img {
  const [H, W] = arr.shape;
  const lo = percentile(arr.data, qLo);
  const hi = percentile(arr.data, qHi);
  return { data: arr.data, W, H, lo, hi: hi > lo ? hi : lo + 1, cmap };
}

const HEAT: [number, number, number][] = [
  [13, 8, 135],
  [84, 2, 163],
  [139, 10, 165],
  [185, 50, 137],
  [219, 92, 104],
  [244, 136, 73],
  [254, 188, 43],
  [240, 249, 33],
];

/** Plasma-like ramp for maps (rank, noise, background). */
export function heat(v: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, v)) * (HEAT.length - 1);
  const i = Math.min(HEAT.length - 2, Math.floor(x));
  const f = x - i;
  const a = HEAT[i];
  const b = HEAT[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export function fillImageData(img: Img, out: ImageData): ImageData {
  const k = 1 / (img.hi - img.lo);
  const d = out.data;
  for (let i = 0; i < img.W * img.H; i++) {
    const v = Math.min(1, Math.max(0, (img.data[i] - img.lo) * k));
    if (img.cmap === "heat") {
      const [r, g, b] = heat(v);
      d[4 * i] = r;
      d[4 * i + 1] = g;
      d[4 * i + 2] = b;
    } else {
      const g = Math.round(v * 255);
      d[4 * i] = g;
      d[4 * i + 1] = g;
      d[4 * i + 2] = g;
    }
    d[4 * i + 3] = 255;
  }
  return out;
}

export function pointInPolygon(x: number, y: number, poly: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Unit whose outline contains (x, y) in pixel-centre coordinates, else the nearest centroid. */
export function pickUnit(units: Unit[], x: number, y: number, maxDist = 8): number | null {
  for (const u of units) {
    if (u.polygons.some((p) => p.length >= 3 && pointInPolygon(x, y, p))) return u.id;
  }
  let best: number | null = null;
  let bestD = maxDist;
  for (const u of units) {
    const d = Math.hypot(u.centroid[1] - x, u.centroid[0] - y);
    if (d <= bestD) {
      best = u.id;
      bestD = d;
    }
  }
  return best;
}

export function drawPolygons(ctx: CanvasRenderingContext2D, units: Unit[], scale: number, sel: number | null) {
  const path = (u: Unit) => {
    ctx.beginPath();
    for (const poly of u.polygons) {
      poly.forEach(([x, y], i) => {
        const px = (x + 0.5) * scale;
        const py = (y + 0.5) * scale;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
    }
  };
  ctx.lineWidth = Math.max(1, scale / 2);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
  for (const u of units) {
    if (u.id === sel) continue;
    path(u);
    ctx.stroke();
  }
  const s = units.find((u) => u.id === sel);
  if (s) {
    ctx.lineWidth = Math.max(2, scale);
    ctx.strokeStyle = "#fb923c";
    path(s);
    ctx.stroke();
  }
}

export function drawPoints(ctx: CanvasRenderingContext2D, pts: number[][], scale: number, color: string) {
  ctx.fillStyle = color;
  const r = Math.max(1.5, scale * 0.9);
  for (const [h, w] of pts) {
    ctx.beginPath();
    ctx.arc((w + 0.5) * scale, (h + 0.5) * scale, r, 0, 2 * Math.PI);
    ctx.fill();
  }
}
```

`web/src/frames.ts`:

```ts
import type { NDArray } from "./api";

export class FrameCache {
  private lru = new Map<string, Promise<NDArray>>();

  constructor(
    private fetcher: (source: string, t: number) => Promise<NDArray>,
    private capacity = 160,
  ) {}

  private key(source: string, t: number) {
    return `${source}:${t}`;
  }

  has(source: string, t: number): boolean {
    return this.lru.has(this.key(source, t));
  }

  get(source: string, t: number): Promise<NDArray> {
    const k = this.key(source, t);
    const hit = this.lru.get(k);
    if (hit) {
      this.lru.delete(k);
      this.lru.set(k, hit);
      return hit;
    }
    const p = this.fetcher(source, t);
    p.catch(() => {
      if (this.lru.get(k) === p) this.lru.delete(k);
    });
    this.lru.set(k, p);
    while (this.lru.size > this.capacity) {
      this.lru.delete(this.lru.keys().next().value as string);
    }
    return p;
  }

  /** Start loading the frames around ``t`` that are not cached yet. */
  prefetch(source: string, t: number, T: number, ahead = 12, behind = 3): void {
    for (let d = -behind; d <= ahead; d++) {
      const u = t + d;
      if (d !== 0 && u >= 0 && u < T && !this.has(source, u)) {
        this.get(source, u).catch(() => {});
      }
    }
  }
}

export const STALE = Symbol("stale");

/** Wrap promises so that only the most recently wrapped one resolves to its value. */
export function latest(): <T>(p: Promise<T>) => Promise<T | typeof STALE> {
  let n = 0;
  return <T>(p: Promise<T>) => {
    const mine = ++n;
    return p.then((v) => (mine === n ? v : STALE));
  };
}
```

`web/src/playback.ts`:

```ts
/** Next frame while playing: ``speed`` x real time at ``fps``; stops at the last frame. */
export function advance(
  t: number,
  T: number,
  speed: number,
  fps: number,
  dtMs: number,
  carry: number,
): { t: number; carry: number; ended: boolean } {
  const step = (speed * fps * dtMs) / 1000 + carry;
  const whole = Math.floor(step);
  const next = Math.min(T - 1, t + whole);
  const ended = next >= T - 1;
  return { t: next, carry: ended ? 0 : step - whole, ended };
}
```

- [ ] **Step 4: Implement the image view, run state, registry, top area and run page**

`web/src/ImageView.tsx`:

```tsx
import { useEffect, useRef } from "react";
import { drawPoints, drawPolygons, fillImageData, pickUnit, type Img } from "./draw";
import type { Unit } from "./types";

export function ImageView(p: {
  image: Img | null;
  units?: Unit[];
  contours?: boolean;
  sel?: number | null;
  onPick?: (id: number | null) => void;
  points?: number[][];
  pointColor?: string;
  label?: string;
  scale?: number;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const buf = useRef<HTMLCanvasElement | null>(null);
  const scale = p.scale ?? 2;
  const units = p.units ?? [];

  useEffect(() => {
    const c = ref.current;
    const img = p.image;
    if (!c || !img) return;
    c.width = img.W * scale;
    c.height = img.H * scale;
    const b = buf.current ?? document.createElement("canvas");
    buf.current = b;
    b.width = img.W;
    b.height = img.H;
    const bctx = b.getContext("2d")!;
    bctx.putImageData(fillImageData(img, bctx.createImageData(img.W, img.H)), 0, 0);
    const ctx = c.getContext("2d")!;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(b, 0, 0, img.W * scale, img.H * scale);
    if (p.contours) drawPolygons(ctx, units, scale, p.sel ?? null);
    if (p.points) drawPoints(ctx, p.points, scale, p.pointColor ?? "#fb923c");
  }, [p.image, units, p.contours, p.sel, p.points, p.pointColor, scale]);

  if (!p.image) return <div className="imageview" aria-busy="true" style={{ aspectRatio: "1 / 1" }} />;
  const img = p.image;
  return (
    <canvas
      ref={ref}
      className={p.className ?? "imageview"}
      style={{ aspectRatio: `${img.W} / ${img.H}` }}
      role="img"
      aria-label={p.label ?? "image"}
      onClick={(e) => {
        if (!p.onPick) return;
        const r = e.currentTarget.getBoundingClientRect();
        const x = ((e.clientX - r.left) / r.width) * img.W - 0.5;
        const y = ((e.clientY - r.top) / r.height) * img.H - 0.5;
        p.onPick(pickUnit(units, x, y));
      }}
    />
  );
}
```

`web/src/run/RunContext.tsx`:

```tsx
import { createContext, useContext } from "react";
import type { SegOption } from "../controls";
import type { Img } from "../draw";
import type { FrameCache } from "../frames";
import type { Available, Summary, Unit } from "../types";

export type Source = "max" | "raw" | "pre" | "recon";

export type RunState = {
  summary: Summary;
  units: Unit[];
  maxProj: Img;
  t: number;
  setT: (t: number) => void;
  sel: number | null;
  setSel: (id: number | null) => void;
  cache: FrameCache;
};

export const RunCtx = createContext<RunState | null>(null);

export function useRun(): RunState {
  const s = useContext(RunCtx);
  if (!s) throw new Error("useRun outside a run page");
  return s;
}

export function sourceOptions(a: Available): SegOption<Source>[] {
  return [
    { value: "max", label: "Max projection" },
    { value: "raw", label: "Raw", disabled: !a.raw, title: a.raw ? undefined : "raw videos not found (moved or unplugged)" },
    {
      value: "pre",
      label: "Preprocessed",
      disabled: !a.pre,
      title: a.pre ? undefined : "tile store not kept (run with --keep-store)",
    },
    { value: "recon", label: "Reconstruction" },
  ];
}
```

`web/src/run/registry.ts`:

```ts
import type { JSX } from "react";
import type { Page } from "../router";

export const PAGE_LABELS: Record<Page, string> = {
  overview: "Overview",
  frames: "Frames",
  motion: "Motion",
  compression: "Compression",
  seeds: "Seeds",
  units: "Units",
  traces: "Traces",
};

/** Stage pages by route; each page task adds its entry. Unregistered pages are hidden. */
export const PAGE_VIEWS: Partial<Record<Page, () => JSX.Element>> = {};
```

`web/src/run/TopArea.tsx`:

```tsx
import { useEffect, useRef, useState } from "react";
import { Computing } from "../api";
import { Segmented, Slider, Toggle } from "../controls";
import type { Img } from "../draw";
import { STALE, latest } from "../frames";
import { fmtClock } from "../format";
import { ImageView } from "../ImageView";
import { advance } from "../playback";
import { sourceOptions, useRun, type Source } from "./RunContext";

const SPEEDS = ["1", "4", "16"] as const;

export function TopArea() {
  const run = useRun();
  const { summary, t, setT } = run;
  const [source, setSource] = useState<Source>("max");
  const [contours, setContours] = useState(true);
  const [gain, setGain] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>("1");
  const [frame, setFrame] = useState<Img | null>(null);
  const [note, setNote] = useState("");
  const [retry, setRetry] = useState(0);
  const wrap = useRef(latest());
  const tRef = useRef(t);
  tRef.current = t;
  const fps = summary.fps ?? 20;

  useEffect(() => {
    if (source === "max") return;
    setNote("");
    let timer: ReturnType<typeof setTimeout> | undefined;
    wrap.current(run.cache.get(source, t)).then(
      (a) => {
        if (a === STALE) return;
        setFrame({ data: a.data, W: summary.W, H: summary.H, lo: 0, hi: 255 });
      },
      (e: unknown) => {
        if (e instanceof Computing) {
          setNote("Computing the frame contrast…");
          timer = setTimeout(() => setRetry((r) => r + 1), 1500);
        } else setNote(e instanceof Error ? e.message : String(e));
      },
    );
    run.cache.prefetch(source, t, summary.T, playing ? 24 : 12, playing ? 0 : 3);
    return () => clearTimeout(timer);
  }, [source, t, run.cache, summary.W, summary.H, summary.T, playing, retry]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    let carry = 0;
    const tick = (now: number) => {
      const a = advance(tRef.current, summary.T, Number(speed), fps, now - last, carry);
      last = now;
      carry = a.carry;
      if (a.t !== tRef.current) setT(a.t);
      if (a.ended) setPlaying(false);
      else raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, fps, summary.T, setT]);

  const base = source === "max" ? run.maxProj : frame;
  const shown = base && { ...base, hi: base.lo + (base.hi - base.lo) / gain };
  return (
    <section className="map" aria-label="Field of view" style={{ aspectRatio: `${summary.W} / ${summary.H}`, maxHeight: "72vh" }}>
      <ImageView
        image={shown}
        units={run.units}
        contours={contours}
        sel={run.sel}
        onPick={run.setSel}
        label={`${source} frame with unit outlines`}
        scale={summary.W > 400 ? 2 : 4}
      />
      <div className="overlay">
        <div className="row">
          <Segmented label="Image" glass options={sourceOptions(summary.available)} value={source} onChange={setSource} />
          <span className="spacer" />
          <div className="glass" style={{ padding: "6px 14px" }}>
            <Toggle label="Contours" checked={contours} onChange={setContours} />
          </div>
        </div>
        <div className="glass player">
          <button
            type="button"
            className="btn"
            aria-label={playing ? "Pause" : "Play"}
            disabled={source === "max"}
            title={source === "max" ? "choose Raw, Preprocessed or Reconstruction to play" : undefined}
            onClick={() => setPlaying(!playing)}
            style={{ background: "rgba(255,255,255,0.92)", color: "#0b1220", width: 44, padding: 0, justifyContent: "center" }}
          >
            {playing ? "❚❚" : "▶"}
          </button>
          <span className="mono" style={{ minWidth: 80 }}>
            {fmtClock(t, summary.fps)}
          </span>
          <input
            type="range"
            aria-label="Scrub recording"
            min={0}
            max={summary.T - 1}
            value={t}
            onChange={(e) => setT(Number(e.target.value))}
          />
          <Segmented label="Speed" glass options={SPEEDS.map((s) => ({ value: s, label: `${s}×` }))} value={speed} onChange={setSpeed} />
          <div style={{ minWidth: 180 }}>
            <Slider label="Contrast" min={0.5} max={4} step={0.1} value={gain} onChange={setGain} format={(v) => `${v.toFixed(1)}×`} />
          </div>
        </div>
        {note && (
          <span className="glass" style={{ padding: "6px 14px", alignSelf: "flex-start" }}>
            {note}
          </span>
        )}
      </div>
    </section>
  );
}
```

`web/src/run/RunPage.tsx`:

```tsx
import { useMemo, useState } from "react";
import { getArray, getJSON } from "../api";
import { ErrorNote, Loading, Segmented } from "../controls";
import { imgFrom } from "../draw";
import { FrameCache } from "../frames";
import { fmtDuration, fmtInt } from "../format";
import { useResource } from "../poll";
import { go, PAGES, type Page } from "../router";
import type { Summary, UnitsResponse } from "../types";
import { PAGE_LABELS, PAGE_VIEWS } from "./registry";
import { RunCtx, type RunState } from "./RunContext";
import { TopArea } from "./TopArea";

export function RunPage({ id, page }: { id: string; page: Page }) {
  const summary = useResource(`summary:${id}`, () => getJSON<Summary>(`/runs/${id}/summary`));
  const units = useResource(`units:${id}`, () => getJSON<UnitsResponse>(`/runs/${id}/units`));
  const maxProj = useResource(`maxproj:${id}`, () => getArray(`/runs/${id}/image/max_proj`));
  const [t, setT] = useState(0);
  const [sel, setSel] = useState<number | null>(null);
  const cache = useMemo(
    () => new FrameCache((source, f) => getArray(`/runs/${id}/frame/${f}?source=${source}`)),
    [id],
  );
  const maxImg = useMemo(() => (maxProj.data ? imgFrom(maxProj.data) : null), [maxProj.data]);

  if (summary.error) return <div className="page"><ErrorNote message={summary.error} /></div>;
  if (units.error) return <div className="page"><ErrorNote message={units.error} /></div>;
  if (!summary.data || !units.data || !maxImg) {
    return <div className="page"><Loading res={summary} what="run">{() => <p className="muted">Loading units…</p>}</Loading></div>;
  }
  const s = summary.data;
  const state: RunState = { summary: s, units: units.data.units, maxProj: maxImg, t, setT, sel, setSel, cache };
  const View = PAGE_VIEWS[page];
  const pages = PAGES.filter((p) => PAGE_VIEWS[p]);
  return (
    <RunCtx.Provider value={state}>
      <div className="page">
        <div className="row">
          <div>
            <span className="label">CaAnFast · run</span>
            <h1>{s.name}</h1>
          </div>
          <span className="spacer" />
          <span className="muted">
            {fmtInt(s.T)} frames · {s.W} × {s.H}
            {s.fps ? ` · ${fmtDuration(s.T / s.fps)} at ${s.fps} fps` : " · frame rate unknown"}
          </span>
        </div>
        <TopArea />
        <Segmented
          label="Stage"
          options={pages.map((p) => ({ value: p, label: PAGE_LABELS[p] }))}
          value={page}
          onChange={(p) => go({ name: "run", id, page: p })}
        />
        {View ? <View /> : <p className="muted">This page is not available.</p>}
      </div>
    </RunCtx.Provider>
  );
}
```

In `web/src/App.tsx` add `import { RunPage } from "./run/RunPage";` and, after the home line:

```tsx
      {route.name === "run" && <RunPage key={route.id} id={route.id} page={route.page} />}
```

- [ ] **Step 5: Run the checks and look at it**

Run: `cd web && npm run typecheck && npm test`
Expected: no type errors; all tests pass.

Manual check (dev server as in Task 2): open a run. Expected: the max projection with white outlines; clicking a footprint turns it orange; switching to Reconstruction and pressing ▶ plays; Raw / Preprocessed are disabled with a tooltip when the run lacks them.

- [ ] **Step 6: Commit**

```bash
git add web/src
git commit -m "Add the run page's top area: max projection, frames with contours, scrubbing and playback"
```

---

### Task 4: Chart, Overview, Frames and Motion pages

**Files:**
- Create: `web/src/Chart.tsx`, `web/src/run/pages/Overview.tsx`, `web/src/run/pages/Frames.tsx`, `web/src/run/pages/Motion.tsx`
- Modify: `web/src/run/registry.ts`
- Test: `web/src/run/pages/overview.test.ts`

**Interfaces:**
- Consumes: Task 3 (`useRun`, `ImageView`, `FrameCache`, `PAGE_VIEWS`); routes `/runs/{id}/motion`, `/runs/{id}/motion/residual`.
- Produces (`Chart.tsx`): `cssVar(name): string`, `Chart({ data: uPlot.AlignedData, series: uPlot.Series[], height?, bands?, cursorX?, xRange?, onSelectX?, onResetX?, sync?, xLabel? })`.
- Produces (`Overview.tsx`): `funnel(counts): {name: string; value: number}[]`, `stageBars(stages): {name: string; seconds: number; pct: number}[]`, `Overview()`.
- Produces: `Frames()`, `Motion()`; registry entries `overview`, `frames`, `motion`.

- [ ] **Step 1: Write the failing test**

`web/src/run/pages/overview.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { funnel, stageBars } from "./Overview";

describe("overview data", () => {
  it("builds the seed funnel in order", () => {
    const f = funnel({ seeds_init: 59504, seeds_pnr: 39769, seeds_merged: 582, units_final: 503 });
    expect(f.map((x) => x.value)).toEqual([59504, 39769, 582, 503]);
    expect(f[0].name).toBe("Seeds");
  });
  it("scales stage bars to the longest stage", () => {
    const b = stageBars([
      { stage: "pass1", seconds: 450 },
      { stage: "pass2", seconds: 503 },
      { stage: "probe", seconds: 0.5 },
    ]);
    expect(b.map((x) => x.pct)).toEqual([(450 / 503) * 100, 100, (0.5 / 503) * 100]);
  });
});
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd web && npx vitest run src/run/pages/overview.test.ts`
Expected: FAIL — cannot resolve `./Overview`.

- [ ] **Step 3: Implement the chart**

`web/src/Chart.tsx`:

```tsx
import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#2563eb";
}

export type ChartProps = {
  data: uPlot.AlignedData;
  series: uPlot.Series[];
  height?: number;
  bands?: uPlot.Band[];
  cursorX?: number | null;
  xRange?: [number, number] | null;
  onSelectX?: (x0: number, x1: number) => void;
  onResetX?: () => void;
  sync?: string;
  xLabel?: string;
};

export function Chart(p: ChartProps) {
  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const props = useRef(p);
  props.current = p;

  useEffect(() => {
    const el = host.current!;
    const axis = { stroke: cssVar("--muted"), grid: { stroke: cssVar("--track") }, ticks: { stroke: cssVar("--track") } };
    const opts: uPlot.Options = {
      width: el.clientWidth || 600,
      height: p.height ?? 180,
      series: [{ label: p.xLabel ?? "x" }, ...p.series],
      bands: p.bands,
      legend: { show: false },
      cursor: {
        sync: p.sync ? { key: p.sync } : undefined,
        drag: { x: true, y: false, setScale: !p.onSelectX },
      },
      axes: [{ ...axis, label: p.xLabel }, axis],
      hooks: {
        draw: [
          (u) => {
            const x = props.current.cursorX;
            if (x == null) return;
            const cx = u.valToPos(x, "x", true);
            const ctx = u.ctx;
            ctx.save();
            ctx.strokeStyle = cssVar("--hot");
            ctx.lineWidth = 2 * devicePixelRatio;
            ctx.beginPath();
            ctx.moveTo(cx, u.bbox.top);
            ctx.lineTo(cx, u.bbox.top + u.bbox.height);
            ctx.stroke();
            ctx.restore();
          },
        ],
        setSelect: [
          (u) => {
            const cb = props.current.onSelectX;
            if (!cb || u.select.width < 3) return;
            const x0 = u.posToVal(u.select.left, "x");
            const x1 = u.posToVal(u.select.left + u.select.width, "x");
            u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
            cb(x0, x1);
          },
        ],
      },
    };
    const u = new uPlot(opts, p.data, el);
    plot.current = u;
    const ro = new ResizeObserver(() => u.setSize({ width: el.clientWidth, height: props.current.height ?? 180 }));
    ro.observe(el);
    el.ondblclick = () => props.current.onResetX?.();
    return () => {
      ro.disconnect();
      u.destroy();
      plot.current = null;
    };
    // the chart is rebuilt only when its layout changes; data and range update in place
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.series.length, p.height, p.sync, p.bands?.length]);

  useEffect(() => {
    plot.current?.setData(p.data);
  }, [p.data]);
  useEffect(() => {
    if (plot.current && p.xRange) plot.current.setScale("x", { min: p.xRange[0], max: p.xRange[1] });
  }, [p.xRange, p.data]);
  useEffect(() => {
    plot.current?.redraw(false);
  }, [p.cursorX]);

  return <div ref={host} className="chart" />;
}
```

- [ ] **Step 4: Implement the three pages**

`web/src/run/pages/Overview.tsx`:

```tsx
import { fmtBytes, fmtDuration, fmtInt, stageLabel } from "../../format";
import type { Stage } from "../../types";
import { useRun } from "../RunContext";

export function funnel(c: Record<string, number>): { name: string; value: number }[] {
  return [
    { name: "Seeds", value: c.seeds_init ?? 0 },
    { name: "After PNR", value: c.seeds_pnr ?? 0 },
    { name: "Merged", value: c.seeds_merged ?? 0 },
    { name: "Units", value: c.units_final ?? 0 },
  ];
}

export function stageBars(stages: Stage[]): { name: string; seconds: number; pct: number }[] {
  const max = Math.max(1e-9, ...stages.map((s) => s.seconds));
  return stages.map((s) => ({ name: s.stage, seconds: s.seconds, pct: (s.seconds / max) * 100 }));
}

export function Overview() {
  const { summary: s } = useRun();
  const f = funnel(s.counts);
  const top = Math.max(1, f[0].value);
  const changed = Object.entries(s.params_changed);
  return (
    <div className="bento">
      <section className="card">
        <span className="label">Units</span>
        <span className="big" style={{ color: "var(--accent)" }}>{fmtInt(s.counts.units_final ?? 0)}</span>
        <span className="muted">after merging {fmtInt(s.counts.seeds_merged ?? 0)} candidates</span>
      </section>
      <section className="card">
        <span className="label">Run time</span>
        <span className="big">{s.wall_total ? fmtDuration(s.wall_total) : "–"}</span>
        <span className="muted">{fmtInt(s.T)} frames{s.workers ? ` · ${s.workers} workers` : ""}</span>
      </section>
      <section className="card">
        <span className="label">Peak memory</span>
        <span className="big">{s.peak_rss ? fmtBytes(s.peak_rss) : "–"}</span>
        <span className="muted">
          {s.peak_rss_children ? `${fmtBytes(s.peak_rss_children)} per worker` : ""}
          {s.bytes.raw ? ` · raw ${fmtBytes(s.bytes.raw)}` : ""}
        </span>
      </section>
      <section className="card">
        <span className="label">Motion</span>
        <span className="big">{s.motion.max_abs_shift_px?.toFixed(1) ?? "–"}<span className="muted" style={{ fontSize: 18 }}> px</span></span>
        <span className="muted">largest shift · median quality {s.motion.quality_median?.toFixed(2) ?? "–"}</span>
      </section>
      <section className="card span2" aria-label="Seed funnel">
        <span className="label">From seeds to units</span>
        {f.map((x) => (
          <div className="barrow" key={x.name}>
            <span>{x.name}</span>
            <div className="bar"><span style={{ width: `${Math.max(1.5, (x.value / top) * 100)}%` }} /></div>
            <span className="mono">{fmtInt(x.value)}</span>
          </div>
        ))}
      </section>
      <section className="card span2" aria-label="Stage timings">
        <span className="label">Where the time goes</span>
        {stageBars(s.stages).filter((b) => b.seconds >= 0.5).map((b) => (
          <div className="barrow" key={b.name}>
            <span>{stageLabel(b.name)}</span>
            <div className="bar"><span style={{ width: `${b.pct}%`, background: "var(--text)" }} /></div>
            <span className="mono">{fmtDuration(b.seconds)}</span>
          </div>
        ))}
      </section>
      <section className="card span4" aria-label="Changed parameters">
        <span className="label">Parameters changed from the defaults</span>
        {changed.length === 0 ? (
          <span className="muted">None: this run used the defaults.</span>
        ) : (
          <table className="data">
            <thead><tr><th>Parameter</th><th>This run</th><th>Default</th></tr></thead>
            <tbody>
              {changed.map(([k, v]) => (
                <tr key={k}><td className="mono">{k}</td><td className="mono">{JSON.stringify(v.value)}</td><td className="mono muted">{JSON.stringify(v.default)}</td></tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
```

`web/src/run/pages/Frames.tsx`:

```tsx
import { useEffect, useState } from "react";
import { Computing } from "../../api";
import type { Img } from "../../draw";
import { STALE, latest } from "../../frames";
import { ImageView } from "../../ImageView";
import { sourceOptions, useRun } from "../RunContext";

const SOURCES = ["raw", "pre", "recon"] as const;

function FrameTile({ source }: { source: (typeof SOURCES)[number] }) {
  const run = useRun();
  const opt = sourceOptions(run.summary.available).find((o) => o.value === source)!;
  const [img, setImg] = useState<Img | null>(null);
  const [err, setErr] = useState("");
  const [retry, setRetry] = useState(0);
  const [wrap] = useState(latest);
  useEffect(() => {
    if (opt.disabled) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    wrap(run.cache.get(source, run.t)).then(
      (a) => {
        if (a === STALE) return;
        setErr("");
        setImg({ data: a.data, W: run.summary.W, H: run.summary.H, lo: 0, hi: 255 });
      },
      (e: unknown) => {
        if (e instanceof Computing) {
          setErr("Computing the contrast…");
          timer = setTimeout(() => setRetry((r) => r + 1), 1500);
        } else setErr(e instanceof Error ? e.message : String(e));
      },
    );
    return () => clearTimeout(timer);
  }, [source, run.t, run.cache, run.summary.W, run.summary.H, opt.disabled, wrap, retry]);
  return (
    <section className={opt.disabled ? "card dim" : "card"} aria-label={opt.label}>
      <span className="label">{opt.label}</span>
      {opt.disabled ? (
        <span className="muted">Not available: {opt.title}.</span>
      ) : (
        <>
          <ImageView image={img} units={run.units} contours={false} label={`${opt.label} frame`} />
          {err && <span className="muted">{err}</span>}
        </>
      )}
    </section>
  );
}

export function Frames() {
  return (
    <div className="bento" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
      {SOURCES.map((s) => (
        <FrameTile key={s} source={s} />
      ))}
    </div>
  );
}
```

`web/src/run/pages/Motion.tsx`:

```tsx
import { useMemo } from "react";
import { getArray } from "../../api";
import { Chart, cssVar } from "../../Chart";
import { Loading } from "../../controls";
import { useResource } from "../../poll";
import { useRun } from "../RunContext";

function columns(data: ArrayLike<number>, n: number, k: number): Float64Array[] {
  const out = Array.from({ length: k }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let j = 0; j < k; j++) out[j][i] = data[i * k + j];
  return out;
}

export function Motion() {
  const { summary: s, t } = useRun();
  const motion = useResource(`motion:${s.id}`, () => getArray(`/runs/${s.id}/motion`));
  const resid = useResource(`resid:${s.id}`, () => getArray(`/runs/${s.id}/motion/residual`));
  const fps = s.fps;
  const xs = useMemo(() => Float64Array.from({ length: s.T }, (_, i) => (fps ? i / fps / 60 : i)), [s.T, fps]);
  const xLabel = fps ? "minutes" : "frame";
  const cursor = fps ? t / fps / 60 : t;
  return (
    <div className="bento">
      <section className="card span4" aria-label="Motion shifts">
        <span className="label">Shifts (px) · y in blue, x in orange</span>
        <Loading res={motion} what="motion">
          {(m) => {
            const [dy, dx, q] = columns(m.data, m.shape[0], 3);
            return (
              <>
                <Chart
                  data={[xs, dy, dx]}
                  series={[{ stroke: cssVar("--accent"), width: 1 }, { stroke: cssVar("--hot"), width: 1 }]}
                  cursorX={cursor}
                  xLabel={xLabel}
                  sync="motion"
                />
                <span className="label">Motion quality (correlation with the template)</span>
                <Chart data={[xs, q]} series={[{ stroke: cssVar("--ok"), width: 1 }]} height={140} cursorX={cursor} xLabel={xLabel} sync="motion" />
              </>
            );
          }}
        </Loading>
      </section>
      <section className="card span4" aria-label="Residual shift per block">
        <span className="label">Leftover shift per block after correction (should stay near 0 px)</span>
        <Loading res={resid} what="the leftover shift per block">
          {(r) => {
            const n = r.shape[0];
            const [ry, rx, rc] = columns(r.data, n, 3);
            const bx = Float64Array.from({ length: n }, (_, i) => i);
            return (
              <>
                <Chart data={[bx, ry, rx]} series={[{ stroke: cssVar("--accent"), width: 1 }, { stroke: cssVar("--hot"), width: 1 }]} xLabel="block" height={160} />
                <Chart data={[bx, rc]} series={[{ stroke: cssVar("--ok"), width: 1 }]} xLabel="block" height={120} />
              </>
            );
          }}
        </Loading>
      </section>
    </div>
  );
}
```

Replace `PAGE_VIEWS` in `web/src/run/registry.ts`:

```ts
import { Frames } from "./pages/Frames";
import { Motion } from "./pages/Motion";
import { Overview } from "./pages/Overview";

export const PAGE_VIEWS: Partial<Record<Page, () => JSX.Element>> = {
  overview: Overview,
  frames: Frames,
  motion: Motion,
};
```

- [ ] **Step 5: Run the checks and look at it**

Run: `cd web && npm run typecheck && npm test`
Expected: no type errors; all tests pass.

Manual check: open a run. Expected: Overview shows units, run time, memory, the funnel, stage bars and changed parameters; Frames shows three frames that follow the scrubber; Motion shows shifts with an orange playhead at the scrubber position; the residual tile shows "Computing…" once, then its charts.

- [ ] **Step 6: Commit**

```bash
git add web/src
git commit -m "Add the Overview, Frames and Motion pages and the chart wrapper"
```

---

### Task 5: Compression, Seeds and Units pages

**Files:**
- Create: `web/src/run/pages/Compression.tsx`, `web/src/run/pages/Seeds.tsx`, `web/src/run/pages/Units.tsx`
- Modify: `web/src/run/registry.ts`
- Test: `web/src/run/pages/units.test.ts`

**Interfaces:**
- Consumes: Tasks 3–4; routes `/runs/{id}/image/rank|noise|background`, `/runs/{id}/seeds`, `/runs/{id}/units?stage=init`.
- Produces (`Compression.tsx`): `rankHistogram(ranks: number[], bins = 10): { lo: number; hi: number; n: number }[]`, `Compression()`.
- Produces (`Units.tsx`): `type SortKey = "id" | "size_px" | "peak" | "row" | "col"`, `sortUnits(units, key, desc): Unit[]`, `Units()`.
- Produces: `Seeds()`; registry entries `compression`, `seeds`, `units`.

- [ ] **Step 1: Write the failing test**

`web/src/run/pages/units.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { Unit } from "../../types";
import { rankHistogram } from "./Compression";
import { sortUnits } from "./Units";

const u = (id: number, size: number, peak: number | null, c: [number, number]): Unit => ({
  id,
  size_px: size,
  peak,
  centroid: c,
  polygons: [],
});

describe("units table", () => {
  it("sorts by any column, nulls last", () => {
    const us = [u(0, 50, 2, [5, 9]), u(1, 80, null, [1, 3]), u(2, 20, 7, [9, 1])];
    expect(sortUnits(us, "size_px", true).map((x) => x.id)).toEqual([1, 0, 2]);
    expect(sortUnits(us, "peak", true).map((x) => x.id)).toEqual([2, 0, 1]);
    expect(sortUnits(us, "col", false).map((x) => x.id)).toEqual([2, 1, 0]);
  });
});

describe("rank histogram", () => {
  it("counts tiles per rank bin", () => {
    const h = rankHistogram([1, 2, 2, 10], 3);
    expect(h.map((b) => b.n)).toEqual([3, 0, 1]);
    expect(h[0].lo).toBe(1);
    expect(h.at(-1)!.hi).toBe(10);
  });
});
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd web && npx vitest run src/run/pages/units.test.ts`
Expected: FAIL — cannot resolve `./Compression` and `./Units`.

- [ ] **Step 3: Implement**

`web/src/run/pages/Compression.tsx`:

```tsx
import { getArray } from "../../api";
import { Loading } from "../../controls";
import { imgFrom } from "../../draw";
import { fmtInt } from "../../format";
import { ImageView } from "../../ImageView";
import { useResource } from "../../poll";
import { useRun } from "../RunContext";

export function rankHistogram(ranks: number[], bins = 10): { lo: number; hi: number; n: number }[] {
  if (ranks.length === 0) return [];
  const lo = Math.min(...ranks);
  const hi = Math.max(...ranks);
  const w = Math.max(1e-9, (hi - lo) / bins);
  const out = Array.from({ length: bins }, (_, i) => ({ lo: lo + i * w, hi: lo + (i + 1) * w, n: 0 }));
  for (const r of ranks) out[Math.min(bins - 1, Math.floor((r - lo) / w))].n++;
  return out;
}

function MapTile({ name, title, note }: { name: string; title: string; note: string }) {
  const { summary: s } = useRun();
  const res = useResource(`img:${s.id}:${name}`, () => getArray(`/runs/${s.id}/image/${name}`));
  return (
    <section className="card span2" aria-label={title}>
      <span className="label">{title}</span>
      <Loading res={res} what={title.toLowerCase()}>
        {(a) => <ImageView image={imgFrom(a, "heat", 1, 99)} label={title} />}
      </Loading>
      <span className="muted">{note}</span>
    </section>
  );
}

export function Compression() {
  const { summary: s } = useRun();
  const hist = rankHistogram(s.ranks);
  const top = Math.max(1, ...hist.map((b) => b.n));
  const sum = s.ranks.reduce((a, b) => a + b, 0);
  return (
    <div className="bento">
      <MapTile name="rank" title="Rank per tile" note={`${s.ranks.length} tiles of ${s.tile} px`} />
      <MapTile name="noise" title="Noise per pixel" note="standard deviation of the residual noise" />
      <section className="card">
        <span className="label">Total rank</span>
        <span className="big">{fmtInt(sum)}</span>
        <span className="muted">largest tile {Math.max(...s.ranks)}</span>
      </section>
      <section className="card span3" aria-label="Rank histogram">
        <span className="label">Tiles per rank</span>
        {hist.map((b) => (
          <div className="barrow" key={b.lo}>
            <span className="mono">{Math.round(b.lo)}–{Math.round(b.hi)}</span>
            <div className="bar"><span style={{ width: `${(b.n / top) * 100}%` }} /></div>
            <span className="mono">{b.n}</span>
          </div>
        ))}
      </section>
      <MapTile name="background" title="Background" note="spatial background b of the final CNMF" />
    </div>
  );
}
```

`web/src/run/pages/Seeds.tsx`:

```tsx
import { useState } from "react";
import { getJSON } from "../../api";
import { Loading, Segmented } from "../../controls";
import { fmtInt } from "../../format";
import { ImageView } from "../../ImageView";
import { useResource } from "../../poll";
import type { Seeds as SeedsT } from "../../types";
import { useRun } from "../RunContext";

type Step = keyof SeedsT;
const STEPS: { value: Step; label: string }[] = [
  { value: "init", label: "Initial" },
  { value: "pnr", label: "After PNR" },
  { value: "merged", label: "Merged" },
];

export function Seeds() {
  const { summary: s, maxProj } = useRun();
  const [step, setStep] = useState<Step>("merged");
  const res = useResource(`seeds:${s.id}`, () => getJSON<SeedsT>(`/runs/${s.id}/seeds`));
  return (
    <div className="bento">
      <section className="card span3" aria-label="Seeds on the max projection">
        <div className="row">
          <span className="label">Seeds on the max projection</span>
          <span className="spacer" />
          <Segmented label="Seed step" options={STEPS} value={step} onChange={setStep} />
        </div>
        <Loading res={res} what="the seeds (older runs: about 2.5 min on 3 h)">
          {(sd) => <ImageView image={maxProj} points={sd[step]} label={`${step} seeds`} />}
        </Loading>
      </section>
      <section className="card" aria-label="Seed counts">
        <span className="label">Seeds per step</span>
        {res.data &&
          STEPS.map((x) => (
            <div key={x.value}>
              <span className="muted">{x.label}</span>
              <div className="big" style={{ fontSize: 28 }}>{fmtInt(res.data![x.value].length)}</div>
            </div>
          ))}
      </section>
    </div>
  );
}
```

`web/src/run/pages/Units.tsx`:

```tsx
import { useMemo, useState } from "react";
import { getJSON } from "../../api";
import { Loading } from "../../controls";
import { ImageView } from "../../ImageView";
import { useResource } from "../../poll";
import type { Unit, UnitsResponse } from "../../types";
import { useRun } from "../RunContext";

export type SortKey = "id" | "size_px" | "peak" | "row" | "col";

function key(u: Unit, k: SortKey): number | null {
  if (k === "row") return u.centroid[0];
  if (k === "col") return u.centroid[1];
  return u[k];
}

export function sortUnits(units: Unit[], k: SortKey, desc: boolean): Unit[] {
  return [...units].sort((a, b) => {
    const x = key(a, k);
    const y = key(b, k);
    if (x === null) return 1;
    if (y === null) return -1;
    return desc ? y - x : x - y;
  });
}

const COLS: { k: SortKey; label: string }[] = [
  { k: "id", label: "Unit" },
  { k: "size_px", label: "Size (px)" },
  { k: "peak", label: "Peak" },
  { k: "row", label: "Row" },
  { k: "col", label: "Column" },
];

export function Units() {
  const { summary: s, units, maxProj, sel, setSel } = useRun();
  const init = useResource(`units-init:${s.id}`, () => getJSON<UnitsResponse>(`/runs/${s.id}/units?stage=init`));
  const [sortKey, setSortKey] = useState<SortKey>("peak");
  const [desc, setDesc] = useState(true);
  const rows = useMemo(() => sortUnits(units, sortKey, desc), [units, sortKey, desc]);
  return (
    <div className="bento">
      <section className="card span2" aria-label="Initial units">
        <span className="label">Initial units (before CNMF){init.data ? ` · ${init.data.units.length}` : ""}</span>
        <Loading res={init} what="the initial units">
          {(d) => <ImageView image={maxProj} units={d.units} contours label="initial unit outlines" />}
        </Loading>
      </section>
      <section className="card span2" aria-label="Final units">
        <span className="label">Final units · {units.length} · click one to select it</span>
        <ImageView image={maxProj} units={units} contours sel={sel} onPick={setSel} label="final unit outlines" />
      </section>
      <section className="card span4" aria-label="Unit table">
        <div className="scroll">
          <table className="data">
            <thead>
              <tr>
                {COLS.map((c) => (
                  <th key={c.k} aria-sort={sortKey === c.k ? (desc ? "descending" : "ascending") : "none"}>
                    <button type="button" onClick={() => (sortKey === c.k ? setDesc(!desc) : (setSortKey(c.k), setDesc(true)))}>
                      {c.label}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((u) => (
                <tr key={u.id} aria-selected={u.id === sel} onClick={() => setSel(u.id)} style={{ cursor: "pointer" }}>
                  <td className="mono">{u.id}</td>
                  <td className="mono">{u.size_px}</td>
                  <td className="mono">{u.peak?.toFixed(2) ?? "–"}</td>
                  <td className="mono">{u.centroid[0].toFixed(0)}</td>
                  <td className="mono">{u.centroid[1].toFixed(0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
```

Add to `PAGE_VIEWS` in `web/src/run/registry.ts` (with the imports):

```ts
import { Compression } from "./pages/Compression";
import { Seeds } from "./pages/Seeds";
import { Units } from "./pages/Units";
// …
  compression: Compression,
  seeds: Seeds,
  units: Units,
```

- [ ] **Step 4: Run the checks and look at it**

Run: `cd web && npm run typecheck && npm test`
Expected: no type errors; all tests pass.

Manual check: Compression shows two heat maps, the total rank and the histogram; Seeds switches between the three steps; Units shows initial and final outlines; sorting the table works; clicking a row highlights the unit in the top area.

- [ ] **Step 5: Commit**

```bash
git add web/src
git commit -m "Add the Compression, Seeds and Units pages"
```

---

### Task 6: Traces page with the low-pass

**Files:**
- Create: `web/src/run/pages/Traces.tsx`, `web/src/run/traceMode.ts`
- Modify: `web/src/run/registry.ts`
- Test: `web/src/run/traceMode.test.ts`

**Interfaces:**
- Consumes: Tasks 3–4; routes `/runs/{id}/traces/overview`, `/runs/{id}/traces/{unit}`, `/runs/{id}/zooms`, `POST /runs/{id}/fps`.
- Produces (`traceMode.ts`): `FULL_RES_MAX = 20000`, `traceMode(range: [number, number] | null, T: number): "overview" | "window"`, `clampRange(x0, x1, T): [number, number]`, `lowpassState(fps: number | null, on: boolean, hz: number): { enabled: boolean; hz: number | null; max: number; reason?: string }`, `overviewColumns(ov: NDArray, unit: number): { lo: Float64Array; hi: Float64Array }`.
- Produces: `Traces()`; registry entry `traces`.

- [ ] **Step 1: Write the failing test**

`web/src/run/traceMode.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { clampRange, lowpassState, overviewColumns, traceMode } from "./traceMode";

describe("trace mode", () => {
  it("uses the overview for wide views and full traces for narrow windows", () => {
    expect(traceMode(null, 216000)).toBe("overview");
    expect(traceMode([0, 216000], 216000)).toBe("overview");
    expect(traceMode([1000, 21000], 216000)).toBe("window");
    expect(traceMode(null, 3000)).toBe("window");
  });
  it("clamps ranges into the recording", () => {
    expect(clampRange(-50, 120.7, 100)).toEqual([0, 100]);
    expect(clampRange(30, 30, 100)).toEqual([30, 31]);
  });
});

describe("low-pass state", () => {
  it("is disabled without a frame rate", () => {
    const s = lowpassState(null, true, 5);
    expect(s.enabled).toBe(false);
    expect(s.hz).toBeNull();
    expect(s.reason).toMatch(/frame rate/);
  });
  it("clamps the cutoff to Nyquist and passes it only when on", () => {
    expect(lowpassState(20, true, 15)).toMatchObject({ enabled: true, hz: 10, max: 10 });
    expect(lowpassState(20, false, 5).hz).toBeNull();
  });
});

describe("overview columns", () => {
  it("reads min and max rows of one unit", () => {
    const ov = { shape: [2, 2, 3], dtype: "float32" as const, data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) };
    const c = overviewColumns(ov, 1);
    expect(Array.from(c.lo)).toEqual([6, 7, 8]);
    expect(Array.from(c.hi)).toEqual([9, 10, 11]);
  });
});
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd web && npx vitest run src/run/traceMode.test.ts`
Expected: FAIL — cannot resolve `./traceMode`.

- [ ] **Step 3: Implement the trace logic**

`web/src/run/traceMode.ts`:

```ts
import type { NDArray } from "../api";

export const FULL_RES_MAX = 20000;

export function traceMode(range: [number, number] | null, T: number): "overview" | "window" {
  const [a, b] = range ?? [0, T];
  return b - a <= FULL_RES_MAX ? "window" : "overview";
}

export function clampRange(x0: number, x1: number, T: number): [number, number] {
  const a = Math.max(0, Math.min(T - 1, Math.floor(Math.min(x0, x1))));
  const b = Math.min(T, Math.max(a + 1, Math.ceil(Math.max(x0, x1))));
  return [a, b];
}

export function lowpassState(
  fps: number | null,
  on: boolean,
  hz: number,
): { enabled: boolean; hz: number | null; max: number; reason?: string } {
  if (!fps) return { enabled: false, hz: null, max: 10, reason: "set the frame rate of the recording first" };
  const max = fps / 2;
  return { enabled: true, hz: on ? Math.min(hz, max) : null, max };
}

export function overviewColumns(ov: NDArray, unit: number): { lo: Float64Array; hi: Float64Array } {
  const bins = ov.shape[2];
  const base = unit * 2 * bins;
  return {
    lo: Float64Array.from(ov.data.subarray(base, base + bins) as ArrayLike<number>),
    hi: Float64Array.from(ov.data.subarray(base + bins, base + 2 * bins) as ArrayLike<number>),
  };
}
```

- [ ] **Step 4: Implement the page**

`web/src/run/pages/Traces.tsx`:

```tsx
import { useEffect, useMemo, useState } from "react";
import { getArray, getJSON, postJSON, type NDArray } from "../../api";
import { Chart, cssVar } from "../../Chart";
import { Loading, Segmented, Slider, Toggle } from "../../controls";
import { fmtClock } from "../../format";
import { useResource } from "../../poll";
import type { Zooms } from "../../types";
import { useRun } from "../RunContext";
import { clampRange, lowpassState, overviewColumns, traceMode } from "../traceMode";

type Kind = "C" | "YrA";

function UnitTrace(p: {
  unit: number;
  kind: Kind;
  overview: NDArray | undefined;
  range: [number, number];
  full: boolean;
  lowpassHz: number | null;
  setRange: (r: [number, number] | null) => void;
}) {
  const { summary: s, t } = useRun();
  const [win, setWin] = useState<NDArray | null>(null);
  const [err, setErr] = useState("");
  const [a, b] = p.range;
  useEffect(() => {
    if (!p.full) return;
    let alive = true;
    const q = `start=${a}&stop=${b}&kind=${p.kind}${p.lowpassHz ? `&lowpass_hz=${p.lowpassHz}` : ""}`;
    getArray(`/runs/${s.id}/traces/${p.unit}?${q}`).then(
      (w) => alive && (setWin(w), setErr("")),
      (e: Error) => alive && setErr(e.message),
    );
    return () => {
      alive = false;
    };
  }, [p.full, a, b, p.kind, p.lowpassHz, p.unit, s.id]);

  const toX = (f: number) => (s.fps ? f / s.fps : f);
  const data = useMemo(() => {
    if (p.full && win) {
      const xs = Float64Array.from({ length: win.shape[0] }, (_, i) => toX(a + i));
      return [xs, Float64Array.from(win.data as ArrayLike<number>)] as [Float64Array, Float64Array];
    }
    if (p.overview) {
      const { lo, hi } = overviewColumns(p.overview, p.unit);
      const bins = lo.length;
      const xs = Float64Array.from({ length: bins }, (_, i) => toX(((i + 0.5) * s.T) / bins));
      return [xs, lo, hi] as [Float64Array, Float64Array, Float64Array];
    }
    return null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.full, win, p.overview, p.unit, a, s.T, s.fps]);

  if (err) return <p className="error" role="alert">Unit {p.unit}: {err}</p>;
  if (!data) return <p className="muted">Loading unit {p.unit}…</p>;
  const band = data.length === 3;
  const stroke = cssVar("--accent");
  return (
    <div>
      <span className="label mono">Unit {p.unit}</span>
      <Chart
        data={data}
        series={band ? [{ stroke, width: 1 }, { stroke, width: 1 }] : [{ stroke, width: 1.2 }]}
        bands={band ? [{ series: [2, 1], fill: "rgba(37, 99, 235, 0.25)" }] : undefined}
        height={120}
        xLabel={s.fps ? "seconds" : "frame"}
        cursorX={toX(t)}
        xRange={[toX(a), toX(b)]}
        onSelectX={(x0, x1) => p.setRange(clampRange(s.fps ? x0 * s.fps : x0, s.fps ? x1 * s.fps : x1, s.T))}
        onResetX={() => p.setRange(null)}
        sync="traces"
      />
    </div>
  );
}

function FpsField() {
  const { summary: s } = useRun();
  const [v, setV] = useState("20");
  const [err, setErr] = useState("");
  return (
    <form
      className="row"
      onSubmit={(e) => {
        e.preventDefault();
        postJSON(`/runs/${s.id}/fps`, { fps: Number(v) }).then(() => window.location.reload(), (x: Error) => setErr(x.message));
      }}
    >
      <label className="field">
        Frame rate of this recording (Hz)
        <input type="number" min={1} max={1000} step="any" value={v} onChange={(e) => setV(e.target.value)} />
      </label>
      <button className="btn" type="submit">Save</button>
      {err && <span className="error">{err}</span>}
    </form>
  );
}

export function Traces() {
  const { summary: s, sel, setSel, setT } = useRun();
  const [kind, setKind] = useState<Kind>("C");
  const [lpOn, setLpOn] = useState(false);
  const [hz, setHz] = useState(5);
  const [range, setRange] = useState<[number, number] | null>(null);
  const zooms = useResource(`zooms:${s.id}`, () => getJSON<Zooms>(`/runs/${s.id}/zooms`));
  const overview = useResource(`ov:${s.id}:${kind}`, () => getArray(`/runs/${s.id}/traces/overview?kind=${kind}&bins=2000`));
  const [extra, setExtra] = useState<number[]>([]);
  const lp = lowpassState(s.fps, lpOn, hz);
  const full = traceMode(range, s.T) === "window";
  const r: [number, number] = range ?? [0, s.T];
  const shown = useMemo(() => {
    const base = sel !== null ? [sel] : (zooms.data?.units.slice(0, 3) ?? []);
    return [...new Set([...base, ...extra])].slice(0, 8);
  }, [sel, zooms.data, extra]);

  return (
    <div className="bento">
      <section className="card span4" aria-label="Trace controls">
        <div className="row">
          <Segmented label="Trace" options={[{ value: "C", label: "C (denoised)" }, { value: "YrA", label: "YrA (raw)" }]} value={kind} onChange={setKind} />
          <Toggle label="Low-pass" checked={lpOn && lp.enabled} disabled={!lp.enabled} title={lp.reason} onChange={setLpOn} />
          <div style={{ minWidth: 240 }}>
            <Slider label="Cutoff" min={0.5} max={lp.max} step={0.5} value={Math.min(hz, lp.max)} disabled={!lp.enabled || !lpOn} onChange={setHz} format={(v) => `${v.toFixed(1)} Hz`} />
          </div>
          <span className="spacer" />
          <button className="btn" type="button" onClick={() => setRange(null)} disabled={!range}>
            Whole recording
          </button>
        </div>
        {!s.fps && <FpsField />}
        <span className="muted">
          {full
            ? `Full resolution${lp.hz ? `, low-pass ${lp.hz.toFixed(1)} Hz` : ""}.`
            : "Overview (min/max per bin); drag across a chart to zoom in to full resolution."}
          {" "}
          {fmtClock(r[0], s.fps)} – {fmtClock(r[1], s.fps)}
        </span>
        <Loading res={zooms} what="zoom suggestions">
          {(z) => (
            <div className="row">
              <span className="label">Jump to</span>
              {z.windows.map((w) => (
                <button key={w.start} className="chip" type="button" onClick={() => (setRange([w.start, w.stop]), setT(w.start))}>
                  {fmtClock(w.start, s.fps)}
                </button>
              ))}
              <span className="label">Add unit</span>
              {z.units.map((u) => (
                <button key={u} className="chip" type="button" aria-pressed={shown.includes(u)} onClick={() => (sel === null ? setSel(u) : setExtra((x) => (x.includes(u) ? x.filter((y) => y !== u) : [...x, u])))}>
                  {u}
                </button>
              ))}
            </div>
          )}
        </Loading>
      </section>
      <section className="card span4" aria-label="Traces">
        {overview.error && <p className="error" role="alert">{overview.error}</p>}
        {overview.computing && !full && <p className="muted">Computing the trace overview… {overview.elapsed.toFixed(0)} s</p>}
        {shown.length === 0 && <p className="muted">Select a unit in the image above.</p>}
        {shown.map((u) => (
          <UnitTrace key={u} unit={u} kind={kind} overview={overview.data} range={r} full={full} lowpassHz={lp.hz} setRange={setRange} />
        ))}
      </section>
    </div>
  );
}
```

Add to `PAGE_VIEWS` in `web/src/run/registry.ts` (with the import):

```ts
import { Traces } from "./pages/Traces";
// …
  traces: Traces,
```

- [ ] **Step 5: Run the checks and look at it**

Run: `cd web && npm run typecheck && npm test`
Expected: no type errors; all tests pass.

Manual check: on a long run, the page first shows min/max bands; dragging across a chart loads the full trace; turning on the low-pass smooths it, and the cutoff slider stops at half the frame rate; "Jump to" chips zoom to the suggested windows; double-click returns to the whole recording.

- [ ] **Step 6: Commit**

```bash
git add web/src
git commit -m "Add the Traces page with overview bands, full-resolution windows and the low-pass"
```

---

### Task 7: Run launcher and job page

**Files:**
- Create: `web/src/launch/ParamsForm.tsx`, `web/src/launch/NewRun.tsx`, `web/src/launch/JobPage.tsx`, `web/src/launch/params.ts`
- Modify: `web/src/App.tsx` (routes `new`, `job`)
- Test: `web/src/launch/params.test.ts`

**Interfaces:**
- Consumes: Tasks 1–2; routes `/params/defaults`, `/raw/info`, `POST /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel`, `GET /jobs/{id}/events` (events `stage`, `log`, `status`), `/runs?root=`.
- Produces (`params.ts`): `parseValue(type: string, raw: string | boolean | [string, string]): unknown` (throws `Error` with a message on bad input), `changedOnly(values, metas): Record<string, unknown>`, `defaultRunName(now: Date): string`, `stageRows(events): { stage: string; seconds: number | null; running: boolean }[]`.
- Produces: `NewRun()`, `JobPage({ id })`.

- [ ] **Step 1: Write the failing test**

`web/src/launch/params.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { changedOnly, defaultRunName, parseValue, stageRows } from "./params";

describe("parameter values", () => {
  it("parses each parameter type", () => {
    expect(parseValue("int", "7")).toBe(7);
    expect(parseValue("float", "0.8")).toBe(0.8);
    expect(parseValue("bool", true)).toBe(true);
    expect(parseValue("float | str", "auto")).toBe("auto");
    expect(parseValue("float | str", "3.5")).toBe(3.5);
    expect(parseValue("tuple[float | None, float | None]", ["25", ""])).toEqual([25, null]);
    expect(parseValue("tuple[float, float]", ["1", "6"])).toEqual([1, 6]);
  });
  it("rejects bad numbers", () => {
    expect(() => parseValue("int", "7.5")).toThrow(/whole number/);
    expect(() => parseValue("float", "abc")).toThrow(/number/);
    expect(() => parseValue("tuple[float, float]", ["1", ""])).toThrow();
  });
  it("sends only values that differ from the defaults", () => {
    const metas = [
      { name: "workers", default: 1 },
      { name: "size_thres", default: [25, null] },
    ];
    expect(changedOnly({ workers: 6, size_thres: [25, null] }, metas)).toEqual({ workers: 6 });
  });
});

describe("launcher helpers", () => {
  it("names runs by date", () => {
    expect(defaultRunName(new Date(2026, 9, 21, 14, 5))).toBe("caanfast_2026-10-21_1405");
  });
  it("turns progress events into stage rows", () => {
    const rows = stageRows([
      { stage: "probe", event: "start", t: 1 },
      { stage: "probe", event: "end", t: 2, seconds: 1.5 },
      { stage: "pass1", event: "start", t: 2 },
    ]);
    expect(rows).toEqual([
      { stage: "probe", seconds: 1.5, running: false },
      { stage: "pass1", seconds: null, running: true },
    ]);
  });
});
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd web && npx vitest run src/launch/params.test.ts`
Expected: FAIL — cannot resolve `./params`.

- [ ] **Step 3: Implement the helpers**

`web/src/launch/params.ts`:

```ts
function num(s: string, what: string): number {
  const v = Number(s.trim());
  if (s.trim() === "" || !Number.isFinite(v)) throw new Error(`${what}: enter a number`);
  return v;
}

/** Value for a ``Params`` field from the form; ``type`` is the dataclass annotation. */
export function parseValue(type: string, raw: string | boolean | [string, string]): unknown {
  if (type === "bool") return Boolean(raw);
  if (type.startsWith("tuple")) {
    const [a, b] = raw as [string, string];
    const nullable = type.includes("None");
    const one = (s: string) => (nullable && s.trim() === "" ? null : num(s, "each value"));
    return [one(a), one(b)];
  }
  const s = String(raw);
  if (type === "int") {
    const v = num(s, "value");
    if (!Number.isInteger(v)) throw new Error("value: enter a whole number");
    return v;
  }
  if (type === "float | str") return s.trim() === "auto" ? "auto" : num(s, "value ('auto' or a number)");
  return num(s, "value");
}

export function changedOnly(
  values: Record<string, unknown>,
  metas: { name: string; default: unknown }[],
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const m of metas) {
    if (m.name in values && JSON.stringify(values[m.name]) !== JSON.stringify(m.default)) out[m.name] = values[m.name];
  }
  return out;
}

export function defaultRunName(now: Date): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `caanfast_${now.getFullYear()}-${p(now.getMonth() + 1)}-${p(now.getDate())}_${p(now.getHours())}${p(now.getMinutes())}`;
}

export type ProgressEvent = { stage: string; event: "start" | "end"; t: number; seconds?: number };

export function stageRows(events: ProgressEvent[]): { stage: string; seconds: number | null; running: boolean }[] {
  const rows = new Map<string, { stage: string; seconds: number | null; running: boolean }>();
  for (const e of events) {
    if (e.event === "start") rows.set(e.stage, { stage: e.stage, seconds: null, running: true });
    else rows.set(e.stage, { stage: e.stage, seconds: e.seconds ?? null, running: false });
  }
  return [...rows.values()];
}
```

- [ ] **Step 4: Implement the form, the launcher and the job page**

`web/src/launch/ParamsForm.tsx`:

```tsx
import { useState } from "react";
import type { ParamMeta } from "../types";
import { parseValue } from "./params";

type Raw = string | boolean | [string, string];

function toRaw(m: ParamMeta, v: unknown): Raw {
  if (m.type === "bool") return Boolean(v);
  if (m.type.startsWith("tuple")) {
    const [a, b] = (v as (number | null)[]) ?? [null, null];
    return [a === null ? "" : String(a), b === null ? "" : String(b)];
  }
  return String(v);
}

function Field({ m, value, onChange }: { m: ParamMeta; value: unknown; onChange: (v: unknown, err: string) => void }) {
  const [raw, setRaw] = useState<Raw>(() => toRaw(m, value));
  const [err, setErr] = useState("");
  const set = (r: Raw) => {
    setRaw(r);
    try {
      onChange(parseValue(m.type, r), "");
      setErr("");
    } catch (e) {
      setErr((e as Error).message);
      onChange(value, (e as Error).message);
    }
  };
  const id = `p-${m.name}`;
  return (
    <div className="field">
      <label htmlFor={id} className="mono">{m.name}</label>
      {m.type === "bool" ? (
        <input id={id} type="checkbox" role="switch" checked={raw as boolean} onChange={(e) => set(e.target.checked)} />
      ) : m.type.startsWith("tuple") ? (
        <div className="row">
          <input id={id} type="text" aria-label={`${m.name} first`} value={(raw as [string, string])[0]} onChange={(e) => set([e.target.value, (raw as [string, string])[1]])} />
          <input type="text" aria-label={`${m.name} second`} value={(raw as [string, string])[1]} onChange={(e) => set([(raw as [string, string])[0], e.target.value])} />
        </div>
      ) : (
        <input id={id} type="text" inputMode="decimal" value={raw as string} onChange={(e) => set(e.target.value)} />
      )}
      <span className="muted">{m.help}</span>
      {err && <span className="error">{err}</span>}
    </div>
  );
}

export function ParamsForm(p: {
  metas: ParamMeta[];
  groups: string[];
  values: Record<string, unknown>;
  onChange: (name: string, v: unknown, err: string) => void;
}) {
  const common = p.metas.filter((m) => m.common);
  return (
    <>
      <div className="bento" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
        {common.map((m) => (
          <Field key={m.name} m={m} value={p.values[m.name]} onChange={(v, e) => p.onChange(m.name, v, e)} />
        ))}
      </div>
      <details>
        <summary className="label" style={{ cursor: "pointer" }}>All parameters</summary>
        {p.groups.map((g) => (
          <fieldset key={g} style={{ border: 0, padding: 0, marginTop: 16 }}>
            <legend className="label">{g}</legend>
            <div className="bento" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
              {p.metas.filter((m) => m.group === g && !m.common).map((m) => (
                <Field key={m.name} m={m} value={p.values[m.name]} onChange={(v, e) => p.onChange(m.name, v, e)} />
              ))}
            </div>
          </fieldset>
        ))}
      </details>
    </>
  );
}
```

`web/src/launch/NewRun.tsx`:

```tsx
import { useMemo, useState } from "react";
import { getJSON, postJSON } from "../api";
import { ErrorNote, Loading, Toggle } from "../controls";
import { fmtBytes, fmtDuration, fmtInt } from "../format";
import { FolderPicker } from "../home/FolderPicker";
import { useResource } from "../poll";
import { go } from "../router";
import type { Job, ParamMeta, RawInfo } from "../types";
import { changedOnly, defaultRunName } from "./params";
import { ParamsForm } from "./ParamsForm";

export function NewRun() {
  const defaults = useResource("params", () => getJSON<{ groups: string[]; params: ParamMeta[] }>("/params/defaults"));
  const [raw, setRaw] = useState("");
  const [picking, setPicking] = useState(true);
  const info = useResource(raw ? `raw:${raw}` : null, () => getJSON<RawInfo>(`/raw/info?path=${encodeURIComponent(raw)}`));
  const [name, setName] = useState(() => defaultRunName(new Date()));
  const [values, setValues] = useState<Record<string, unknown>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [keep, setKeep] = useState(false);
  const [overwrite, setOverwrite] = useState(false);
  const [submitErr, setSubmitErr] = useState("");
  const [busy, setBusy] = useState(false);

  const merged = useMemo(() => {
    const d: Record<string, unknown> = {};
    for (const m of defaults.data?.params ?? []) d[m.name] = m.default;
    return { ...d, ...values };
  }, [defaults.data, values]);
  const invalid = Object.values(errors).some(Boolean);
  const outDir = raw ? `${raw.replace(/\/+$/, "").split("/").slice(0, -1).join("/")}/processed/${name}` : "";

  const start = () => {
    setBusy(true);
    setSubmitErr("");
    postJSON<Job>("/jobs", {
      raw_dir: raw,
      name,
      params: changedOnly(merged, defaults.data!.params),
      keep_store: keep,
      overwrite,
    }).then(
      (job) => go({ name: "job", id: job.id }),
      (e: Error) => {
        setSubmitErr(e.message);
        setBusy(false);
      },
    );
  };

  return (
    <div className="page">
      <h1>New run</h1>
      <div className="bento">
        <section className="card span4" aria-label="1. Raw videos">
          <span className="label">1 · Folder with the raw videos (0.avi, 1.avi, …)</span>
          <div className="row">
            <span className="mono">{raw || "none chosen"}</span>
            <span className="spacer" />
            <button className="btn" type="button" onClick={() => setPicking(true)}>Choose folder</button>
          </div>
          {raw && (
            <Loading res={info} what="the videos">
              {(i) => (
                <span className="muted">
                  {i.segments} files · {fmtInt(i.frames)} frames · {i.width} × {i.height}
                  {i.fps ? ` · ${i.fps} fps · ${fmtDuration(i.frames / i.fps)}` : " · frame rate unknown"} · {fmtBytes(i.bytes)}
                </span>
              )}
            </Loading>
          )}
        </section>
        {picking && (
          <FolderPicker
            start={raw}
            pickLabel="Use these videos"
            accept={(e) => e.has_avi}
            onPick={(p) => {
              setRaw(p);
              setPicking(false);
            }}
            onClose={() => setPicking(false)}
          />
        )}
        <section className="card span4" aria-label="2. Output">
          <span className="label">2 · Name of the run</span>
          <label className="field">
            Run name
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          {outDir && <span className="muted mono">{outDir}</span>}
          <Toggle label="Keep the preprocessed tile store (as big as the raw videos; needed for the Preprocessed view)" checked={keep} onChange={setKeep} />
        </section>
        <section className="card span4" aria-label="3. Parameters">
          <span className="label">3 · Parameters</span>
          <Loading res={defaults} what="parameters">
            {(d) => (
              <ParamsForm
                metas={d.params}
                groups={d.groups}
                values={merged}
                onChange={(n, v, e) => {
                  setValues((x) => ({ ...x, [n]: v }));
                  setErrors((x) => ({ ...x, [n]: e }));
                }}
              />
            )}
          </Loading>
        </section>
      </div>
      {submitErr && <ErrorNote message={submitErr} />}
      {submitErr.includes("already holds a run") && <Toggle label="Overwrite the existing run" checked={overwrite} onChange={setOverwrite} />}
      <div className="row">
        <span className="spacer" />
        <button className="btn primary" type="button" disabled={!raw || !info.data || invalid || busy || !defaults.data} onClick={start}>
          Start run
        </button>
      </div>
    </div>
  );
}
```

`web/src/launch/JobPage.tsx`:

```tsx
import { useEffect, useState } from "react";
import { getJSON, postJSON, tokenQuery } from "../api";
import { ErrorNote } from "../controls";
import { fmtDuration, stageLabel } from "../format";
import { go } from "../router";
import type { Job, RunListing } from "../types";
import { stageRows, type ProgressEvent } from "./params";

export function JobPage({ id }: { id: string }) {
  const [job, setJob] = useState<Job | null>(null);
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [log, setLog] = useState<string[]>([]);
  const [err, setErr] = useState("");
  const [now, setNow] = useState(Date.now() / 1000);

  useEffect(() => {
    getJSON<Job>(`/jobs/${id}`).then(setJob, (e: Error) => setErr(e.message));
    const es = new EventSource(`/api/jobs/${id}/events?${tokenQuery()}`);
    es.addEventListener("stage", (e) => setEvents((x) => [...x, JSON.parse((e as MessageEvent).data)]));
    es.addEventListener("log", (e) => setLog((x) => [...x.slice(-299), JSON.parse((e as MessageEvent).data)]));
    es.addEventListener("status", (e) => setJob(JSON.parse((e as MessageEvent).data)));
    es.onerror = () => es.close();
    const tick = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => {
      es.close();
      clearInterval(tick);
    };
  }, [id]);

  const openRun = () =>
    getJSON<{ runs: RunListing[] }>(`/runs?root=${encodeURIComponent(job!.out_dir)}`).then(
      (r) => go({ name: "run", id: r.runs[0].id, page: "overview" }),
      (e: Error) => setErr(e.message),
    );

  if (err) return <div className="page"><ErrorNote message={err} /></div>;
  if (!job) return <div className="page"><p className="muted">Loading the job…</p></div>;
  const end = job.ended ?? now;
  return (
    <div className="page">
      <div className="row">
        <div>
          <span className="label">Run · {job.status}</span>
          <h1>{job.name}</h1>
        </div>
        <span className="spacer" />
        {job.started && <span className="big" style={{ fontSize: 28 }}>{fmtDuration(end - job.started)}</span>}
        {(job.status === "queued" || job.status === "running") && (
          <button className="btn" type="button" onClick={() => postJSON<Job>(`/jobs/${id}/cancel`, {}).then(setJob)}>Cancel</button>
        )}
        {job.status === "done" && <button className="btn primary" type="button" onClick={openRun}>Open run</button>}
      </div>
      <div className="bento">
        <section className="card span2" aria-label="Stages">
          <span className="label">Stages</span>
          {stageRows(events).map((r) => (
            <div className="barrow" key={r.stage}>
              <span>{r.running ? "…" : "✓"} {stageLabel(r.stage)}</span>
              <span className="spacer" />
              <span className="mono">{r.seconds === null ? "" : fmtDuration(r.seconds)}</span>
            </div>
          ))}
          {events.length === 0 && <span className="muted">{job.status === "queued" ? "Waiting for the run before it." : "Starting…"}</span>}
        </section>
        <section className="card span2" aria-label="Log">
          <span className="label">Log</span>
          <pre className="log">{log.slice(-40).join("\n")}</pre>
        </section>
      </div>
      {job.status === "failed" && <ErrorNote message={job.error || "the run failed"} />}
      {job.status === "interrupted" && <ErrorNote message={job.error} />}
      <span className="muted mono">{job.out_dir}</span>
    </div>
  );
}
```

In `web/src/App.tsx` add the imports `import { JobPage } from "./launch/JobPage";` and `import { NewRun } from "./launch/NewRun";` and the routes:

```tsx
      {route.name === "new" && <NewRun />}
      {route.name === "job" && <JobPage key={route.id} id={route.id} />}
```

- [ ] **Step 5: Run the checks and try a run**

Run: `cd web && npm run typecheck && npm test`
Expected: no type errors; all tests pass.

Manual check: `caanfast synth /tmp/caanfast-demo/raw --minutes 0.2`, then in the app: New run → choose `/tmp/caanfast-demo/raw` → the info line shows 1 file, 360 frames, 600 × 600, 30 fps → set `tile` to 64 → Start run. Expected: the job page ticks off the stages live, the log fills, "Open run" opens it. Starting again with the same name shows the "already holds a run" error and the overwrite switch.

- [ ] **Step 6: Commit**

```bash
git add web/src
git commit -m "Add the run launcher and the live job page"
```

---

### Task 8: Ship the front end in the wheel, smoke test, docs

**Files:**
- Modify: `pyproject.toml`, `README.md`
- Create: `web/playwright.config.ts`, `web/e2e/make_run.py`, `web/e2e/smoke.spec.ts`

**Interfaces:**
- Consumes: everything above and `caanfast serve --token` (back-end Task 11).
- Produces: a wheel that contains `caanfast/serve/static/index.html`; `npm run e2e`.

- [ ] **Step 1: Ship the build output**

In `pyproject.toml` replace the wheel section with:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/caanfast"]
artifacts = ["src/caanfast/serve/static/**"]
```

Run:

```bash
cd web && npm run build && cd .. && uv build --wheel && unzip -l dist/caanfast-0.0.1-py3-none-any.whl | grep serve/static/index.html
```

Expected: the last command prints one line with `caanfast/serve/static/index.html`.

- [ ] **Step 2: Write the smoke test**

`web/e2e/make_run.py`:

```python
"""Make the tiny run the smoke test serves (once; reused while web/e2e/.data exists)."""

import sys
from pathlib import Path

from caanfast import pipeline
from caanfast.params import Params
from caanfast.testing import synth

root = Path(sys.argv[1])
if not (root / "processed" / "t" / "report.json").exists():
    raw = root / "raw"
    synth.write_session(raw, minutes=300 / 30 / 60, fps=30, segment=100, seed=0, side=64)
    params = Params(
        tile=32,
        seeds_wnd=100,
        seeds_stp=50,
        seeds_max_wnd=8,
        template_frames=50,
        inita_wnd=8,
        dl_wnd=5,
        size_thres=(9, None),
        svd_k0=16,
        motion_before_tophat=False,
    )
    pipeline.run(raw, params, run_name="t", workers=1, keep_store=True)
```

`web/playwright.config.ts`:

```ts
import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const py = process.env.CAANFAST_PYTHON ?? resolve(here, "../.venv/bin/python");
const data = resolve(here, "e2e/.data");

export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  use: { baseURL: "http://127.0.0.1:8799" },
  webServer: {
    command: `"${py}" e2e/make_run.py "${data}" && CAANFAST_CACHE="${data}/cache" "${py}" -m caanfast serve "${data}" --no-browser --port 8799 --token e2e`,
    url: "http://127.0.0.1:8799/",
    timeout: 300_000,
    reuseExistingServer: false,
  },
});
```

`web/e2e/smoke.spec.ts`:

```ts
import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL(".", import.meta.url)), ".data");
const pages = ["Overview", "Frames", "Motion", "Compression", "Seeds", "Units", "Traces"];

test("opens a run and every stage page without errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`/?token=e2e&root=${encodeURIComponent(root)}`);
  await page.getByRole("link", { name: "Open run t" }).click();
  await expect(page.getByRole("heading", { name: "t", exact: true })).toBeVisible();
  await expect(page.getByRole("img", { name: /frame with unit outlines/ })).toBeVisible();
  for (const name of pages) {
    await page.getByRole("group", { name: "Stage" }).getByRole("button", { name }).click();
    await expect(page.getByRole("alert")).toHaveCount(0, { timeout: 60_000 });
  }
  await page.getByRole("group", { name: "Image" }).getByRole("button", { name: "Reconstruction" }).click();
  await page.getByRole("slider", { name: "Scrub recording" }).fill("150");
  await expect(page.getByText(/0:00:05/)).toBeVisible();
  expect(errors).toEqual([]);
});

test("the launcher lists the parameters", async ({ page }) => {
  await page.goto(`/?token=e2e#/new`);
  await expect(page.getByLabel("workers")).toBeVisible();
  await expect(page.getByRole("button", { name: "Start run" })).toBeDisabled();
});
```

- [ ] **Step 3: Run it**

Run:

```bash
cd web && npx playwright install chromium && npm run build && npm run e2e
```

Expected: 2 passed (the first start builds the tiny run, ~1 min).

- [ ] **Step 4: Document the front end**

Add to `README.md`, under "Browser viewer":

```markdown
### Building the front end

The browser app lives in `web/` (React + TypeScript, Vite). `caanfast serve` serves the
build in `src/caanfast/serve/static/`, which is not in git:

    cd web && npm install && npm run build

During development run the back end and Vite's dev server side by side:

    caanfast serve FOLDER --no-browser --port 8765 --token dev
    cd web && npm run dev      # then open http://localhost:5173/?token=dev

Checks: `npm run typecheck`, `npm test` (unit tests), `npm run e2e` (smoke test in Chromium
against a tiny synthetic run; set `CAANFAST_PYTHON` if the venv is not `../.venv`).
```

- [ ] **Step 5: Final checks**

Run: `.venv/bin/pytest -q && (cd web && npm run typecheck && npm test && npm run build)`
Expected: everything passes.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md web/playwright.config.ts web/e2e/make_run.py web/e2e/smoke.spec.ts
git commit -m "Ship the built front end in the wheel and add a browser smoke test"
```
