import { useRef, useState } from "react";
import { Segmented } from "../controls";
import { fmtBytes } from "../format";
import { offerPreset } from "../launch/preset";
import { go } from "../router";
import {
  approxBytes,
  DEMO_PAGE,
  DEMOS,
  type DemoKind,
  downloadDemo,
  HF_PAGE,
  handleTarget,
  type Progress,
} from "./demo";
import { DEMO_PRESET } from "./demoPreset";
import { FileListDir, HandleDir } from "./dir";
import { OPFS_NAME } from "./folders";
import { state } from "./state";
import { pickDirectory, support } from "./support";
import { joinVpath, splitVpath } from "./vpath";

const KINDS: DemoKind[] = ["short", "full"];
const label = (k: DemoKind) => `${DEMOS[k].minutes} min · ${DEMOS[k].segments} videos · ${fmtBytes(approxBytes(k))}`;
const last = (k: DemoKind) => DEMOS[k].segments - 1;

/** Open New run with the demo settings (on the folder last used). */
function openWithDemoSettings(): void {
  offerPreset(DEMO_PRESET);
  go({ name: "new" });
}

/** Open New run on the folder `vpath` (a root or a folder in one), with the demo settings. */
async function openNewRun(vpath: string): Promise<void> {
  await state.folders.touch(splitVpath(vpath).root, vpath);
  openWithDemoSettings();
}

/** Refuse a download into browser storage that would not fit (`need` bytes still to fetch). */
async function checkRoom(need: number): Promise<void> {
  const { quota, usage } = (await navigator.storage?.estimate?.().catch(() => undefined)) ?? {};
  if (quota === undefined || usage === undefined || need <= quota - usage) return;
  throw new Error(
    `the demo needs ${fmtBytes(need)} of browser storage, ${fmtBytes(Math.max(0, quota - usage))} are free; choose a folder instead`,
  );
}

/** Home's demo card (browser build): download the synthetic demo session from its Hugging Face
 * mirror into a folder or browser storage, then open New run on its videos with the demo settings.
 * The manual route (UCLA Dataverse, by hand: its bot check keeps this page from fetching it, see
 * `demo.ts`) stays below, with a button that opens New run on a folder downloaded that way. */
export function DemoCard() {
  const [kind, setKind] = useState<DemoKind>("short");
  const [err, setErr] = useState("");
  const [progress, setProgress] = useState<Progress | null>(null);
  const abort = useRef<AbortController | null>(null);
  const can = support();
  const d = DEMOS[kind];

  /** Download into `root` (the folder `rootVpath` names) and open New run on its `raw/`. */
  const download = async (root: FileSystemDirectoryHandle, rootVpath: string, inBrowser: boolean) => {
    const ac = new AbortController();
    abort.current = ac;
    setProgress({ done: 0, total: approxBytes(kind), index: 0, count: 0, file: "", skipped: 0 });
    try {
      await downloadDemo(kind, handleTarget(root), {
        signal: ac.signal,
        onProgress: setProgress,
        beforeStart: inBrowser ? checkRoom : undefined,
      });
      await openNewRun(joinVpath(rootVpath, d.folder, "raw"));
    } finally {
      abort.current = null;
      setProgress(null);
    }
  };
  const run = (start: () => Promise<void>) => async () => {
    setErr("");
    try {
      await start();
    } catch (e) {
      if ((e as Error).name !== "AbortError") setErr((e as Error).message);
    }
  };
  const toFolder = run(async () => {
    // the dialog first: it needs the click's user activation
    const h = await pickDirectory("minime");
    await download(h, await state.folders.pick(new HandleDir(h)), false);
  });
  const toBrowser = run(async () => {
    const h = await (await navigator.storage.getDirectory()).getDirectoryHandle("runs", { create: true });
    await download(h, OPFS_NAME, true);
  });

  const pick = run(async () => {
    await openNewRun(await state.folders.pick(new HandleDir(await pickDirectory("minime"))));
  });
  const upload = (files: FileList | null) => {
    if (!files?.length) return;
    setErr("");
    state.folders
      .add(FileListDir.fromFiles([...files]), "files")
      .then(openNewRun)
      .catch((e: Error) => setErr(e.message));
  };

  const busy = progress !== null;
  return (
    <section className="card demo" aria-label="Try the demo">
      <div className="row" style={{ flexWrap: "wrap" }}>
        <h2 className="run-name">Try it on a demo recording</h2>
        <span className="spacer" />
        <Segmented
          label="Demo length"
          options={KINDS.map((k) => ({ value: k, label: label(k) }))}
          value={kind}
          onChange={(k) => !busy && setKind(k)}
        />
      </div>
      <p className="muted" style={{ margin: 0 }}>
        A synthetic one-photon miniscope session (600 × 600 px, 20 fps, through a GRIN lens) with known ground truth,
        free to use (CC0). It comes as 36 videos of 50 s each; the first six make a 5 min demo. The download goes into a{" "}
        <span className="mono">{d.folder}</span> folder; a download cut short goes on where it stopped. New run then
        opens on it with the demo settings, the parameters Python's auto-tune found for this session (Reset to defaults
        there to go back).
      </p>
      {busy ? (
        <div className="barrow">
          <span className="mono">
            {progress.count ? `${progress.file} · ${progress.index} of ${progress.count}` : "Listing the files…"}
          </span>
          <div className="bar">
            <span style={{ width: `${(100 * progress.done) / Math.max(1, progress.total)}%` }} />
          </div>
          <span className="mono">
            {fmtBytes(progress.done)} of {fmtBytes(progress.total)}
          </span>
          <button className="btn" type="button" onClick={() => abort.current?.abort()}>
            Cancel
          </button>
        </div>
      ) : (
        <div className="row" style={{ flexWrap: "wrap" }}>
          {can.fsAccess && (
            <button className="btn primary" type="button" onClick={toFolder}>
              Download into a folder…
            </button>
          )}
          {can.opfs && (
            <button className={can.fsAccess ? "btn" : "btn primary"} type="button" onClick={toBrowser}>
              Download into browser storage
            </button>
          )}
          <button className="btn" type="button" onClick={openWithDemoSettings}>
            Use the demo settings
          </button>
        </div>
      )}
      {err && <p className="error">{err}</p>}
      <details>
        <summary className="muted">Already downloaded it, or want to by hand?</summary>
        <ol className="demo-steps">
          <li>
            Download {kind === "short" ? "the videos " : "all the videos, "}
            <span className="mono">0.avi</span> to <span className="mono">{last(kind)}.avi</span> ({d.segments} files,
            about {fmtBytes(approxBytes(kind))}) from{" "}
            <a href={DEMO_PAGE} target="_blank" rel="noreferrer">
              UCLA Dataverse
            </a>{" "}
            (the dataset's citable copy) or{" "}
            <a href={`${HF_PAGE}/tree/main/raw`} target="_blank" rel="noreferrer">
              Hugging Face
            </a>
            .
          </li>
          <li>
            Put them in one folder and keep their names (<span className="mono">0.avi</span>,{" "}
            <span className="mono">1.avi</span>, …): MiniMe reads the videos in the order of that number. For the frame
            rate, put the dataset's <span className="mono">meta.json</span> in the folder above, or type 20 fps in New
            run.
          </li>
          <li>Choose that folder: New run opens on it with the demo settings.</li>
        </ol>
        <div className="row" style={{ flexWrap: "wrap", marginTop: 8 }}>
          {can.fsAccess ? (
            <button className="btn" type="button" onClick={pick} disabled={busy}>
              Choose the demo folder…
            </button>
          ) : (
            <label className="btn">
              Choose the demo folder…
              <input type="file" hidden onChange={(e) => upload(e.target.files)} {...{ webkitdirectory: "" }} />
            </label>
          )}
        </div>
      </details>
    </section>
  );
}
