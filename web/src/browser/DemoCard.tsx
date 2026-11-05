import { useState } from "react";
import { Segmented } from "../controls";
import { fmtBytes } from "../format";
import { go } from "../router";
import { approxBytes, DEMO_PAGE, DEMOS, type DemoKind } from "./demo";
import { FileListDir, HandleDir } from "./dir";
import { state } from "./state";
import { pickDirectory, support } from "./support";
import { splitVpath } from "./vpath";

const KINDS: DemoKind[] = ["short", "full"];
const label = (k: DemoKind) => `${DEMOS[k].minutes} min · ${DEMOS[k].segments} videos · ${fmtBytes(approxBytes(k))}`;
const last = (k: DemoKind) => DEMOS[k].segments - 1;

/** Open New run on the folder `vpath` (a root or a folder in one). */
async function openNewRun(vpath: string): Promise<void> {
  await state.folders.touch(splitVpath(vpath).root, vpath);
  go({ name: "new" });
}

/** Home's demo card (browser build): what to download of the synthetic demo session on UCLA
 * Dataverse (by hand: Dataverse's bot check keeps this page from fetching it, see `demo.ts`), then
 * a button that opens New run on the folder with the videos. */
export function DemoCard() {
  const [kind, setKind] = useState<DemoKind>("short");
  const [err, setErr] = useState("");
  const can = support();
  const d = DEMOS[kind];

  const pick = async () => {
    setErr("");
    try {
      // the dialog first: it needs the click's user activation
      await openNewRun(await state.folders.pick(new HandleDir(await pickDirectory("minime"))));
    } catch (e) {
      if ((e as Error).name !== "AbortError") setErr((e as Error).message);
    }
  };
  const upload = (files: FileList | null) => {
    if (!files?.length) return;
    setErr("");
    state.folders
      .add(FileListDir.fromFiles([...files]), "files")
      .then(openNewRun)
      .catch((e: Error) => setErr(e.message));
  };

  return (
    <section className="card demo" aria-label="Try the demo">
      <div className="row" style={{ flexWrap: "wrap" }}>
        <h2 className="run-name">Try it on a demo recording</h2>
        <span className="spacer" />
        <Segmented
          label="Demo length"
          options={KINDS.map((k) => ({ value: k, label: label(k) }))}
          value={kind}
          onChange={setKind}
        />
      </div>
      <p className="muted" style={{ margin: 0 }}>
        A synthetic one-photon miniscope session (600 × 600 px, 20 fps, through a GRIN lens) with known ground truth,
        free to use (CC0). It comes as 36 videos of 50 s each; the first six make a 5 min demo.
      </p>
      <ol className="demo-steps">
        <li>
          Open the{" "}
          <a href={DEMO_PAGE} target="_blank" rel="noreferrer">
            demo dataset on UCLA Dataverse
          </a>{" "}
          and download {kind === "short" ? "the videos " : "all the videos, "}
          <span className="mono">0.avi</span> to <span className="mono">{last(kind)}.avi</span> ({d.segments} files,
          about {fmtBytes(approxBytes(kind))}).
        </li>
        <li>
          Put them in one folder and keep their names (<span className="mono">0.avi</span>,{" "}
          <span className="mono">1.avi</span>, …): MiniMe reads the videos in the order of that number. For the frame
          rate, put the dataset's <span className="mono">meta.json</span> in the folder above, or type 20 fps in New
          run.
        </li>
        <li>Choose that folder: New run opens on it.</li>
      </ol>
      <div className="row" style={{ flexWrap: "wrap" }}>
        {can.fsAccess ? (
          <button className="btn primary" type="button" onClick={pick}>
            Choose the demo folder…
          </button>
        ) : (
          <label className="btn primary">
            Choose the demo folder…
            <input type="file" hidden onChange={(e) => upload(e.target.files)} {...{ webkitdirectory: "" }} />
          </label>
        )}
      </div>
      {err && <p className="error">{err}</p>}
    </section>
  );
}
