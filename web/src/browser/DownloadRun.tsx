import { useEffect, useState } from "react";
import { runZip } from "./download";
import { OPFS_NAME } from "./folders";
import { state } from "./state";

/** "Download run (.zip)" for a run in Browser storage (a run in a folder on disk already is one).
 * The zip is put together before the download starts (every file is read for its checksum: tens
 * of seconds for a long session), so the button says so and takes no second click meanwhile. */
export function DownloadRun({ runId }: { runId: string }) {
  const [vpath, setVpath] = useState<string | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    state.registry.get(runId).then(
      (r) => setVpath(r.vpath),
      () => setVpath(null),
    );
  }, [runId]);
  if (!vpath?.startsWith(`${OPFS_NAME}/`)) return null;
  const save = async () => {
    if (busy) return;
    setBusy(true);
    setErr("");
    try {
      const url = URL.createObjectURL(await runZip(await state.registry.get(runId)));
      Object.assign(document.createElement("a"), { href: url, download: `${vpath.split("/").pop()}.zip` }).click();
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (e) {
      setErr(`could not make the zip: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <button className="btn" type="button" onClick={save} disabled={busy} aria-busy={busy}>
        {busy ? "Preparing…" : "Download run (.zip)"}
      </button>
      {err && <span className="error">{err}</span>}
    </>
  );
}
