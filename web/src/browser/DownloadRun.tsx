import { useEffect, useState } from "react";
import { runZip } from "./download";
import { OPFS_NAME } from "./folders";
import { state } from "./state";

/** "Download run (.zip)" for a run in Browser storage (a run in a folder on disk already is one). */
export function DownloadRun({ runId }: { runId: string }) {
  const [vpath, setVpath] = useState<string | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    state.registry.get(runId).then(
      (r) => setVpath(r.vpath),
      () => setVpath(null),
    );
  }, [runId]);
  if (!vpath?.startsWith(`${OPFS_NAME}/`)) return null;
  const save = async () => {
    setErr("");
    try {
      const url = URL.createObjectURL(await runZip(await state.registry.get(runId)));
      Object.assign(document.createElement("a"), { href: url, download: `${vpath.split("/").pop()}.zip` }).click();
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <>
      <button className="btn" type="button" onClick={save}>
        Download run (.zip)
      </button>
      {err && <span className="error">{err}</span>}
    </>
  );
}
