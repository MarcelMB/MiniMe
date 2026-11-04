import { type ComponentProps, useState } from "react";
import type { FolderPicker as MainFolderPicker } from "../home/FolderPicker";
import { FileListDir, HandleDir } from "./dir";
import { state } from "./state";
import { pickDirectory, support } from "./support";

/** The browser's folder chooser: the system dialog (Chrome, Edge) or a read-only folder upload
 * (other browsers), in main's card layout and with the props of main's `home/FolderPicker`
 * (there is nothing to browse: `start` and `accept` have no use). */
export function FolderPicker(p: ComponentProps<typeof MainFolderPicker>) {
  const [err, setErr] = useState("");
  const pick = async () => {
    setErr("");
    try {
      // asks for write access in the dialog itself: a prompt later would have no user activation;
      // a folder that is a root already gets its permission back instead of a second root, and a
      // folder inside a root is its path there (a raw folder's run then goes to <parent>/processed)
      p.onPick(await state.folders.pick(new HandleDir(await pickDirectory("minime"))));
    } catch (e) {
      if ((e as Error).name !== "AbortError") setErr((e as Error).message);
    }
  };
  const upload = (files: FileList | null) => {
    if (!files?.length) return;
    setErr("");
    state.folders.add(FileListDir.fromFiles([...files]), "files").then(p.onPick, (e: Error) => setErr(e.message));
  };
  const canPick = support().fsAccess;
  return (
    <section className="card span4" aria-label="Choose a folder">
      <div className="row">
        <span className="label">Folder</span>
        <span className="spacer" />
        <button className="btn" type="button" onClick={p.onClose}>
          Close
        </button>
        {canPick ? (
          <button className="btn primary" type="button" onClick={pick}>
            {p.pickLabel}
          </button>
        ) : (
          <label className="btn primary">
            {p.pickLabel}
            <input type="file" hidden onChange={(e) => upload(e.target.files)} {...{ webkitdirectory: "" }} />
          </label>
        )}
      </div>
      {!canPick && <p className="muted">This browser opens folders read-only: runs go to Browser storage.</p>}
      {err && <p className="error">{err}</p>}
    </section>
  );
}
