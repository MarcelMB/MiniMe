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
  // kept outside Loading so a folder that fails to load (privacy-protected) can still be left
  const shown = res.data?.path ?? path;
  const parent = res.data?.parent ?? (shown ? shown.replace(/[\\/][^\\/]*[\\/]?$/, "") || "/" : undefined);
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
      {parent !== undefined && (
        <button className="btn" type="button" onClick={() => setPath(parent)}>
          ↑ {parent}
        </button>
      )}
      <Loading res={res} what="folder">
        {(fs) => (
          <div className="picker">
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
