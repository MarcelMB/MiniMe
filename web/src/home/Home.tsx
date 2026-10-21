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
