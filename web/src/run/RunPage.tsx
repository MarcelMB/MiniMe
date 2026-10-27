import { useMemo, useState } from "react";
import { getArray, getJSON } from "../api";
import { ComputingNote, ErrorNote, Segmented } from "../controls";
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
  const state = useMemo<RunState | null>(
    () =>
      summary.data && units.data && maxImg
        ? { summary: summary.data, units: units.data.units, maxProj: maxImg, t, setT, sel, setSel, cache }
        : null,
    [summary.data, units.data, maxImg, t, sel, cache],
  );

  if (summary.error) return <div className="page"><ErrorNote message={summary.error} /></div>;
  if (units.error) return <div className="page"><ErrorNote message={units.error} /></div>;
  if (maxProj.error) return <div className="page"><ErrorNote message={maxProj.error} /></div>;
  if (!state) {
    return (
      <div className="page">
        {maxProj.computing ? (
          <ComputingNote elapsed={maxProj.elapsed} what="the max projection" />
        ) : (
          <p className="muted">Loading the run…</p>
        )}
      </div>
    );
  }
  const s = state.summary;
  const View = PAGE_VIEWS[page];
  const pages = PAGES.filter((p) => PAGE_VIEWS[p]);
  return (
    <RunCtx.Provider value={state}>
      <div className="page">
        <div className="row">
          <div>
            <span className="label">MiniMe · run</span>
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
