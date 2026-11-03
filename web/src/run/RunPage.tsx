import { useMemo, useRef, useState } from "react";
import { getArray, getJSON } from "../api";
import { ComputingNote, ErrorNote, Segmented } from "../controls";
import { imgFrom } from "../draw";
import { FrameCache } from "../frames";
import { fmtDuration, fmtInt } from "../format";
import { useResource } from "../poll";
import { go, PAGES, type Page } from "../router";
import type { Rerun, Summary, UnitsResponse } from "../types";
import { PAGE_LABELS, PAGE_VIEWS } from "./registry";
import { RunCtx, rerunQuery, type RunState } from "./RunContext";
import { TopArea } from "./TopArea";

export function RunPage({ id, page }: { id: string; page: Page }) {
  const summary = useResource(`summary:${id}`, () => getJSON<Summary>(`/runs/${id}/summary`));
  const [rerun, setRerunState] = useState<string | null>(null);
  const [gen, setGen] = useState(0);
  const reruns = useResource(`reruns:${id}:${gen}`, () => getJSON<{ reruns: Rerun[] }>(`/runs/${id}/reruns`));
  const list = useMemo(() => reruns.data?.reruns ?? [], [reruns.data]);
  const units = useResource(`units:${id}:${rerun ?? ""}`, () =>
    getJSON<UnitsResponse>(`/runs/${id}/units${rerunQuery(rerun)}`),
  );
  // the last loaded units stay on screen while a rerun's load (the top area keeps its state)
  const lastUnits = useRef<UnitsResponse["units"] | null>(null);
  if (units.data) lastUnits.current = units.data.units;
  const unitList = units.data?.units ?? lastUnits.current;
  const maxProj = useResource(`maxproj:${id}`, () => getArray(`/runs/${id}/image/max_proj`));
  const [t, setT] = useState(0);
  const [sel, setSel] = useState<number | null>(null);
  const setRerun = (name: string | null) => {
    setSel(null); // unit ids differ between reruns
    setRerunState(name);
  };
  const cache = useMemo(
    () => new FrameCache((source, f) => getArray(`/runs/${id}/frame/${f}?source=${source}`)),
    [id],
  );
  const maxImg = useMemo(() => (maxProj.data ? imgFrom(maxProj.data) : null), [maxProj.data]);
  const state = useMemo<RunState | null>(
    () =>
      summary.data && unitList && maxImg
        ? {
            summary: summary.data,
            units: unitList,
            maxProj: maxImg,
            t,
            setT,
            sel,
            setSel,
            cache,
            rerun,
            setRerun,
            reruns: list,
            refreshReruns: () => setGen((g) => g + 1),
          }
        : null,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [summary.data, unitList, maxImg, t, sel, cache, rerun, list],
  );

  if (summary.error) return <div className="page"><ErrorNote message={summary.error} /></div>;
  if (units.error && !rerun) return <div className="page"><ErrorNote message={units.error} /></div>;
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
        {units.error && <ErrorNote message={units.error} />}
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
