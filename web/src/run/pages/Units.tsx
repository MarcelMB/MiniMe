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
    if (x === null && y === null) return a.id - b.id;
    if (x === null) return 1;
    if (y === null) return -1;
    if (x === y) return a.id - b.id;
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
  const onSort = (k: SortKey) => {
    if (sortKey === k) setDesc(!desc);
    else {
      setSortKey(k);
      setDesc(true);
    }
  };
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
                    <button type="button" onClick={() => onSort(c.k)}>
                      {c.label}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((u) => (
                <tr
                  key={u.id}
                  className="clickable"
                  aria-current={u.id === sel ? "true" : undefined}
                  onClick={() => setSel(u.id)}
                >
                  <td className="mono">
                    <button type="button" aria-pressed={u.id === sel} onClick={() => setSel(u.id)}>
                      {u.id}
                    </button>
                  </td>
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
