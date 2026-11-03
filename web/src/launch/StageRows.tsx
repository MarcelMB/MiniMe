import { fmtDuration, stageLabel } from "../format";
import type { StageRow } from "./params";

/** One row per stage of a run: state mark, name, an optional note and the seconds it took
 * (`minime serve`'s job page and the browser run share it). `final`: the run will not
 * change any more, so a row still running was stopped. */
export function StageRows({ rows, final }: { rows: StageRow[]; final: boolean }) {
  return (
    <>
      {rows.map((r) => (
        <div className="barrow" key={r.stage}>
          <span>
            {r.running && !final ? (
              "…"
            ) : r.running ? (
              <span role="img" aria-label="stopped">
                –
              </span>
            ) : r.failed ? (
              <span role="img" aria-label="failed">
                ✗
              </span>
            ) : (
              <span role="img" aria-label="done">
                ✓
              </span>
            )}{" "}
            {stageLabel(r.stage)}
            {r.note && <span className="muted"> · {r.note}</span>}
          </span>
          <span className="spacer" />
          <span className="mono">{r.seconds === null ? "" : fmtDuration(r.seconds)}</span>
        </div>
      ))}
    </>
  );
}
